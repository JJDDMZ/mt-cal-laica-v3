"""checkpoint、恢复与轻量部署导出（技术方案 §10/§13，任务 T8）。

要求：
- 训练 checkpoint 保存 β/φ/ψ/Q/target、所有 optimizers、动作注册表、buffer 与
  immutable 行为字段、pending 状态、RNG、rollout 快照、配置、阶段/步数与映射版本；
- 恢复时 variant / schema / 映射结构 / schedule 不匹配就**拒绝**，不能只靠网络维度
  恰好相同放行；
- 部署导出只留推理必需组件；同一 ``policy.pt`` 在两游戏上 hash 必须一致。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import tempfile
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import Config, SCHEMA_VERSION, run_dir
from .envs import features as F
from .models.continuous import ContinuousModel
from .models.discrete import DiscreteModel
from .replay import Episode, ReplayBuffer, Transition

CHECKPOINT_FORMAT = 3


class CheckpointError(RuntimeError):
    """身份不匹配或结构损坏：必须失败，不能"尽量加载"。"""


# ---------------------------------------------------------------- RNG
def rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def _validate_rng_state(state: Mapping[str, Any]) -> None:
    """Validate every process RNG stream using isolated probe generators."""
    required = {"python", "numpy", "torch"}
    if not isinstance(state, Mapping):
        raise CheckpointError("checkpoint rng 必须为映射")
    if not required.issubset(state):
        raise CheckpointError(f"checkpoint rng 缺少字段: {sorted(required - set(state))}")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        torch.Generator(device="cpu").set_state(state["torch"])
    except Exception as exc:
        raise CheckpointError("checkpoint rng 含非法 Python/NumPy/Torch 状态") from exc


def _validate_v3_payload(
    payload: Mapping[str, Any], *, optimizers: Optional[Mapping[str, torch.optim.Optimizer]],
    restore_replay: bool, restore_rng: bool,
) -> None:
    """Fail closed on malformed format-3 state before applying any state."""
    model_rng = payload.get("model_rng")
    if payload.get("variant") == "continuous":
        expected = {"beta", "phi", "psi"}
        if not isinstance(model_rng, Mapping) or set(model_rng) != expected:
            raise CheckpointError(
                "checkpoint 字段 model_rng 非法：continuous checkpoint 必须恰含 beta/phi/psi"
            )
        for name, state in model_rng.items():
            if not isinstance(state, torch.Tensor):
                raise CheckpointError(f"checkpoint 字段 model_rng.{name} 非法：必须为 Tensor")
            try:
                torch.Generator(device="cpu").set_state(state)
            except Exception as exc:
                raise CheckpointError(
                    f"checkpoint 字段 model_rng.{name} 非法：不是有效的 Torch generator 状态"
                ) from exc
    elif "model_rng" not in payload:
        raise CheckpointError("checkpoint 缺少字段 model_rng")

    if "replay_rng" not in payload:
        raise CheckpointError("checkpoint 缺少字段 replay_rng")
    try:
        probe_rng = np.random.default_rng()
        probe_rng.bit_generator.state = payload["replay_rng"]
    except Exception as exc:
        raise CheckpointError("checkpoint 字段 replay_rng 非法：不是有效的 NumPy bit_generator 状态") from exc

    checkpoint_optimizers = payload.get("optimizers")
    if not isinstance(checkpoint_optimizers, Mapping):
        raise CheckpointError("checkpoint 字段 optimizers 非法：必须为映射")
    if optimizers is not None:
        for name in optimizers:
            if name not in checkpoint_optimizers:
                raise CheckpointError(f"checkpoint 缺少 optimizer {name!r}")
            state = checkpoint_optimizers[name]
            if (not isinstance(state, Mapping)
                    or not isinstance(state.get("state"), Mapping)
                    or not isinstance(state.get("param_groups"), list)):
                raise CheckpointError(
                    f"checkpoint optimizer {name!r} 非法：缺少 state/param_groups 映射结构"
                )
            current_groups = optimizers[name].param_groups
            saved_groups = state["param_groups"]
            if len(saved_groups) != len(current_groups) or any(
                not isinstance(saved_group, Mapping)
                or not isinstance(saved_group.get("params"), list)
                or len(saved_group["params"]) != len(current_group["params"])
                for saved_group, current_group in zip(saved_groups, current_groups)
            ):
                raise CheckpointError(
                    f"checkpoint optimizer {name!r} 非法：param_groups 结构不匹配"
                )

    if not isinstance(payload.get("trainer_state"), Mapping):
        raise CheckpointError("checkpoint 字段 trainer_state 非法：必须为映射")
    if restore_replay and not isinstance(payload.get("replay"), Mapping):
        raise CheckpointError("checkpoint 字段 replay 非法：必须为映射")
    if restore_rng and not isinstance(payload.get("rng"), Mapping):
        raise CheckpointError("checkpoint 字段 rng 非法：必须为映射")


# ---------------------------------------------------------------- replay 序列化
def replay_to_state(buf: ReplayBuffer) -> Dict[str, Any]:
    def ep_to_dict(e: Episode) -> Dict[str, Any]:
        return {
            "episode_id": e.episode_id,
            "game": e.game,
            "completed": e.completed,
            "termination_reason": e.termination_reason,
            "mc_filled": e.mc_filled,
            "transitions": [
                {
                    "sample_id": t.sample_id, "task_id": t.task_id,
                    "episode_id": t.episode_id, "request_id": t.request_id,
                    "x": t.x, "e_behavior": t.e_behavior, "global_token": t.global_token,
                    "local_action": t.local_action, "reward_train": t.reward_train,
                    "next_x": t.next_x, "score_raw": t.score_raw,
                    "terminated": t.terminated, "truncated": t.truncated,
                    "termination_reason": t.termination_reason,
                    "behavior_id": t.behavior_id, "behavior_kernel": t.behavior_kernel,
                    "behavior_mapping_version": t.behavior_mapping_version,
                    "log_b_action": t.log_b_action,
                    "reference_return": t.reference_return, "ref_valid": t.ref_valid,
                    "env_hash": t.env_hash, "obs_schema": t.obs_schema,
                    "reward_schema": t.reward_schema,
                    "collection_split": t.collection_split, "clock_mode": t.clock_mode,
                }
                for t in e.transitions
            ],
        }

    return {
        "config": {
            "per_task_capacity": buf.per_task_capacity,
            "max_stored_transitions": buf.max_stored_transitions,
            "gamma": buf.gamma,
            "batch_size": buf.batch_size,
            "active_task_history_fraction": buf.active_task_history_fraction,
        },
        "historical": {str(k): [ep_to_dict(e) for e in v] for k, v in buf.historical.items()},
        "online": {str(k): [ep_to_dict(e) for e in v] for k, v in buf.online.items()},
        "pending": {f"{k[0]}:{k[1]}": ep_to_dict(v) for k, v in buf.pending.items()},
        "compat": dict(buf.compat),
        "next_sample_id": buf._next_sample_id,
        "evictions": buf.evictions,
        "rejected_finish": buf.rejected_finish,
    }


def replay_from_state(state: Mapping[str, Any], *, seed: int = 0) -> ReplayBuffer:
    cfg = state["config"]
    buf = ReplayBuffer(seed=seed, **cfg)

    def dict_to_ep(d: Mapping[str, Any]) -> Episode:
        e = Episode(episode_id=d["episode_id"], game=d["game"],
                    completed=d["completed"], termination_reason=d["termination_reason"],
                    mc_filled=d["mc_filled"])
        for td in d["transitions"]:
            if "clock_mode" not in td or td["clock_mode"] not in ("lockstep-v1", "realtime-v1"):
                raise CheckpointError("checkpoint replay transition 缺少合法 clock_mode")
            e.transitions.append(Transition(**td))
        return e

    for k, v in state["historical"].items():
        buf.historical[int(k)] = [dict_to_ep(d) for d in v]
    for k, v in state["online"].items():
        buf.online[int(k)] = [dict_to_ep(d) for d in v]
    for key, d in state["pending"].items():
        task, ep_id = (int(x) for x in key.split(":"))
        buf.pending[(task, ep_id)] = dict_to_ep(d)
    buf.compat = dict(state["compat"])
    buf._next_sample_id = int(state["next_sample_id"])
    buf.evictions = int(state["evictions"])
    buf.rejected_finish = int(state["rejected_finish"])
    return buf


# ---------------------------------------------------------------- 保存
def save_checkpoint(
    path: str | Path,
    *,
    cfg: Config,
    model: Any,
    optimizers: Mapping[str, torch.optim.Optimizer],
    replay: ReplayBuffer,
    trainer_state: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "identity_hash": cfg.identity_hash(),
        "config": cfg.to_dict(),
        "variant": cfg.variant,
        "schedule": cfg.schedule,
        "mode": cfg.mode,
        "seed": cfg.seed,
        "game": cfg.game,
        "test_only": cfg.test_only,
        "stage_schedule": cfg.stage_schedule,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_rng": model.generator_state() if hasattr(model, "generator_state") else None,
        "replay_rng": replay.get_rng_state(),
        "codebook_rows": _codebook_rows(model),
        # φ 的动作注册表：权重能 load，但"哪些 token 已注册"是 _token_meta 的
        # 运行时状态，必须显式保存，否则 resume 后 φ 会以为什么都没注册。
        "phi_token_meta": ([list(t) for t in getattr(model, "phi")._token_meta]
                           if hasattr(model, "phi") else None),
        "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
        "replay": replay_to_state(replay),
        "trainer_state": dict(trainer_state),
        "rng": rng_state(),
    }
    if extra:
        payload["extra"] = dict(extra)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{p.name}.", suffix=".tmp", dir=p.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, p)
    finally:
        temporary_path.unlink(missing_ok=True)
    return p


def _codebook_rows(model: Any) -> Optional[Dict[str, Any]]:
    """ParameterList 不会进入 state_dict，需要显式保存 D 的码本行。"""
    cb = getattr(model, "codebook", None)
    if cb is None:
        return None
    return {
        "token_ids": list(cb.token_ids),
        "rows": [r.detach().cpu() for r in cb.rows],
    }


# ---------------------------------------------------------------- 加载
def load_checkpoint(
    path: str | Path,
    *,
    cfg: Optional[Config] = None,
    model: Optional[Any] = None,
    optimizers: Optional[Mapping[str, torch.optim.Optimizer]] = None,
    restore_replay: bool = True,
    restore_rng: bool = True,
    strict_identity: bool = True,
    pre_apply_validator: Optional[
        Callable[[Mapping[str, Any], Optional[ReplayBuffer], Mapping[str, Any]], None]
    ] = None,
) -> Dict[str, Any]:
    """加载并（可选）校验身份。身份不匹配直接抛 CheckpointError。"""
    p = Path(path)
    if not p.is_file():
        raise CheckpointError(f"checkpoint 不存在: {p}")
    payload = torch.load(p, map_location="cpu", weights_only=False)

    fmt = int(payload.get("checkpoint_format", 0))
    if fmt != CHECKPOINT_FORMAT:
        raise CheckpointError(
            f"checkpoint_format 不匹配：期望 {CHECKPOINT_FORMAT}，得到 {fmt}；"
            "旧实验仅用于分析，不能无损续训"
        )
    if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise CheckpointError(
            f"schema_version 不匹配（拒绝 v1 恢复到 v2）："
            f"期望 {SCHEMA_VERSION}，得到 {payload.get('schema_version')}"
        )

    if cfg is not None:
        if payload["variant"] != cfg.variant:
            raise CheckpointError(f"variant 不匹配: {payload['variant']!r} vs {cfg.variant!r}")
        if payload["mode"] != cfg.mode:
            raise CheckpointError(f"mode 不匹配: {payload['mode']!r} vs {cfg.mode!r}")
        if payload.get("test_only") != cfg.test_only:
            raise CheckpointError(
                f"test_only 标记不匹配: {payload.get('test_only')} vs {cfg.test_only}"
            )
        if list(payload["stage_schedule"]) != list(cfg.stage_schedule):
            raise CheckpointError(
                "stage_schedule 不匹配：拒绝按不同阶段表恢复"
            )
        if strict_identity and payload["identity_hash"] != cfg.identity_hash():
            raise CheckpointError(
                "identity_hash 不匹配：配置/schema/schedule 与 checkpoint 不一致，"
                "不能只靠网络维度恰好相同放行"
            )

    _validate_v3_payload(
        payload, optimizers=optimizers, restore_replay=restore_replay,
        restore_rng=restore_rng,
    )

    # Prepare every state that can be decoded or validated without mutating the
    # destination model, optimizer, process RNG, or caller-owned trainer.
    prepared_replay: Optional[ReplayBuffer] = None
    if restore_replay:
        try:
            prepared_replay = replay_from_state(payload["replay"], seed=cfg.seed if cfg else 0)
            prepared_replay.set_rng_state(payload["replay_rng"])
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"checkpoint replay 无法完整解码: {exc}") from exc
    if restore_rng:
        _validate_rng_state(payload["rng"])
    if pre_apply_validator is not None:
        try:
            pre_apply_validator(payload["trainer_state"], prepared_replay,
                                payload.get("extra", {}))
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"checkpoint pre-apply validation failed: {exc}") from exc

    # Probe model/optimizer loading on isolated copies so shape/state failures do
    # not leave the caller half-restored during the apply phase.
    if model is not None:
        try:
            model_probe = copy.deepcopy(model)
            _load_model(model_probe, payload)
            if payload.get("model_rng") is not None and hasattr(model_probe, "set_generator_state"):
                model_probe.set_generator_state(payload["model_rng"])
        except CheckpointError:
            raise
        except Exception as exc:
            raise CheckpointError(f"checkpoint model state 无法应用: {exc}") from exc
    if optimizers is not None:
        try:
            for name, opt in optimizers.items():
                probe = copy.deepcopy(opt)
                probe.load_state_dict(payload["optimizers"][name])
        except Exception as exc:
            raise CheckpointError(f"checkpoint optimizer state 无法应用: {exc}") from exc

    if model is not None:
        _load_model(model, payload)
        if payload.get("model_rng") is not None and hasattr(model, "set_generator_state"):
            model.set_generator_state(payload["model_rng"])
    if optimizers is not None:
        for name, opt in optimizers.items():
            if name not in payload["optimizers"]:
                raise CheckpointError(f"checkpoint 缺少 optimizer {name!r}")
            opt.load_state_dict(payload["optimizers"][name])
    if restore_rng and "rng" in payload:
        set_rng_state(payload["rng"])

    out: Dict[str, Any] = {
        "config": payload["config"],
        "trainer_state": payload.get("trainer_state", {}),
        "identity_hash": payload["identity_hash"],
        "extra": payload.get("extra", {}),
    }
    if restore_replay:
        assert prepared_replay is not None
        out["replay"] = prepared_replay
    return out


def _load_model(model: Any, payload: Mapping[str, Any]) -> None:
    """加载模型参数。

    **φ 的形状处理（跨游戏 resume 的关键）**：
    阶段 1 结束时 φ 只注册了 F 的 2 个 token（输出层 4×2），
    阶段 2 接入 T 后 φ 变成 4 个 token（输出层 4×4）。
    直接用 ``load_state_dict`` 会因形状不匹配直接抛错。

    正确做法（§6.4 规则 2）：**先把 old 的行逐个拷进去**（不是整层 load），
    再让调用方去注册新游戏的 token（新行随机初始化）。
    """
    state = dict(payload["model_state"])
    if isinstance(model, DiscreteModel):
        rows = payload.get("codebook_rows")
        if rows is None:
            raise CheckpointError("D 的 checkpoint 缺少 codebook_rows")
        with torch.no_grad():
            # D：码本按 token 逐行拷贝，已学游戏行保持不变
            for tok, row in zip(rows["token_ids"], rows["rows"]):
                if tok in model.codebook.token_ids:
                    model.codebook.rows[model.codebook.row_for(tok)].copy_(row)
        state.pop("codebook._registered", None)

    # 注册表类 buffer 由 _token_meta 派生，不参与 state_dict 拷贝
    # （否则 _token_ids 会因 token 数不同而形状冲突，而它本来就能重建）
    for derived in [k for k in state if k.endswith("_token_ids") or k.endswith("_registered")]:
        state.pop(derived, None)

    # 形状不一致的张量（只会是 φ 的输出层）逐行拷贝，不整层 load
    row_wise: list = []
    for name, tensor in list(state.items()):
        target = dict(model.named_parameters()).get(name)
        if target is None:
            continue
        if tuple(target.shape) != tuple(tensor.shape):
            if name.startswith("phi.linear.") and tensor.dim() == 2 and target.dim() == 2:
                n = min(target.shape[0], tensor.shape[0])
                with torch.no_grad():
                    target[:n].copy_(tensor[:n])
                row_wise.append(name)
            elif name.startswith("phi.linear.") and tensor.dim() == 1 and target.dim() == 1:
                n = min(target.shape[0], tensor.shape[0])
                with torch.no_grad():
                    target[:n].copy_(tensor[:n])
                row_wise.append(name)
            else:
                raise CheckpointError(
                    f"参数 {name} 形状不匹配且不是 φ 的输出层: "
                    f"{tuple(target.shape)} vs {tuple(tensor.shape)}"
                )
            state.pop(name)
        elif "phi.linear" in name and name.endswith("weight"):
            # 形状恰好相同（两次都是 4 行）也走整层拷贝即可
            pass

    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "codebook" not in k
               and not k.endswith("_token_ids") and not k.endswith("_registered")]
    if missing:
        raise CheckpointError(f"checkpoint 缺少参数: {missing}")
    if unexpected:
        raise CheckpointError(f"checkpoint 含未知参数: {unexpected}")

    # C 的动作注册表来自 checkpoint（权重能 load，但"哪些 token 已注册"是运行时状态）
    phi = getattr(model, "phi", None)
    meta = payload.get("phi_token_meta")
    if phi is not None and meta is not None:
        phi._token_meta = [(int(t), str(g), int(a)) for t, g, a in meta]
        phi._token_ids = torch.tensor([t for t, _, _ in phi._token_meta], dtype=torch.long)
        if getattr(phi, "_registered", None) is not None:
            phi._registered.zero_()
            for t, _, _ in phi._token_meta:
                if t < phi._registered.numel():
                    phi._registered[t] = True


# ---------------------------------------------------------------- 部署导出
def export_policy(
    path: str | Path,
    *,
    cfg: Config,
    model: Any,
    inference_rule: Optional[Mapping[str, Any]] = None,
) -> Path:
    """只留推理必需组件（§13）。

    C 导出 β/φ；D 导出 actor/E；都带 schema、注册表与推理规则。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rule = dict(inference_rule or {})
    rule.setdefault("primary_policy_mode", cfg.eval["primary_policy_mode"])
    payload: Dict[str, Any] = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "variant": cfg.variant,
        "identity_hash": cfg.identity_hash(),
        "obs_schema": cfg.env["obs_schema"],
        "reward_schema": cfg.env["reward_schema"],
        "horizon": cfg.horizon,
        "obs_dim": F.OBS_DIM,
        "action_registry": [list(x) for x in F.ACTION_REGISTRY],
        "action_mask": {g: F.action_mask(g).tolist() for g in F.GAMES},
        "inference_rule": rule,
    }
    if cfg.variant == "continuous":
        payload["components"] = ["beta", "phi"]
        payload["latent_dim"] = cfg.continuous["latent_dim"]
        payload["beta_state"] = {k: v.detach().cpu() for k, v in model.beta.state_dict().items()}
        payload["phi_state"] = {k: v.detach().cpu() for k, v in model.phi.state_dict().items()}
        payload["registered_tokens"] = list(model.phi.tokens)
        payload["registered"] = {str(k): list(v) for k, v in model.phi.registered.items()}
    elif cfg.variant == "discrete":
        payload["components"] = ["actor", "codebook"]
        payload["actor_state"] = {k: v.detach().cpu() for k, v in model.actor.state_dict().items()}
        payload["codebook"] = _codebook_rows(model)
        payload["registered_games"] = list(model.registered_games)
    else:
        raise CheckpointError(f"未知 variant: {cfg.variant!r}")
    torch.save(payload, p)
    return p


def policy_hash(path: str | Path) -> str:
    """同一 policy.pt 在两个游戏上必须 hash 一致（§12.5）。"""
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_policy_into(model: Any, path: str | Path) -> Dict[str, Any]:
    """把部署文件加载回模型（评测用），不改动训练组件。"""
    p = Path(path)
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise CheckpointError("部署文件的 schema_version 不匹配")
    if payload["variant"] == "continuous":
        model.beta.load_state_dict(payload["beta_state"], strict=True)
        model.phi.load_state_dict(payload["phi_state"], strict=True)
    elif payload["variant"] == "discrete":
        model.actor.load_state_dict(payload["actor_state"], strict=True)
        rows = payload["codebook"]
        if list(rows["token_ids"]) != list(model.codebook.token_ids):
            raise CheckpointError("部署文件的码本结构不匹配")
        with torch.no_grad():
            for i, row in enumerate(rows["rows"]):
                model.codebook.rows[i].copy_(row)
    else:
        raise CheckpointError(f"未知 variant: {payload['variant']!r}")
    return payload
