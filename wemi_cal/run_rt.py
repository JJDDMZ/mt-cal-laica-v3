"""用原版（实时）协议跑真实训练：``python -m wemi_cal.run_rt``。

按用户确认的方案：**不做 lockstep 改动，按原来训练成功的方式训**。

与原版 ``wemi_mt/train.py`` 的区别只在"学什么"：
- 原版用 SB3 的 MaskablePPO 学一个离散策略；
- 这里学本方案的 C 主方案（连续 β + 可学习 φ + 逆动力学 ψ + Q 校准），
  环境侧完全沿用原版实时协议（客户端自带 40ms 物理 + 100ms 回包，保持原速）。

环境侧要点：
- Python 监听 **7911**（原版客户端硬编码端口）→ **游戏 JS 零改动**；
- 一次只开一个游戏项目（原版限制，两个游戏都占 7911）。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import numbers
import os
import sys
import time
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import checkpoint as CK
from .config import GAMES, ConfigError, validate_config
from .envs import features as F
from .learner import ContinuousLearner, LearnerConfig, validate_update_stats
from .models.continuous import ContinuousModel
from .models.discrete import DiscreteModel, adapt_discrete_representation
from .learner_discrete import DiscreteLearner, DiscreteLearnerConfig
from .real_env import LEGACY_PORT, RealTimeGameEnv
from .replay import ReplayError, make_transition
from .repr_train import adapt_representation, make_batches, split_by_episode
from .schedule import SubPhase, assert_frozen, set_trainable

ENV_HASH = {
    "Flappybird": "flappybird-realtime-7911",
    "Timberman": "timberman-realtime-7911",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_run_root(path: str | Path) -> Path:
    runs_root = (REPO_ROOT / "runs_v3").resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(runs_root)
    except ValueError as exc:
        raise SystemExit(f"--run-root 必须位于仓库 runs_v3/ 下: {resolved}") from exc
    return resolved


def _publish_json_no_overwrite(path: Path, payload: Dict[str, Any]) -> None:
    temporary_path: Optional[Path] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                              dir=path.parent)
        temporary_path = Path(temporary_name)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise SystemExit(f"结果文件已存在，拒绝覆盖: {path}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class LegacyBundle:
    """让 RealTrainer 能用上原版实时环境（只暴露它需要的东西）。"""

    def __init__(self, env: RealTimeGameEnv) -> None:
        self.legacy = env
        self.env = env
        self.session = None      # 原版协议没有 session 概念
        self.conn = None


class RealTimeTrainer:
    """C 主方案在实时（原版）环境上的训练循环。

    与 ``real_train.RealTrainer`` 的区别：环境是原版实时接口，重连走
    ``RealTimeGameEnv.reconnect``（原版没有 session，只能重建服务端）。
    """

    def __init__(self, cfg: Config, game: str, legacy: RealTimeGameEnv, *,
                 run_root: Path, seed: int = 0) -> None:
        self.cfg = cfg
        self.game = game
        self.legacy = legacy
        self.env = legacy
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_root / "train_log.jsonl"
        self.variant = cfg.variant

        if cfg.variant == "continuous":
            self.model = ContinuousModel(seed=seed)
            self.learner = ContinuousLearner(self.model, LearnerConfig(
                gamma=cfg.gamma, actor_lr=float(cfg.train["actor_lr"]),
                critic_lr=float(cfg.train["critic_lr"]),
                entropy_alpha=float(cfg.train["entropy_alpha"]),
                lambda_cal=cfg.lambda_cal, tau=float(cfg.train["tau"]),
                grad_clip=float(cfg.train["grad_clip"]),
                target_samples=int(cfg.train["target_samples"]),
                calibration_samples=int(cfg.train["calibration_samples"]),
                calibration_mode=cfg.calibration_mode))
        elif cfg.variant == "discrete":
            self.model = DiscreteModel(seed=seed)
            self.learner = DiscreteLearner(self.model, DiscreteLearnerConfig(
                gamma=cfg.gamma, actor_lr=float(cfg.train["actor_lr"]),
                critic_lr=float(cfg.train["critic_lr"]),
                entropy_alpha=float(cfg.train["entropy_alpha"]),
                lambda_cal=cfg.lambda_cal, tau=float(cfg.train["tau"]),
                grad_clip=float(cfg.train["grad_clip"]),
                calibration_mode=cfg.calibration_mode))
        else:
            raise ValueError(f"未知 variant: {cfg.variant!r}")
        from .replay import ReplayBuffer
        self.replay = ReplayBuffer(
            per_task_capacity=int(cfg.replay["per_task_capacity"]),
            max_stored_transitions=int(cfg.replay["max_stored_transitions"]),
            gamma=cfg.gamma, batch_size=int(cfg.replay["batch_size"]),
            active_task_history_fraction=float(cfg.replay["active_task_history_fraction"]),
            seed=seed,
        )
        self.first_data: List[Dict[str, Any]] = []
        self.mapping_version = 0
        self.offline_done = False
        self.completed_tasks: set[int] = set()
        self.stats: Dict[str, Any] = {
            "random_steps": 0, "online_steps": 0, "learner_updates": 0,
            "repr_updates": 0, "offline_updates": 0, "episodes": [], "latencies": []}
        self.stage_game = game
        self.stage_phase = "pre_online"
        self.stage_target_online_steps: Optional[int] = None
        self.next_episode_ids = {0: 0, 1: 0}
        self.clock_mode = str(cfg.env["clock_mode"])

    def _allocate_episode_id(self) -> int:
        task = F.task_id(self.game)
        episode_id = self.next_episode_ids[task]
        self.next_episode_ids[task] += 1
        return episode_id

    # ---- 日志 ----
    def log(self, rec: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def finalize_stage(self) -> None:
        """成功完成当前阶段后固化其 replay 数据。"""
        task = F.task_id(self.game)
        if any(key[0] == task for key in self.replay.pending):
            raise RuntimeError(f"task {task} 仍有 pending episode，拒绝固化")
        self.replay.freeze_historical(task)
        self.completed_tasks.add(task)
        self.log({"event": "stage_finalized", "game": self.game,
                  "task": task, "replay": self.replay.diagnostics()})

    # ---- 首次随机采集 ----
    def collect_first_data(self, steps: int) -> None:
        legal = F.legal_tokens(self.game)
        n = 0
        while n < steps:
            x, info = self.env.reset(seed=int(self.rng.integers(0, 1 << 30)))
            ep_id = self._allocate_episode_id()
            while n < steps:
                u = self.rng.normal(size=2).astype(np.float32)
                e = np.tanh(u).astype(np.float32)
                token = int(legal[n % len(legal)])
                log_b = -float(np.log(len(legal)))
                local = F.local_action(self.game, token)
                nx, r, term, trunc, sinfo = self.env.step(local)
                n += 1
                self.first_data.append({
                    "x": x, "next_x": nx, "e_behavior": e, "global_token": token,
                    "reward_train": float(r), "log_b_action": log_b, "episode_id": ep_id})
                self.replay.add_confirmed(make_transition(
                    buffer=self.replay, game=self.game, episode_id=ep_id, request_id=n,
                    x=x, next_x=nx, e_behavior=e, global_token=token,
                    reward_train=float(r), log_b_action=log_b,
                    behavior_kernel="uniform", behavior_mapping_version=self.mapping_version,
                    terminated=term, truncated=trunc,
                    termination_reason=sinfo.termination_reason,
                    env_hash=ENV_HASH[self.game], clock_mode=self.clock_mode), game=self.game)
                x = nx
                if term or trunc:
                    self.replay.finish_episode(
                        game=self.game, episode_id=ep_id, terminated=term,
                        truncated=trunc, termination_reason=sinfo.termination_reason)
                    break
                if n >= steps:
                    self.replay.finish_episode(
                        game=self.game, episode_id=ep_id, terminated=False,
                        truncated=True, termination_reason="none")
                    break
            self.stats["random_steps"] = n
            if n % 200 == 0:
                print(f"[collect] {n}/{steps}", flush=True)
        counts: Dict[int, int] = {}
        for r in self.first_data:
            counts[r["global_token"]] = counts.get(r["global_token"], 0) + 1
        self.log({"event": "first_data_done", "game": self.game, "steps": n,
                  "token_counts": counts})
        print(f"[collect] 完成 {n} 步，动作分布 {counts}", flush=True)

    # ---- 表示适配 ----
    def adapt(self, updates: int) -> int:
        x = torch.tensor(np.stack([r["x"] for r in self.first_data]), dtype=torch.float32)
        nx = torch.tensor(np.stack([r["next_x"] for r in self.first_data]), dtype=torch.float32)
        tok = torch.tensor([int(r["global_token"]) for r in self.first_data], dtype=torch.long)
        eps = [int(r["episode_id"]) for r in self.first_data]
        tr_idx, _ = split_by_episode(eps, val_fraction=0.2, seed=self.seed)
        if self.variant == "continuous":
            if self.game not in self.model.phi.registered_games():
                if self.model.phi.num_tokens == 0:
                    self.model.register_game_as_first(self.game)
                else:
                    self.model.register_game(self.game)
            batches = make_batches(x=x[tr_idx], next_x=nx[tr_idx], token=tok[tr_idx],
                                   game=self.game,
                                   batch_size=int(self.cfg.replay["batch_size"]), seed=self.seed)
            diag = adapt_representation(
                self.model, batches, updates=updates,
                lr=float(self.cfg.continuous["repr_lr"]),
                kl_weight=float(self.cfg.continuous["kl_weight"]),
                grad_clip=float(self.cfg.train["grad_clip"]))
            coverage = self.model.phi.coverage_grid(self.game, res=41)
            rec = {"event": "adapt_done", "game": self.game,
                   "mapping_version": self.mapping_version + 1,
                   "variant": "continuous",
                   "train_accuracy": diag.accuracy, "train_nll": diag.nll, "kl": diag.kl,
                   "per_action_accuracy": diag.per_action_accuracy,
                   "coverage_counts": coverage["counts"],
                   "all_actions_reachable": coverage["all_actions_reachable"]}
        else:
            if self.game not in self.model.registered_games:
                self.model.register_game(self.game)
            d = adapt_discrete_representation(
                self.model, data=[(x[tr_idx], nx[tr_idx], tok[tr_idx])],
                game=self.game, updates=updates,
                lr=float(self.cfg.continuous["repr_lr"]))
            rec = {"event": "adapt_done", "game": self.game,
                   "mapping_version": self.mapping_version + 1,
                   "variant": "discrete",
                   "ce": d.ce, "sep": d.sep, "train_accuracy": d.accuracy}
        self.mapping_version += 1
        self.stats["repr_updates"] = updates
        self.log(rec)
        print(f"[adapt] {json.dumps(rec, ensure_ascii=False)}", flush=True)
        return self.mapping_version

    # ---- 初始离线 ----
    def make_learner_batch(self, active_task: int) -> Dict[str, Any]:
        if active_task == F.task_id("Timberman") and 0 in self.completed_tasks:
            if self.replay.batch_size != 128:
                raise ReplayError(
                    f"Timberman 联合 batch size 必须为 128，得到 {self.replay.batch_size}"
                )
            transitions = (
                self.replay.sample_bucket(0, "historical", 64)
                + self.replay.sample_bucket(1, "online", 64)
            )
            self.replay.assert_task_counts(transitions, {0: 64, 1: 64})
        else:
            transitions = self.replay.sample(active_task)
        batch_task_counts = self.replay.task_counts(transitions)
        batch = self.replay.stack(
            transitions, game_of_task={0: "Flappybird", 1: "Timberman"})
        batch["games"] = [GAMES[int(t)] for t in batch["task_id"].tolist()]
        batch["batch_task_counts"] = batch_task_counts
        return batch

    def initial_offline(self, updates: int) -> int:
        if self.offline_done:
            raise RuntimeError("initial_offline 在整个实验中只能执行一次")
        if updates <= 0:
            raise ValueError("initial_offline updates 必须为正数")
        task = F.task_id(self.game)
        set_trainable(self.model, SubPhase.INITIAL_OFFLINE)
        assert_frozen(self.model, SubPhase.INITIAL_OFFLINE, game=self.game)
        completed = 0
        for _ in range(updates):
            batch = self.make_learner_batch(task)
            st = self.learner.update(batch)
            try:
                validate_update_stats(
                    st, expected_batch_size=int(batch["x"].shape[0]),
                    variant=self.variant,
                )
            except RuntimeError as exc:
                raise RuntimeError(f"{self.game} initial offline 更新无效: {exc}") from exc
            completed += 1
        self.offline_done = True
        self.stats["offline_updates"] += completed
        self.log({"event": "initial_offline_done", "game": self.game,
                  "updates": completed})
        return completed

    # ---- 在线 ----
    def online(self, steps: int, *, update_every: int, save_every: int = 5000,
               reconnect_timeout_s: float = 900.0) -> None:
        if self.stage_game != self.game:
            self.stage_game = self.game
            self.stage_phase = "online"
            self.stats = {"random_steps": 0, "online_steps": 0, "learner_updates": 0,
                          "repr_updates": 0, "offline_updates": 0, "episodes": [],
                          "latencies": []}
        self.stage_phase = "online"
        self.stage_target_online_steps = int(steps)
        set_trainable(self.model, SubPhase.ONLINE)
        assert_frozen(self.model, SubPhase.ONLINE, game=self.game)
        mix = float(self.cfg.continuous["decoder_behavior_uniform_mix"])
        done = int(self.stats["online_steps"])
        if done > steps:
            raise CK.CheckpointError(
                f"已完成 online 步数 {done} 超过阶段目标 {steps}，拒绝续训"
            )
        x = None
        ep_id = None
        while done < steps:
            if x is None:
                try:
                    x, info = self.env.reset(seed=int(self.rng.integers(0, 1 << 30)))
                except Exception as exc:
                    print(f"[warn] reset 失败({type(exc).__name__})，重连…", flush=True)
                    self.log({"event": "reset_failed", "error": str(exc)[:200], "step": done})
                    self.env.reconnect()
                    continue
                ep_id = self._allocate_episode_id()

            xb = torch.as_tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0)
            if self.variant == "continuous":
                with torch.no_grad():
                    s = self.model.beta.rsample(xb, samples=1)
                    e = s.e_squeezed
                    logb_all = self.model.phi.behavior_log_probs(e, self.game, uniform_mix=mix)
                    probs = torch.softmax(logb_all, dim=-1)
                    token = int(torch.multinomial(probs.clamp_min(0.0), 1,
                                                  generator=self.model.phi.gen).item())
                    log_b = float(self.model.phi.behavior_log_prob_of(
                        e, self.game, torch.tensor([token]), uniform_mix=mix).item())
                    e_np = e[0].numpy()
            else:
                # D：actor 直接给离散概率，token 与动作是固定翻译（§16.1）
                with torch.no_grad():
                    token = int(self.model.actor.sample_token(
                        xb, self.model.codebook, self.game).item())
                log_b = 0.0
                e_np = np.zeros(2, dtype=np.float32)
            local = F.local_action(self.game, token)
            try:
                nx, r, term, trunc, sinfo = self.env.step(local)
            except Exception as exc:
                print(f"[warn] step 失败({type(exc).__name__})，重连…", flush=True)
                self.log({"event": "step_failed", "error": str(exc)[:200], "step": done})
                self.env.reconnect()
                x = None
                continue

            done += 1
            self.stats["online_steps"] = done
            self.stats["latencies"].append(sinfo.latency_s)
            self.replay.add_confirmed(make_transition(
                buffer=self.replay, game=self.game, episode_id=ep_id, request_id=done,
                x=x, next_x=nx, e_behavior=e_np, global_token=token,
                reward_train=float(r), log_b_action=log_b,
                behavior_kernel="decoder_exploration",
                behavior_mapping_version=self.mapping_version,
                terminated=term, truncated=trunc,
                termination_reason=sinfo.termination_reason,
                env_hash=ENV_HASH[self.game], clock_mode=self.clock_mode), game=self.game)
            x = nx

            if term or trunc:
                self.replay.finish_episode(
                    game=self.game, episode_id=ep_id, terminated=term,
                    truncated=trunc, termination_reason=sinfo.termination_reason)
                self.stats["episodes"].append({
                    "episode_id": ep_id, "score_raw": sinfo.score_raw,
                    "return_train": sinfo.episode_return_train,
                    "length": sinfo.steps_taken, "reason": sinfo.termination_reason})
                x = None
                print(f"[online] ep{ep_id} score={sinfo.score_raw:.0f} "
                      f"len={sinfo.steps_taken} step={done}/{steps}", flush=True)

            if done % update_every == 0 and self.replay.total_transitions() >= 8:
                batch = self.make_learner_batch(F.task_id(self.game))
                st = self.learner.update(batch)
                try:
                    validate_update_stats(
                        st, expected_batch_size=int(batch["x"].shape[0]),
                        variant=self.variant,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"{self.game} step={done} learner update 无效: {exc}"
                    ) from exc
                self.stats["learner_updates"] += 1
                if self.stats["learner_updates"] % 50 == 0:
                    record = {
                        "event": "learner_update", "game": self.game, "step": done,
                        "batch_task_counts": batch["batch_task_counts"],
                        "td_loss": st.td_loss,
                        "calibration_regularizer": st.cal_regularizer,
                        "actor_loss": st.actor_loss,
                        "reference_coverage": st.reference_coverage,
                        "grad_norm_q1": st.grad_norm_q1,
                        "grad_norm_q2": st.grad_norm_q2,
                        "grad_norm_actor": st.grad_norm_actor,
                    }
                    if self.variant == "continuous":
                        record.update({
                            "q_minus_reference_frac": st.q_minus_reference_frac,
                            "w_mean": st.w_mean,
                            "w_max": st.w_max,
                            "ess_over_batch": st.ess_over_batch,
                        })
                    self.log(record)
                    if self.variant == "continuous":
                        tail = f"w_max={st.w_max:.2f} ess={st.ess_over_batch:.2f}"
                    else:
                        tail = "(D: unweighted)"
                    print(f"[online] step={done}/{steps} upd={self.stats['learner_updates']} "
                          f"td={st.td_loss:.4f} actor={st.actor_loss:.4f} {tail}", flush=True)

            if save_every > 0 and done % save_every == 0:
                try:
                    self.save(self.run_root / "ckpt_latest.pt", stage=f"{self.game}@{done}")
                    self.log({"event": "periodic_checkpoint", "step": done})
                except Exception as exc:
                    print(f"[warn] 周期存盘失败: {exc}", flush=True)

        # 用户给定的交互预算可能在 episode 中间耗尽。保留尾段，但不伪造 MC。
        task = F.task_id(self.game)
        if ep_id is not None and (task, ep_id) in self.replay.pending:
            tail = self.replay.pending[(task, ep_id)].transitions[-1]
            tail.truncated = True
            tail.termination_reason = "none"
            self.replay.finish_episode(
                game=self.game, episode_id=ep_id,
                terminated=False, truncated=True, termination_reason="none",
            )

    # ---- 保存 ----
    def save(self, path: Path, *, stage: str) -> Path:
        return CK.save_checkpoint(
            path, cfg=self.cfg, model=self.model,
            optimizers=self.optimizer_map(),
            replay=self.replay,
            trainer_state=self.trainer_state(stage=stage),
            extra={"stats": {k: v for k, v in self.stats.items() if k != "latencies"},
                   "latency": self._latency()})

    def optimizer_map(self) -> Dict[str, torch.optim.Optimizer]:
        if self.variant == "continuous":
            return {"q1": self.learner.opt_q1, "q2": self.learner.opt_q2,
                    "beta": self.learner.opt_beta}
        return {"q1": self.learner.opt_q1, "q2": self.learner.opt_q2,
                "actor": self.learner.opt_actor}

    def trainer_state(self, *, stage: str) -> Dict[str, Any]:
        return {
            "game": self.game,
            "stage": stage,
            "stage_game": self.stage_game,
            "stage_phase": self.stage_phase,
            "stage_target_online_steps": self.stage_target_online_steps,
            "stats": {k: v for k, v in self.stats.items() if k != "latencies"},
            "next_episode_ids": {str(k): int(v) for k, v in self.next_episode_ids.items()},
            "mapping_version": self.mapping_version,
            "offline_done": self.offline_done,
            "completed_tasks": sorted(self.completed_tasks),
            "trainer_rng": self.rng.bit_generator.state,
        }

    def restore(self, path: str | Path) -> None:
        prepared: Dict[str, Any] = {}

        def validate_before_apply(state, replay, extra) -> None:
            prepared.update(self._prepare_restore(state, replay, extra))

        CK.load_checkpoint(
            path, cfg=self.cfg, model=self.model,
            optimizers=self.optimizer_map(), restore_replay=True,
            pre_apply_validator=validate_before_apply,
        )
        candidate = prepared
        self.replay = candidate["replay"]
        self.mapping_version = candidate["mapping_version"]
        self.offline_done = candidate["offline_done"]
        self.completed_tasks = candidate["completed_tasks"]
        self.rng.bit_generator.state = candidate["trainer_rng"]
        self.next_episode_ids = candidate["next_episode_ids"]
        self.stage_game = candidate["stage_game_current"]
        self.stage_phase = candidate["stage_phase_current"]
        self.stage_target_online_steps = candidate["stage_target_current"]
        self.stats = candidate["stats_current"]

    def _prepare_restore(self, state, replay, extra) -> Dict[str, Any]:
        """Validate trainer-owned resume state without mutating this trainer."""
        if not isinstance(state, dict):
            raise CK.CheckpointError("trainer_state 必须为映射")
        required = {"mapping_version", "offline_done", "completed_tasks", "trainer_rng",
                    "stage_game", "stage_phase", "stage_target_online_steps", "stats",
                    "next_episode_ids"}
        missing = required - set(state)
        if missing:
            raise CK.CheckpointError(f"trainer_state 缺少字段: {sorted(missing)}")

        def nonnegative_int(value, label):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CK.CheckpointError(f"trainer_state.{label} 必须为非负整数")
            return int(value)

        def validate_episodes(value, label):
            if not isinstance(value, list):
                raise CK.CheckpointError(f"{label} 必须为列表")
            allowed_reasons = {"none", "collision", "ground", "blood", "horizon"}
            for index, episode in enumerate(value):
                prefix = f"{label}[{index}]"
                if not isinstance(episode, dict):
                    raise CK.CheckpointError(f"{prefix} 必须为映射")
                for field in ("episode_id", "score_raw", "return_train", "length", "reason"):
                    if field not in episode:
                        raise CK.CheckpointError(f"{prefix}.{field} 缺失")
                episode_id = episode["episode_id"]
                if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id < 0:
                    raise CK.CheckpointError(f"{prefix}.episode_id 必须为非负整数")
                for field in ("score_raw", "return_train"):
                    number = episode[field]
                    try:
                        finite = (not isinstance(number, bool)
                                  and isinstance(number, numbers.Real)
                                  and math.isfinite(float(number)))
                    except (OverflowError, ValueError):
                        finite = False
                    if not finite:
                        raise CK.CheckpointError(f"{prefix}.{field} 必须为有限数值")
                length = episode["length"]
                if isinstance(length, bool) or not isinstance(length, int) or length < 0:
                    raise CK.CheckpointError(f"{prefix}.length 必须为非负整数")
                reason = episode["reason"]
                if not isinstance(reason, str) or reason not in allowed_reasons:
                    raise CK.CheckpointError(
                        f"{prefix}.reason 非法，允许值为 {sorted(allowed_reasons)}"
                    )
            return copy.deepcopy(value)

        if state["stage_phase"] != "online" or state["stage_target_online_steps"] is None:
            raise CK.CheckpointError("pre-online/无阶段目标 trainer_state 不支持无损续训")
        target = nonnegative_int(state["stage_target_online_steps"], "stage_target_online_steps")
        mapping_version = nonnegative_int(state["mapping_version"], "mapping_version")
        if not isinstance(state["offline_done"], bool):
            raise CK.CheckpointError("trainer_state.offline_done 必须为 bool")
        if not isinstance(state["completed_tasks"], list):
            raise CK.CheckpointError("trainer_state.completed_tasks 必须为列表")
        completed_values = [nonnegative_int(x, "completed_tasks")
                            for x in state["completed_tasks"]]
        if len(set(completed_values)) != len(completed_values) or not set(completed_values) <= {0, 1}:
            raise CK.CheckpointError("trainer_state.completed_tasks 非法")
        try:
            trainer_rng_probe = np.random.default_rng()
            trainer_rng_probe.bit_generator.state = state["trainer_rng"]
        except Exception as exc:
            raise CK.CheckpointError("trainer_state.trainer_rng 非法") from exc

        next_ids_raw = state["next_episode_ids"]
        if not isinstance(next_ids_raw, dict):
            raise CK.CheckpointError("trainer_state.next_episode_ids 必须为映射")
        try:
            next_ids = {int(k): nonnegative_int(v, f"next_episode_ids.{k}")
                        for k, v in next_ids_raw.items()}
        except (TypeError, ValueError) as exc:
            raise CK.CheckpointError("trainer_state.next_episode_ids 非法") from exc
        if set(next_ids) != {0, 1}:
            raise CK.CheckpointError("trainer_state.next_episode_ids 非法")
        if state["stage_game"] not in GAMES or state.get("game") != state["stage_game"]:
            raise CK.CheckpointError("trainer_state 当前阶段游戏非法或相互矛盾")
        stats = state["stats"]
        counters = ("random_steps", "online_steps", "learner_updates", "repr_updates",
                    "offline_updates")
        if not isinstance(stats, dict) or not set(counters + ("episodes",)).issubset(stats):
            raise CK.CheckpointError("trainer_state.stats 不完整，拒绝续训")
        validated_stats = {key: nonnegative_int(stats[key], f"stats.{key}") for key in counters}
        validated_stats["episodes"] = validate_episodes(
            stats["episodes"], "trainer_state.stats.episodes"
        )
        if validated_stats["online_steps"] > target:
            raise CK.CheckpointError("trainer_state online progress/target 非法")

        if not isinstance(replay, type(self.replay)):
            raise CK.CheckpointError("checkpoint replay 未完成解码")
        for task in (0, 1):
            episode_ids = [ep.episode_id for bucket in (replay.historical, replay.online)
                           for ep in bucket[task]]
            episode_ids.extend(ep.episode_id for (pending_task, _), ep in replay.pending.items()
                               if pending_task == task)
            if episode_ids and next_ids[task] <= max(episode_ids):
                raise CK.CheckpointError(
                    f"trainer_state.next_episode_ids[{task}] 与 replay episode id 冲突"
                )

        if not isinstance(extra, dict) or not isinstance(extra.get("stats"), dict):
            raise CK.CheckpointError("checkpoint extra.stats 缺失或非法")
        extra_stats = extra["stats"]
        for key in counters:
            if key not in extra_stats:
                raise CK.CheckpointError(f"checkpoint extra.stats.{key} 缺失")
            nonnegative_int(extra_stats[key], f"extra.stats.{key}")
        if "episodes" not in extra_stats:
            raise CK.CheckpointError("checkpoint extra.stats.episodes 缺失")
        validated_extra_episodes = validate_episodes(
            extra_stats["episodes"], "extra.stats.episodes"
        )
        for key in counters + ("episodes",):
            expected = validated_stats[key] if key in counters else validated_stats["episodes"]
            actual = (nonnegative_int(extra_stats[key], f"extra.stats.{key}")
                      if key in counters else validated_extra_episodes)
            if actual != expected:
                raise CK.CheckpointError(f"checkpoint extra.stats 与 trainer_state.stats.{key} 不一致")

        same_game = state["stage_game"] == self.game
        if same_game:
            task = F.task_id(self.game)
            for pending_task, pending_ep_id in list(replay.pending):
                if pending_task == task:
                    pending = replay.pending[(pending_task, pending_ep_id)]
                    if pending.transitions:
                        tail = pending.transitions[-1]
                        tail.terminated = False
                        tail.truncated = True
                        tail.termination_reason = "none"
                    replay.finish_episode(
                        game=self.game, episode_id=pending_ep_id, terminated=False,
                        truncated=True, termination_reason="none",
                    )
            current_stats = {**validated_stats, "latencies": []}
            current_phase = "online"
            current_target = target
        else:
            current_stats = {"random_steps": 0, "online_steps": 0, "learner_updates": 0,
                             "repr_updates": 0, "offline_updates": 0, "episodes": [],
                             "latencies": []}
            current_phase = "pre_online"
            current_target = None
        return {
            "replay": replay,
            "mapping_version": mapping_version, "offline_done": state["offline_done"],
            "completed_tasks": set(completed_values),
            "trainer_rng": copy.deepcopy(state["trainer_rng"]),
            "next_episode_ids": next_ids,
            "stage_game_current": self.game, "stage_phase_current": current_phase,
            "stage_target_current": current_target, "stats_current": current_stats,
        }

    def _latency(self) -> Dict[str, float]:
        if not self.stats["latencies"]:
            return {}
        a = np.array(self.stats["latencies"], dtype=np.float64)
        return {"p50_ms": float(np.percentile(a, 50) * 1000),
                "p95_ms": float(np.percentile(a, 95) * 1000),
                "mean_ms": float(a.mean() * 1000), "n": int(a.size)}


# ---------------------------------------------------------------- 联合评测
def _build_eval_model(cfg, game: str, ckpt_path: str):
    """建模型、载权重、检查该游戏已注册。评测与多 checkpoint 评测共用。"""
    from .models.continuous import ContinuousModel
    from .models.discrete import DiscreteModel

    if cfg.variant == "continuous":
        model = ContinuousModel(seed=cfg.seed)
    else:
        model = DiscreteModel(seed=cfg.seed)
    CK.load_checkpoint(ckpt_path, cfg=cfg, model=model, restore_replay=False,
                       restore_rng=False)
    # 注册表来自 checkpoint；若该游戏未注册则无法评测（明确失败）
    if cfg.variant == "continuous":
        if game not in model.phi.registered_games():
            raise RuntimeError(f"{game} 未在该 checkpoint 的 φ 注册表中，无法评测")
    else:
        if game not in model.registered_games:
            raise RuntimeError(f"{game} 未在该 checkpoint 的 D 注册表中，无法评测")
    return model


def evaluate_many(*, specs: List[Dict[str, Any]], game: str, episodes: int,
                  run_root: Path, port: int = 17911,
                  wemi_env_root: Path = Path("WemiEnv"),
                  connect_timeout_s: float = 3600.0) -> List[Dict[str, Any]]:
    """**一个客户端连接里连续评测多个 checkpoint**（联合评测用）。

    为什么需要：原版客户端**没有重连**，服务端一关 socket 就死，操作者就得重新
    编译一次。要比较"F 阶段权重"和"联合权重"，各起一个进程就要编译两次；
    这里把 env 起一次、复用同一条连接，把 N 个 checkpoint 依次评完。

    每个 spec 是 ``{"label": str, "ckpt": str, "steps": int, "variant": str}``；
    ``steps`` 必须与训练时一致（checkpoint 身份含阶段表）。
    """
    from . import evaluate as EV
    from .envs.features import PolicyMode

    legacy = RealTimeGameEnv(game, wemi_env_root=wemi_env_root, port=port,
                             horizon=50000, connect_timeout_s=connect_timeout_s)
    legacy.start_server()
    out: List[Dict[str, Any]] = []
    try:
        for spec in specs:
            label = spec["label"]
            print(f"\n===== 评测 [{label}] {spec['ckpt']} (steps={spec['steps']}, "
                  f"variant={spec['variant']}) =====", flush=True)
            cfg = _cfg_for_spec(spec)
            legacy.horizon = cfg.horizon
            model = _build_eval_model(cfg, game, spec["ckpt"])
            hash_before = model.parameter_hash()

            def _reset(game_name, seed, _lg=legacy):
                return _lg.reset(seed)

            def _step(game_name, local_action, _lg=legacy):
                return _lg.step(local_action)

            rep = EV.evaluate_game(
                cfg=cfg, model=model, game=game,
                env_reset=_reset, env_step=_step,
                policy_mode=PolicyMode.GREEDY.value,
                episodes=episodes,
                seed_start=int(cfg.eval["test_seed_start"]),
                verbose=True,
            ).as_dict()
            rep["label"] = label
            rep["ckpt"] = spec["ckpt"]
            rep["variant"] = spec["variant"]
            rep["param_hash_unchanged"] = (hash_before == model.parameter_hash())
            out.append(rep)
            p = run_root / f"eval_{game}_{label}_greedy.json"
            _publish_json_no_overwrite(p, rep)
            print(f"  [{label}] 均分={rep['score_mean']:.2f} 中位={rep['score_median']:.1f} "
                  f"达标={rep['pass_count']}/{rep['episodes']} "
                  f"终止={rep['terminations']} → {p}", flush=True)
    finally:
        legacy.close()
    return out


def parse_eval_specs(text: str) -> List[Dict[str, Any]]:
    """解析 ``label=CKPT@steps[:variant]``（逗号分隔）。

    例：``joint=runs/t/stage.pt@50000,f_only=runs/f/stage.pt@50000``
    """
    specs: List[Dict[str, Any]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part or "@" not in part:
            raise ValueError(f"评测 spec 格式应为 label=CKPT@steps[:variant]，得到 {part!r}")
        label, rest = part.split("=", 1)
        path, tail = rest.rsplit("@", 1)
        variant = "continuous"
        if ":" in tail:
            tail, variant = tail.split(":", 1)
        specs.append({"label": label.strip(), "ckpt": path.strip(),
                      "steps": int(tail), "variant": variant.strip()})
    if not specs:
        raise ValueError("没有解析出任何评测 spec")
    return specs


def _cfg_for_spec(spec: Dict[str, Any]):
    """按 ``--steps`` / ``--variant`` 还原评测用的 cfg（身份必须与训练一致）。"""
    import argparse
    ns = argparse.Namespace(
        config=spec.get("config", "wemi_cal/configs/mt_cal.yaml"),
        variant=spec["variant"], mode="full", seed=int(spec.get("seed", 0)),
        schedule=spec.get("schedule", "strict"), steps=int(spec["steps"]),
        first_data=None, repr_updates=None, offline_updates=None)
    return build_cfg(ns)


def evaluate_policy(*, cfg, game: str, ckpt_path: str, episodes: int,
                    port: int = 17911, wemi_env_root: Path = Path("WemiEnv"),
                    connect_timeout_s: float = 1800.0,
                    seed_start: Optional[int] = None) -> Dict[str, Any]:
    """用**同一个 checkpoint** 在某个游戏上做纯评测（§12.1）。

    严格用 ``PolicyMode.GREEDY``（确定性部署规则）：
    C 用 e=tanh(μβ(x)) 再 argmax φ(j|e,g)；**不使用**训练时的行为混合 b。
    评测不写 replay、不更新任何权重。
    """
    from . import evaluate as EV
    from .envs.features import PolicyMode

    model = _build_eval_model(cfg, game, ckpt_path)

    hash_before = model.parameter_hash()
    legacy = RealTimeGameEnv(game, wemi_env_root=wemi_env_root, port=port,
                             horizon=cfg.horizon, connect_timeout_s=connect_timeout_s)
    legacy.start_server()
    try:
        # 原版接口不带 game 参数；evaluate_game 的契约是 reset(game,seed)/step(game,action)
        def _reset(game_name, seed):
            return legacy.reset(seed)

        def _step(game_name, local_action):
            return legacy.step(local_action)

        rep = EV.evaluate_game(
            cfg=cfg, model=model, game=game,
            env_reset=_reset, env_step=_step,
            policy_mode=PolicyMode.GREEDY.value,
            episodes=episodes,
            seed_start=(seed_start if seed_start is not None
                        else int(cfg.eval["test_seed_start"])),
            verbose=True,
        )
        hash_after = model.parameter_hash()
        out = rep.as_dict()
        out["param_hash_unchanged"] = (hash_before == hash_after)
        out["ckpt"] = str(ckpt_path)
        out["variant"] = cfg.variant
        return out
    finally:
        legacy.close()


# ---------------------------------------------------------------- CLI
def build_cfg(args: argparse.Namespace) -> Any:
    import yaml
    from pathlib import Path as P

    raw = yaml.safe_load(P(args.config).read_text(encoding="utf-8"))
    raw["variant"] = getattr(args, "variant", "continuous")
    raw["mode"] = "full"
    raw["seed"] = args.seed
    raw["schedule"] = args.schedule
    raw["env"]["protocol"] = "wemi-rt-v1"
    raw["env"]["clock_mode"] = "realtime-v1"
    port = int(getattr(args, "port", LEGACY_PORT))
    raw["env"]["ports"] = {"Flappybird": port, "Timberman": port}
    if args.steps is not None and args.schedule == "rehearsal":
        raise ConfigError("rehearsal 使用 YAML 中的固定 30k/30k/10k/10k/10k/10k 预算")
    if args.steps is not None:
        per_game = int(args.steps)
        if per_game <= 0:
            raise ConfigError("--steps 必须为正，且表示每个游戏的交互步数")
        raw["train"]["stage_schedules"]["strict"] = [
            {"game": "Flappybird", "steps": per_game},
            {"game": "Timberman", "steps": per_game}]
    test_only = (
        (args.steps is not None and int(args.steps) != 50000)
        or args.first_data is not None
        or args.repr_updates is not None
        or args.offline_updates is not None
    )
    if getattr(args, "no_initial_offline", False) and not test_only:
        raise ConfigError("--no-initial-offline 仅允许 test_only 配置")
    if args.first_data:
        raw["train"]["random_steps_first_visit"] = args.first_data
    if args.repr_updates:
        raw["train"]["repr_updates_first_visit"] = args.repr_updates
    if args.offline_updates is not None:
        raw["train"]["initial_offline_updates"] = max(1, args.offline_updates)
    return validate_config(raw, test_only=test_only)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="C 主方案在**原版实时环境**上训练")
    ap.add_argument("--config", default="wemi_cal/configs/mt_cal.yaml")
    ap.add_argument("--game", required=True, choices=("Flappybird", "Timberman"))
    ap.add_argument("--variant", default="continuous", choices=("continuous", "discrete"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--schedule", default="strict", choices=("strict", "rehearsal"))
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--wemi-env-root", default="WemiEnv")
    ap.add_argument("--port", type=int, default=LEGACY_PORT)
    ap.add_argument("--first-data", type=int, default=None)
    ap.add_argument("--repr-updates", type=int, default=None)
    ap.add_argument("--offline-updates", type=int, default=None)
    ap.add_argument("--no-initial-offline", action="store_true")
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--evaluate", metavar="CKPT", default=None,
                    help="只做纯评测：加载该 checkpoint，在 --game 上用 greedy 跑 --episodes 回合")
    ap.add_argument("--evaluate-multi", default=None, metavar="SPEC[,SPEC...]",
                    help="一个客户端连接里连续评多个 checkpoint。SPEC 形如 "
                         "label=CKPT@steps[:variant]，例："
                         "joint=runs/a.pt@50000,f_only=runs/b.pt@50000")
    ap.add_argument("--episodes", type=int, default=50, help="评测回合数（§12.1 固定 50）")
    ap.add_argument("--connect-timeout", type=float, default=None,
                    help="等待模拟器连接的秒数（默认取配置，可放大以便你先启动项目）")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args(argv)

    cfg = build_cfg(args)
    run_root = _resolve_run_root(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    if args.evaluate_multi:
        specs = parse_eval_specs(args.evaluate_multi)
        outputs = [run_root / f"eval_{args.game}_{spec['label']}_greedy.json"
                   for spec in specs]
        existing = [p for p in outputs if p.exists() or p.is_symlink()]
        if existing:
            raise SystemExit(f"结果文件已存在，拒绝覆盖: {existing[0]}")
        print(f"=== 多 checkpoint 联合评测: game={args.game} episodes={args.episodes} "
              f"policy=greedy ===", flush=True)
        cto = (args.connect_timeout if args.connect_timeout is not None else 3600.0)
        reps = evaluate_many(specs=specs, game=args.game, episodes=args.episodes,
                             run_root=run_root, port=args.port,
                             wemi_env_root=Path(args.wemi_env_root),
                             connect_timeout_s=cto)
        print("\n=== 联合评测汇总（greedy）===", flush=True)
        print(f"  {'标签':<14}{'均分':>8}{'中位':>8}{'最高':>8}{'达标':>12}  终止")
        for r in reps:
            print(f"  {r['label']:<14}{r['score_mean']:>8.2f}{r['score_median']:>8.1f}"
                  f"{r.get('score_max', float('nan')):>8.1f}"
                  f"{str(r['pass_count']) + '/' + str(r['episodes']):>12}  "
                  f"{r['terminations']}  动作={r.get('token_counts')}")
        return 0

    if args.evaluate:
        out_path = run_root / f"eval_{args.game}_greedy.json"
        if out_path.exists() or out_path.is_symlink():
            raise SystemExit(f"结果文件已存在，拒绝覆盖: {out_path}")
        print(f"=== 纯评测: game={args.game} variant={cfg.variant} "
              f"episodes={args.episodes} policy=greedy ===", flush=True)
        cto = (args.connect_timeout if args.connect_timeout is not None else 1800.0)
        rep = evaluate_policy(
            cfg=cfg, game=args.game, ckpt_path=args.evaluate, episodes=args.episodes,
            port=args.port, wemi_env_root=Path(args.wemi_env_root),
            connect_timeout_s=cto)
        _publish_json_no_overwrite(out_path, rep)
        print("\n=== 评测结果（greedy）===", flush=True)
        print(f"  回合数      : {rep['episodes']}")
        print(f"  原始分 均值 : {rep['score_mean']:.3f}")
        print(f"  原始分 中位 : {rep['score_median']:.1f}")
        print(f"  原始分 标准差: {rep['score_std']:.3f}")
        print(f"  原始分 最高 : {rep.get('score_max', 0.0):.1f}")
        print(f"  达标(>=门槛): {rep['pass_count']}/{rep['episodes']} = {rep['pass_rate']:.1%}")
        print(f"  终止原因    : {rep['terminations']}")
        print(f"  动作直方图  : {rep.get('token_counts')}")
        print(f"  权重未被改动: {rep['param_hash_unchanged']}")
        print(f"  结果写入    : {out_path}", flush=True)
        return 0
    stage_steps = args.steps or cfg.steps_for_game(args.game)
    stage_checkpoint = run_root / f"stage_{args.game}.pt"
    stage_summary = run_root / f"stage_{args.game}_summary.json"
    existing = next((p for p in (stage_checkpoint, stage_summary)
                     if p.exists() or p.is_symlink()), None)
    if existing is not None:
        raise SystemExit(f"阶段产物已存在，拒绝覆盖: {existing}")

    print(f"=== 实时训练: game={args.game} port={args.port} steps={stage_steps} "
          f"test_only={cfg.test_only} ===", flush=True)

    cto = (args.connect_timeout if args.connect_timeout is not None
           else float(cfg.env["connect_timeout_s"]))
    legacy = RealTimeGameEnv(args.game, wemi_env_root=Path(args.wemi_env_root),
                             port=args.port, horizon=cfg.horizon,
                             connect_timeout_s=cto)
    trainer = RealTimeTrainer(cfg, args.game, legacy, run_root=run_root, seed=args.seed)

    if args.resume:
        trainer.restore(args.resume)
        print(f"[resume] mapping_version={trainer.mapping_version} "
              f"offline_done={trainer.offline_done}", flush=True)
    legacy.start_server()

    # "首次接入"必须按**该游戏在 replay 里有没有数据**判断，不能只看有没有 resume。
    # 方案 §9.1 的 S2 就是这样：T 是新游戏，必须走"随机采集 → 适配(扩 φ) → 在线"，
    # 只从 F 的 checkpoint 继承 β/Q，不继承 T 的数据。
    # 反例（真实发生过）：只看 resume 会跳过 T 的采集与适配 → φ 的 T 行从未训练 →
    # 动作近乎随机（实测均分 2.32、回合长度恒定 5.6 步、完全无学习趋势）。
    task = F.task_id(args.game)
    task_transitions = (sum(len(e) for e in trainer.replay.online.get(task, []))
                        + sum(len(e) for e in trainer.replay.historical.get(task, [])))
    first_visit = task_transitions == 0
    print(f"[stage] {args.game}(task={task}) replay 已有 {task_transitions} 条 → "
          f"first_visit={first_visit}", flush=True)

    t0 = time.monotonic()
    finalized_state = None
    try:
        if (args.resume and trainer.stage_game == args.game
                and trainer.stage_target_online_steps is not None):
            online = trainer.stage_target_online_steps
        elif first_visit:
            rs = int(cfg.train["random_steps_first_visit"])
            print(f"[stage] 随机采集 {rs} 步", flush=True)
            trainer.collect_first_data(rs)
            ru = int(cfg.train["repr_updates_first_visit"])
            print(f"[stage] 表示适配 {ru} updates", flush=True)
            trainer.adapt(ru)
            if not args.no_initial_offline and not trainer.offline_done:
                ou = int(cfg.train["initial_offline_updates"])
                print(f"[stage] 初始离线 {ou} updates", flush=True)
                trainer.initial_offline(ou)
            online = stage_steps - rs
        else:
            online = stage_steps
        trainer.online(online, update_every=int(cfg.train["update_every_online_steps"]),
                       save_every=args.save_every)
        finalized_state = (
            task in trainer.completed_tasks,
            list(trainer.replay.historical[task]),
            list(trainer.replay.online[task]),
        )
        trainer.finalize_stage()
        ckpt = trainer.save(stage_checkpoint, stage=args.game)
        summary = {"game": args.game, "steps": stage_steps, "wall_clock_s": time.monotonic() - t0,
                   "stats": {k: v for k, v in trainer.stats.items() if k != "latencies"},
                   "latency": trainer._latency(), "checkpoint": str(ckpt),
                   "replay": trainer.replay.diagnostics(),
                   "mapping_version": trainer.mapping_version}
        _publish_json_no_overwrite(stage_summary, summary)
        print(f"\n=== 完成 {args.game} ===", flush=True)
        print(json.dumps({"latency": summary["latency"],
                          "online_steps": summary["stats"]["online_steps"],
                          "learner_updates": summary["stats"]["learner_updates"]},
                         indent=2, ensure_ascii=False), flush=True)
        sc = [e["score_raw"] for e in trainer.stats["episodes"]]
        if sc:
            print(f"原始分数: n={len(sc)} mean={sum(sc)/len(sc):.2f} max={max(sc):.0f}", flush=True)
        return 0
    except BaseException as exc:
        # Final阶段保存失败时恢复 finalize 前的分桶和完成状态；救援快照不能冒充完成。
        if finalized_state is not None and not finalized_state[0]:
            trainer.replay.historical[task] = finalized_state[1]
            trainer.replay.online[task] = finalized_state[2]
            trainer.completed_tasks.discard(task)
        try:
            rescue_path = run_root / f"stage_{args.game}_INTERRUPTED_{uuid.uuid4().hex}.pt"
            rescue = trainer.save(rescue_path,
                                  stage=f"{args.game}-interrupted")
            print(f"[rescue] 已保存中断现场: {rescue}", flush=True)
        except Exception as se:
            print(f"[rescue] 保存失败: {se}", flush=True)
        raise
    finally:
        legacy.close()


if __name__ == "__main__":
    raise SystemExit(main())
