"""训练入口与端到端装配（技术方案 §13，任务 T8）。

两种运行方式：

- ``python -m wemi_cal.train --config ...``：真实实验入口，连接微信模拟器桥接。
  拒绝 ``test_only`` 配置，避免小预算测试被当成正式结果。
- ``python -m wemi_cal.train --smoke --steps 400``：**无模拟器**自检。用进程内
  特征级游戏模拟器跑通 采集→适配→初始离线→在线→checkpoint→评测 的完整链路。
  它不是能力实验，只证明装配与状态机正确。

CLI override 解析后的**最终配置**会写入 checkpoint 与日志。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import checkpoint as CK
from . import evaluate as EV
from .config import Config, ConfigError, GAMES, load_config, run_dir, validate_config
from .envs import features as F
from .envs.game_env import WemiGameEnv
from .envs.session import Session
from .envs import bridge
from .learner import ContinuousLearner, LearnerConfig
from .models.continuous import ContinuousModel
from .models.discrete import DiscreteModel, adapt_discrete_representation
from .replay import ReplayBuffer, make_transition
from .repr_train import ReprBatch, adapt_representation, make_batches
from .schedule import Trainer, TrainerCallbacks

# ---------------------------------------------------------------- 进程内假游戏
@dataclass
class LocalGameSim:
    """特征级本地游戏模拟器（只在 --smoke 下使用）。

    刻意做得很简单：它不复制 JS 物理，只产生**形状与语义合法**的 29 维状态、
    bounded 奖励与终止原因，用来在没有模拟器时验证装配与状态机。
    真实能力实验必须走 WebSocket 桥接 + 微信模拟器。
    """

    game: str
    horizon: int = 1000
    episode_len: int = 40
    seed: int = 0
    _t: int = 0
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    _episode_id: int = -1
    _score: float = 0.0
    _lift: float = 0.0
    _pipe_x: float = 1.0
    _sides: Any = None
    _npc_right: bool = True
    _blood: float = 1.0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._sides = [0.0] * 11
        # 有限时域与游戏内回合取更短者：H 到达时 reason=horizon 而不是游戏死亡
        self._limit = min(int(self.episode_len), int(self.horizon))

    def reset(self, game: str = "", seed: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._t = 0
        self._episode_id += 1
        self._score = 0.0
        self._lift = 0.0
        self._pipe_x = 1.0
        self._blood = 1.0
        self._npc_right = bool(self._rng.random() > 0.5)
        self._sides = [float(self._rng.choice([-1.0, 0.0, 1.0])) for _ in range(11)]
        self._rng = np.random.default_rng(seed)
        return self._obs(), {"episode_id": self._episode_id, "score_raw": 0.0}

    def _obs(self) -> np.ndarray:
        """构造 29 维状态。

        关键：**下一状态必须能被动作区分**，否则逆动力学 ψ 无法从 (x,x') 推断动作，
        表示适配必然坍缩到只选一个动作（`all_actions_reachable=false`）。
        第一版模拟器的 x' 与动作无关，φ 立刻坍缩——这里显式修掉。
        """
        x = np.zeros(F.OBS_DIM, dtype=np.float32)
        task = F.task_id(self.game)
        if self.game == "Flappybird":
            # f0 随点击抬升、f3 随时间左移：动作 1 → f0 变大且 f3 变大
            x[0] = float(np.clip(0.5 + self._lift - 0.02 * self._t, -0.5, 1.5))
            x[1] = 1.0 / 3.0
            x[2] = 2.0 / 3.0
            x[3] = float(np.clip(self._pipe_x, -1.0, 1.25))
            x[13:17] = 1.0
        else:
            # 树方位由上一个动作决定：动作 0 → 最前树在左，动作 1 → 在右
            sides = self._sides.copy()
            x[0:11] = sides
            x[11] = 1.0 if self._npc_right else -1.0
            x[12] = float(np.clip(self._blood, 0.0, 1.0))
            x[13:26] = 1.0
        x[26 + task] = 1.0
        x[F.BUDGET_INDEX] = max(0.0, (self.horizon - self._t) / self.horizon)
        return x

    def step(self, game: str = "", local: int = 0) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._t += 1
        if self.game == "Flappybird":
            flap = 1 if local == 1 else 0
            self._lift = 0.12 * flap                       # 动作改变观测
            self._pipe_x -= 0.05
            # 间隙距离：点击抬高会改善（有确定性的动作-奖励关系）
            gap_center = 0.5
            r = float(np.clip(0.4 + 0.5 * flap - abs(0.5 + self._lift - gap_center), -1.0, 1.0))
            self._score += 1.0 if flap else 0.0
        else:
            # 最前树在动作的对侧才算砍到
            self._sides[0] = -1.0 if local == 0 else 1.0
            hit = (local == 1 and self._sides[1] != 1.0) or (local == 0 and self._sides[1] != -1.0)
            r = 1.0 if hit else -0.2
            self._blood = min(1.0, self._blood + (0.1 if hit else -0.05))
            self._score += 1.0 if hit else 0.0
        done = self._t >= self._limit
        # 到 H 是任务定义的 horizon 终止，不是游戏死亡（§4.5）
        reason = ("horizon" if self._t >= self.horizon else "collision") if done else "none"
        info = {
            "episode_id": self._episode_id,
            "score_raw": self._score,
            "termination_reason": reason,
            "termination_reason_code": reason,
            "episode_return_train": 0.0,
        }
        return self._obs(), r, bool(done), False, info


# ---------------------------------------------------------------- 运行器
class CalLaicaRunner:
    """把 config、模型、replay、learner 与状态机装配起来。"""

    def __init__(self, cfg: Config, *, device: str = "cpu", out_root: str | Path = "runs_cal/v2") -> None:
        self.cfg = cfg
        self.device = device
        torch.set_num_threads(int(cfg.raw.get("torch_threads", 1)))
        self.rng = np.random.default_rng(cfg.seed)

        if cfg.variant == "continuous":
            self.model: Any = ContinuousModel(seed=cfg.seed)
        elif cfg.variant == "discrete":
            self.model = DiscreteModel(seed=cfg.seed)
        else:
            raise ConfigError(f"未知 variant: {cfg.variant!r}")

        self.replay = ReplayBuffer(
            per_task_capacity=int(cfg.replay["per_task_capacity"]),
            max_stored_transitions=int(cfg.replay["max_stored_transitions"]),
            gamma=cfg.gamma,
            batch_size=int(cfg.replay["batch_size"]),
            active_task_history_fraction=float(cfg.replay["active_task_history_fraction"]),
            seed=cfg.seed,
        )
        self.learner = ContinuousLearner(self.model, LearnerConfig(
            gamma=cfg.gamma,
            actor_lr=float(cfg.train["actor_lr"]),
            critic_lr=float(cfg.train["critic_lr"]),
            entropy_alpha=float(cfg.train["entropy_alpha"]),
            lambda_cal=cfg.lambda_cal,
            tau=float(cfg.train["tau"]),
            grad_clip=float(cfg.train["grad_clip"]),
            target_samples=int(cfg.train["target_samples"]),
            calibration_samples=int(cfg.train["calibration_samples"]),
            calibration_mode=cfg.calibration_mode,
        )) if cfg.variant == "continuous" else None

        self.out_root = Path(out_root)
        self.run_dir = run_dir(cfg, self.out_root)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "train_log.jsonl"
        self.mapping_version = 0
        self.first_data: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
        self.repr_diag: Dict[str, Any] = {}
        self.rollout_snapshot: Dict[int, int] = {}

    # ---- 日志 ----
    def log(self, rec: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # ---- 回放构造 ----
    def _batch_from_replay(self, active_task: int) -> Dict[str, Any]:
        trs = self.replay.sample(active_task)
        return self.replay.stack(trs, game_of_task={0: "Flappybird", 1: "Timberman"})

    # ---- 表示适配 ----
    def _adapt(self, game: str, updates: int) -> int:
        task = F.task_id(game)
        rows = self.first_data[task]
        if not rows:
            raise RuntimeError(f"{game} 没有首次采集数据，无法适配")
        if self.cfg.variant == "continuous":
            x = torch.tensor(np.stack([r["x"] for r in rows]), dtype=torch.float32)
            nx = torch.tensor(np.stack([r["next_x"] for r in rows]), dtype=torch.float32)
            tok = torch.tensor([int(r["global_token"]) for r in rows], dtype=torch.long)
            eps = [int(r["episode_id"]) for r in rows]
            train_idx, diag_idx = __import__("wemi_cal.repr_train", fromlist=["x"]).split_by_episode(
                eps, val_fraction=0.2, seed=self.cfg.seed
            )
            batches = make_batches(
                x=x[train_idx], next_x=nx[train_idx], token=tok[train_idx],
                game=game, batch_size=int(self.cfg.replay["batch_size"]), seed=self.cfg.seed,
            )
            # 已注册则直接训练；首次接入时先扩展输出层
            if game not in self.model.phi.registered_games():
                if self.model.phi.num_tokens == 0:
                    self.model.register_game_as_first(game)
                else:
                    self.model.register_game(game)
            diag = adapt_representation(
                self.model, batches, updates=updates,
                lr=float(self.cfg.continuous["repr_lr"]),
                kl_weight=float(self.cfg.continuous["kl_weight"]),
                grad_clip=float(self.cfg.train["grad_clip"]),
            )
            diagb = None
            if diag_idx:
                diag_batches = make_batches(
                    x=x[diag_idx], next_x=nx[diag_idx], token=tok[diag_idx],
                    game=game, batch_size=min(128, max(2, len(diag_idx))), seed=self.cfg.seed + 1,
                )
                diagb = adapt_representation(
                    self.model, diag_batches, updates=1,
                    lr=float(self.cfg.continuous["repr_lr"]),
                    kl_weight=float(self.cfg.continuous["kl_weight"]),
                )
            self.repr_diag[game] = {
                "train_accuracy": diag.accuracy, "train_nll": diag.nll, "kl": diag.kl,
                "per_action_accuracy": diag.per_action_accuracy,
                "diag_accuracy": (diagb.accuracy if diagb else None),
                "coverage": self.model.phi.coverage_grid(game, res=41),
            }
            self.model.phi.registered_only  # noqa: B018 - 保持接口被使用
        else:
            if game not in self.model.registered_games:
                self.model.register_game(game)
            x = torch.tensor(np.stack([r["x"] for r in rows]), dtype=torch.float32)
            nx = torch.tensor(np.stack([r["next_x"] for r in rows]), dtype=torch.float32)
            tok = torch.tensor([int(r["global_token"]) for r in rows], dtype=torch.long)
            data = [(x, nx, tok)]
            d = adapt_discrete_representation(
                self.model, data=data, game=game, updates=updates,
                lr=float(self.cfg.continuous["repr_lr"]),
            )
            self.repr_diag[game] = {"ce": d.ce, "sep": d.sep, "accuracy": d.accuracy}

        self.mapping_version += 1
        self.log({"event": "first_data_balance", "game": game,
                  "counts": self._token_counts(task)})
        self.log({"event": "adapt_diagnostics", "game": game,
                  "mapping_version": self.mapping_version,
                  "diagnostics": self.repr_diag[game]})
        return self.mapping_version

    # ---- 首次采集 ----
    def _collect_first_data(self, game: str, steps: int, mapping_version: int,
                            env: Any) -> None:
        task = F.task_id(game)
        n = 0
        legal = F.legal_tokens(game)
        while n < steps:
            x, info = env.reset(seed=int(self.rng.integers(0, 1 << 30)))
            ep_id = info["episode_id"]
            # 未扩展 φ 之前：独立采 u~N(0,I)，e=tanh(u)，动作从合法集合均匀采样。
            # 显式轮转 token 保证**精确均衡**：ψ/φ 的适配数据若某动作一条都没有，
            # 重构目标必然坍缩（per-action accuracy 会出现 0）。这在真实环境同样适用。
            while n < steps:
                u = self.rng.normal(size=2).astype(np.float32)
                e = np.tanh(u).astype(np.float32)
                token = int(legal[n % len(legal)])
                local = F.local_action(game, token)
                log_b = -math.log(len(F.legal_tokens(game)))
                nx, r, term, trunc, sinfo = env.step(game, local)
                n += 1
                self.first_data[task].append({
                    "x": x, "next_x": nx, "e_behavior": e,
                    "global_token": token, "reward_train": float(r),
                    "log_b_action": float(log_b), "episode_id": int(ep_id),
                })
                x = nx
                # 均匀采集也写入 replay（ref_valid 先为假）
                self.replay.add_confirmed(make_transition(
                    buffer=self.replay, game=game, episode_id=int(ep_id), request_id=n,
                    x=x, next_x=nx, e_behavior=e, global_token=token,
                    reward_train=float(r), log_b_action=log_b,
                    behavior_kernel="uniform",
                    behavior_mapping_version=mapping_version,
                    terminated=term, truncated=trunc,
                    termination_reason=sinfo.get("termination_reason", "none"),
                    env_hash=f"local-{game}-sim", clock_mode=self.cfg.env["clock_mode"],
                ), game=game)
                if term or trunc:
                    self.replay.finish_episode(
                        game=game, episode_id=int(ep_id), terminated=term, truncated=trunc,
                        termination_reason=sinfo.get("termination_reason", "none"),
                    )
                    break
                if n >= steps:
                    # §6.4：随机采集恰好到 2000 步时，未完成回合标操作性截断
                    self.replay.finish_episode(
                        game=game, episode_id=int(ep_id), terminated=False, truncated=True,
                        termination_reason="none",
                    )
                    break
            else:
                continue

    # ---- 在线动作选择 ----
    def _choose_action(self, game: str, mapping_version: int, rollout_id: int,
                       x: np.ndarray) -> Tuple[int, int, float, np.ndarray, int]:
        xb = torch.as_tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            if self.cfg.variant == "continuous":
                s = self.model.beta.rsample(xb, samples=1)
                e = s.e_squeezed
                mix = float(self.cfg.continuous["decoder_behavior_uniform_mix"])
                logb_all = self.model.phi.behavior_log_probs(e, game, uniform_mix=mix)
                probs = torch.softmax(logb_all, dim=-1)
                token = int(torch.multinomial(probs.clamp_min(0.0), 1,
                                              generator=self.model.phi.gen).item())
                log_b = float(self.model.phi.behavior_log_prob_of(
                    e, game, torch.tensor([token]), uniform_mix=mix).item())
            else:
                token = int(self.model.actor.sample_token(xb, self.model.codebook, game).item())
                log_b = 0.0
            e_np = (e[0].numpy() if self.cfg.variant == "continuous"
                    else np.zeros(2, dtype=np.float32))
        return token, F.local_action(game, token), log_b, e_np, 0

    # ---- learner ----
    def _learner_update(self, mapping_version: int) -> Dict[str, Any]:
        if self.cfg.variant != "continuous":
            return {"td_loss": 0.0, "ess_over_batch": 1.0}
        batch = self._batch_from_replay(active_task=self._active_task)
        batch["games"] = [GAMES[int(t)] for t in batch["task_id"].tolist()]
        stats = self.learner.update(batch)
        return {
            "td_loss": stats.td_loss, "cal_regularizer": stats.cal_regularizer,
            "actor_loss": stats.actor_loss, "w_mean": stats.w_mean, "w_max": stats.w_max,
            "ess_over_batch": stats.ess_over_batch,
            "reference_coverage": stats.reference_coverage,
        }

    # ---- 初始离线 ----
    def _initial_offline(self, game: str, updates: int) -> int:
        if self.cfg.variant != "continuous":
            return updates
        task = F.task_id(game)
        rows = self.first_data[task]
        x = torch.tensor(np.stack([r["x"] for r in rows]), dtype=torch.float32)
        nx = torch.tensor(np.stack([r["next_x"] for r in rows]), dtype=torch.float32)
        e = torch.tensor(np.stack([r["e_behavior"] for r in rows]), dtype=torch.float32)
        tok = torch.tensor([int(r["global_token"]) for r in rows], dtype=torch.long)
        rew = torch.tensor([float(r["reward_train"]) for r in rows], dtype=torch.float32)
        logb = torch.tensor([float(r["log_b_action"]) for r in rows], dtype=torch.float32)
        batch = {
            "x": x, "next_x": nx, "e_behavior": e, "global_token": tok,
            "local_action": torch.tensor([F.local_action(game, int(t)) for t in tok],
                                         dtype=torch.long),
            "reward": rew, "log_b_action": logb,
            "reference_return": torch.zeros_like(rew),
            "ref_valid": torch.zeros_like(rew, dtype=torch.bool),
            "task_id": torch.full((len(rows),), task, dtype=torch.long),
            "mapping_version": torch.full((len(rows),), self.mapping_version, dtype=torch.long),
            "terminated": torch.zeros(len(rows), dtype=torch.bool),
            "truncated": torch.zeros(len(rows), dtype=torch.bool),
            "games": [game] * len(rows),
        }
        for _ in range(updates):
            self.learner.update(batch)
        return updates

    # ---- 真实回合回调（状态机用） ----
    def build_callbacks(self, envs: Dict[str, Any]) -> TrainerCallbacks:
        state = {"x": None}

        def env_reset(game, seed):
            x, info = envs[game].reset(game, seed)
            state["x"] = x
            return x, info

        def env_step(game, local):
            x = state["x"]
            nx, r, term, trunc, info = envs[game].step(game, local)
            score_before = float(getattr(envs[game], "_score", 0.0)) - (
                float(r) if r > 0 else 0.0)
            info = dict(info)
            info.setdefault("termination_reason", "none")
            info["score_raw"] = float(getattr(envs[game], "_score", 0.0))
            state["x"] = nx
            return nx, r, term, trunc, info

        def choose_action(game, mapping_version, rollout_id):
            token, local, log_b, e_np, bid = self._choose_action(
                game, mapping_version, rollout_id, state["x"])
            return token, local, log_b, e_np, bid

        def on_transition(**kw):
            game = kw["game"]
            # 同一 batch 含旧 F 数据和新 T 数据 ≠ 同时在线采样；这里只写入对应桶
            self.replay.add_confirmed(make_transition(
                buffer=self.replay, game=game, episode_id=int(kw.get("episode_id", -1)),
                request_id=int(kw.get("request_id", 0)),
                x=kw["x"], next_x=kw["next_x"], e_behavior=kw["e_beh"],
                global_token=kw["token"], reward_train=float(kw["r"]),
                log_b_action=float(kw["log_b"]),
                behavior_kernel="decoder_exploration",
                behavior_mapping_version=int(kw["mapping_version"]),
                behavior_id=int(kw["behavior_id"]),
                terminated=kw["terminated"], truncated=kw["truncated"],
                termination_reason=kw["termination_reason"],
                env_hash=f"local-{game}-sim", clock_mode=self.cfg.env["clock_mode"],
            ), game=game)

        def on_episode_end(game, episode_id, term, trunc, reason):
            if episode_id is None or episode_id < 0:
                return
            try:
                self.replay.finish_episode(
                    game=game, episode_id=int(episode_id), terminated=term,
                    truncated=trunc, termination_reason=reason,
                )
            except Exception as exc:  # 已经结束/不存在：记录而不是静默
                self.log({"event": "episode_end_rejected", "game": game,
                          "episode_id": episode_id, "error": str(exc)})

        return TrainerCallbacks(
            collect_first_data=lambda game, steps, mv: self._collect_first_data(
                game, steps, mv, envs[game]),
            adapt_representation=self._adapt,
            learner_update=self._learner_update,
            initial_offline=self._initial_offline,
            env_reset=env_reset,
            env_step=env_step,
            choose_action=choose_action,
            on_episode_end=on_episode_end,
            on_transition=on_transition,
        )

    # ---- 主流程 ----
    def run(self, *, envs: Dict[str, Any], eval_envs: Optional[Dict[str, Any]] = None,
            do_eval: bool = False, episodes: Optional[int] = None) -> Dict[str, Any]:
        self._active_task = F.task_id(self.cfg.stage_schedule[0]["game"])
        callbacks = self.build_callbacks(envs)
        def on_stage_end(game: str, mapping_version: int, res: Any) -> None:
            """阶段结束把该任务在线数据固化进历史（§8.4）。

            strict 的 S1→S2 要求 F 的旧数据固化后不再补写；这里在阶段边界统一固化。
            """
            task = F.task_id(game)
            if self._task_has_pending(game):
                self.log({"event": "commit_skipped_pending", "game": game})
                return
            self.replay.commit_online_to_history(task)
            self.log({"event": "commit_stage_to_history", "game": game,
                      "task": task, "mapping_version": mapping_version})

        trainer = Trainer(cfg=self.cfg, model=self.model, callbacks=callbacks,
                          horizon=self.cfg.horizon, logger=self.log,
                          on_stage_end=on_stage_end)
        summary = trainer.run()

        ckpt_path = CK.save_checkpoint(
            self.run_dir / "ckpt.pt", cfg=self.cfg, model=self.model,
            optimizers=self._optimizers(), replay=self.replay,
            trainer_state={"counters": trainer.counters,
                           "mapping_version": self.mapping_version,
                           "subphase": trainer.subphase.value},
            extra={"repr_diagnostics": self.repr_diag, "summary": summary},
        )
        policy_path = CK.export_policy(self.run_dir / "policy.pt", cfg=self.cfg, model=self.model)
        out = {
            "summary": summary,
            "checkpoint": str(ckpt_path),
            "policy": str(policy_path),
            "policy_hash": CK.policy_hash(policy_path),
            "replay": self.replay.diagnostics(),
            "repr_diagnostics": self.repr_diag,
            "param_count": sum(p.numel() for p in self.model.parameters()),
        }
        if do_eval and eval_envs:
            out["eval"] = {}
            n = episodes if episodes is not None else int(self.cfg.eval["validation_episodes"])
            for game in GAMES:
                if game not in eval_envs:
                    continue
                out["eval"][game] = {}
                for mode in (self.cfg.eval["primary_policy_mode"],
                             self.cfg.eval["diagnostic_policy_mode"]):
                    rep = EV.evaluate_game(
                        cfg=self.cfg, model=self.model, game=game,
                        env_reset=eval_envs[game].reset, env_step=eval_envs[game].step,
                        policy_mode=mode, episodes=n,
                        seed_start=int(self.cfg.eval["validation_seed_start"]),
                    )
                    out["eval"][game][mode] = rep.as_dict()
        # 评测不得改变权重（§14 T8）
        out["param_hash_after"] = self.model.parameter_hash()
        (self.run_dir / "summary.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return out

    def _token_counts(self, task: int) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for r in self.first_data[task]:
            t = int(r["global_token"])
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _task_has_pending(self, game: str) -> bool:
        task = F.task_id(game)
        return any(k[0] == task for k in self.replay.pending)

    def _optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        if self.cfg.variant != "continuous" or self.learner is None:
            return {}
        return {
            "q1": self.learner.opt_q1, "q2": self.learner.opt_q2,
            "beta": self.learner.opt_beta,
        }


# ---------------------------------------------------------------- smoke 配置
SMOKE_TRAIN = {
    "random_steps_first_visit": 60,
    "repr_updates_first_visit": 20,
    "initial_offline_updates": 10,
    "update_every_online_steps": 4,
}


def smoke_config(base: str | Path, *, variant: str = "continuous", schedule: str = "strict",
                 total_steps: int = 400, seed: int = 0) -> Config:
    """构造小预算自检配置（显式 test_only）。"""
    import yaml

    raw = yaml.safe_load(Path(base).read_text(encoding="utf-8"))
    raw["variant"] = variant
    raw["schedule"] = schedule
    raw["mode"] = "full"
    raw["seed"] = seed
    if schedule == "rehearsal":
        # 必须保持 6 阶段顺序（§9.2）；同时每游戏总预算相等，且首次访问的阶段步数
        # 足以覆盖随机采集（否则状态机会拒绝）。
        half = total_steps // 2
        first = max(SMOKE_TRAIN["random_steps_first_visit"] + 4, half // 2)
        rest = (half - first) // 2
        raw["train"]["stage_schedules"]["rehearsal"] = [
            {"game": "Flappybird", "steps": first},
            {"game": "Timberman", "steps": first},
            {"game": "Flappybird", "steps": rest},
            {"game": "Timberman", "steps": rest},
            {"game": "Flappybird", "steps": half - first - rest},
            {"game": "Timberman", "steps": half - first - rest},
        ]
    else:
        raw["train"]["stage_schedules"]["strict"] = [
            {"game": "Flappybird", "steps": total_steps // 2},
            {"game": "Timberman", "steps": total_steps // 2},
        ]
    raw["train"].update(SMOKE_TRAIN)
    return validate_config(raw, test_only=True)


def run_smoke(*, base_config: str | Path, variant: str = "continuous",
              schedule: str = "strict", total_steps: int = 400, seed: int = 0,
              out_root: str | Path = "runs_cal/smoke") -> Dict[str, Any]:
    """无模拟器端到端自检：真实状态机 + 真实 learner + 真实 replay + 真实 checkpoint。"""
    cfg = smoke_config(base_config, variant=variant, schedule=schedule,
                       total_steps=total_steps, seed=seed)
    if cfg.variant == "discrete":
        raise ConfigError("discrete smoke 暂未接入（D 的 learner 不在 T0–T8 范围内）")
    runner = CalLaicaRunner(cfg, out_root=out_root)
    envs = {g: LocalGameSim(game=g, horizon=cfg.horizon, episode_len=12, seed=seed + i)
            for i, g in enumerate(GAMES)}
    eval_envs = {g: LocalGameSim(game=g, horizon=cfg.horizon, episode_len=12,
                                 seed=9000 + i) for i, g in enumerate(GAMES)}
    return runner.run(envs=envs, eval_envs=eval_envs, do_eval=True, episodes=2)


# ---------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wemi_cal.train", description="MT-Cal-LAICA 训练入口")
    p.add_argument("--config", default="wemi_cal/configs/mt_cal.yaml")
    p.add_argument("--variant", choices=("continuous", "discrete"))
    p.add_argument("--schedule", choices=("strict", "rehearsal", "single"))
    p.add_argument("--mode", choices=("full", "replay", "uncalibrated", "single"))
    p.add_argument("--game", default=None, help="mode=single 时指定游戏")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-root", default="runs_cal/v2")
    p.add_argument("--resume", default=None, help="从 checkpoint 恢复（必须身份匹配）")
    # 无模拟器自检
    p.add_argument("--smoke", action="store_true",
                   help="用进程内假游戏跑通装配与状态机（不是能力实验）")
    p.add_argument("--smoke-steps", type=int, default=400)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        out = run_smoke(base_config=args.config, variant=args.variant or "continuous",
                        schedule=args.schedule or "strict", total_steps=args.smoke_steps,
                        seed=args.seed or 0)
        print(json.dumps({k: out[k] for k in ("policy", "policy_hash", "param_count",
                                              "param_hash_after")},
                         indent=2, ensure_ascii=False))
        print("smoke OK（这不是能力实验）")
        return 0

    cfg = load_config(
        args.config, variant=args.variant, schedule=args.schedule, mode=args.mode,
        seed=args.seed, game=args.game,
    )
    if cfg.test_only:
        raise SystemExit("拒绝：正式入口不接受 test_only 配置")
    if args.resume:
        raise SystemExit(
            "真实训练需要微信模拟器连接（§4.1）。请先按 T2 完成 JS lockstep 并启动对应端口，"
            "再运行本入口；当前环境未验证模拟器链路。"
        )
    raise SystemExit(
        "真实训练需要微信模拟器连接（§4.1）：启动桥接 → 在开发者工具打开对应项目 → "
        "确认 game_id/协议/env_hash。当前未运行模拟器实验，请使用 --smoke 验证装配。"
    )


if __name__ == "__main__":
    raise SystemExit(main())
