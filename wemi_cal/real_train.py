"""正式训练入口：真实微信模拟器 + wemi-cal-v1 桥接（技术方案 §4.1/§9/§13）。

与 ``--smoke`` 的区别：这里跑的是**真机链路**——
Python 起 WebSocket 服务端，微信开发者工具里的游戏项目连上来，逐条 step 交互。

用法：

    # 单个游戏（阶段 1）
    python -m wemi_cal.run_real --game Flappybird --port 17911 --steps 50000 --run-id c-strict

    # 单个游戏（阶段 2，接第一阶段的权重）
    python -m wemi_cal.run_real --game Timberman --port 17912 --steps 50000 \\
        --run-id c-strict --resume runs_real/c-strict/ckpt.pt

设计取舍（必须写清楚，避免让人误以为可以无人值守跑完 10 万步）：
- 微信模拟器**一次只能开一个项目**（两个游戏都占各自端口，但同时编译两个项目会互相干扰
  且吃满机器），所以双任务需要人工在两个阶段之间切换项目；
- 因此本脚本一次只跑**一个阶段**，阶段之间由 ``run_strict.py`` 编排；
- 卡住不隐藏：连不上客户端、超时、协议错都会明确报错并保存现场。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import checkpoint as CK
from . import evaluate as EV
from .config import GAMES, Config, run_dir, validate_config
from .envs import bridge, features as F
from .envs.bridge import EnvTimeout, NoClientError
from .envs.game_env import WemiGameEnv
from .envs.session import Session
from .learner import ContinuousLearner, LearnerConfig
from .models.continuous import ContinuousModel
from .replay import ReplayBuffer, make_transition
from .repr_train import adapt_representation, make_batches, split_by_episode
from .schedule import TrainerCallbacks, set_trainable
from .schedule import SubPhase, assert_frozen

DEVTOOLS_CLI = "/Applications/wechatwebdevtools.app/Contents/MacOS/cli"
PROJECTS = {
    "Flappybird": "wechat_games_cal/Flappybird",
    "Timberman": "wechat_games_cal/Timberman",
}
DEFAULT_PORTS = {"Flappybird": 17911, "Timberman": 17912}
ENV_HASH = {
    "Flappybird": "flappybird-lockstep-v1",
    "Timberman": "timberman-lockstep-v1",
}


# ---------------------------------------------------------------- devtools 控制
def open_project(game: str, workspace: Path, *, wait: bool = True,
                 timeout_s: float = 180.0) -> bool:
    """用 CLI 打开对应游戏项目。失败返回 False，不抛（调用方决定是否继续）。"""
    proj = workspace / PROJECTS[game]
    if not proj.is_dir():
        raise FileNotFoundError(f"项目目录不存在: {proj}")
    cmd = [DEVTOOLS_CLI, "open", "--project", str(proj)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[devtools] 打开 {game} 失败: {exc}", flush=True)
        return False
    blob = (out.stdout or "") + (out.stderr or "")
    if "✔ open" in blob:
        print(f"[devtools] 已打开 {game}: {proj}", flush=True)
        return True
    print(f"[devtools] 打开 {game} 未确认成功。输出尾部:\n{blob[-800:]}", flush=True)
    return False


# ---------------------------------------------------------------- 真实环境装配
@dataclass
class RealEnvBundle:
    conn: bridge.Connection
    session: Session
    env: WemiGameEnv


def make_real_env(cfg: Config, game: str, *, port: int,
                  connect_timeout_s: float = 600.0) -> RealEnvBundle:
    """起服务端并等客户端连上来。连不上就明确失败，不静默重试到天荒地老。"""
    host = str(cfg.env["host"])
    conn = bridge.Connection(host=host, port=port)
    conn.create_server()
    print(f"[env] {game} 服务端已启动 {host}:{conn.bound_port}，等待模拟器连接…", flush=True)
    t0 = time.monotonic()
    conn.waiting_client(timeout_s=connect_timeout_s)
    print(f"[env] 客户端已连接（{time.monotonic() - t0:.1f}s）", flush=True)

    sess = Session(
        connection=conn,
        game_id=game,
        step_timeout_s=float(cfg.env["step_timeout_s"]),
        connect_timeout_s=60.0,
        expected_env_hash=ENV_HASH[game],
    )
    info = sess.connect()
    print(f"[env] 握手成功: {json.dumps(info, ensure_ascii=False)}", flush=True)
    env = WemiGameEnv(game, sess, horizon=cfg.horizon)
    return RealEnvBundle(conn=conn, session=sess, env=env)


# ---------------------------------------------------------------- 训练器
class RealTrainer:
    """单阶段真实训练：随机采集 → 表示适配 → 初始离线 → 在线改进。"""

    def __init__(self, cfg: Config, game: str, bundle: RealEnvBundle, *,
                 run_root: Path, seed: int = 0) -> None:
        self.cfg = cfg
        self.game = game
        self.bundle = bundle
        self.env = bundle.env
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_root / "train_log.jsonl"

        self.model = ContinuousModel(seed=seed)
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
        ))
        self.replay = ReplayBuffer(
            per_task_capacity=int(cfg.replay["per_task_capacity"]),
            max_stored_transitions=int(cfg.replay["max_stored_transitions"]),
            gamma=cfg.gamma,
            batch_size=int(cfg.replay["batch_size"]),
            active_task_history_fraction=float(cfg.replay["active_task_history_fraction"]),
            seed=seed,
        )
        self.first_data: List[Dict[str, Any]] = []
        self.mapping_version = 0
        self.stats: Dict[str, Any] = {"online_steps": 0, "learner_updates": 0,
                                      "random_steps": 0, "repr_updates": 0,
                                      "offline_updates": 0, "episodes": [],
                                      "latencies": []}

    def log(self, rec: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # ---- 首次随机采集 ----
    def collect_first_data(self, steps: int) -> None:
        legal = F.legal_tokens(self.game)
        n = 0
        ep = 0
        while n < steps:
            x, info = self.env.reset(seed=int(self.rng.integers(0, 1 << 30)))
            ep_id = int(info["episode_id"])
            while n < steps:
                u = self.rng.normal(size=2).astype(np.float32)
                e = np.tanh(u).astype(np.float32)
                token = int(legal[n % len(legal)])       # 精确均衡（保证 ψ 可辨识）
                log_b = -float(np.log(len(legal)))
                local = F.local_action(self.game, token)
                nx, r, term, trunc, sinfo = self.env.step(local)
                n += 1
                self.first_data.append({
                    "x": x, "next_x": nx, "e_behavior": e, "global_token": token,
                    "reward_train": float(r), "log_b_action": log_b,
                    "episode_id": ep_id,
                })
                self.replay.add_confirmed(make_transition(
                    buffer=self.replay, game=self.game, episode_id=ep_id, request_id=n,
                    x=x, next_x=nx, e_behavior=e, global_token=token,
                    reward_train=float(r), log_b_action=log_b,
                    behavior_kernel="uniform", behavior_mapping_version=self.mapping_version,
                    terminated=term, truncated=trunc,
                    termination_reason=sinfo.termination_reason,
                    env_hash=ENV_HASH[self.game], clock_mode=self.cfg.env["clock_mode"],
                ), game=self.game)
                x = nx
                if term or trunc:
                    self.replay.finish_episode(
                        game=self.game, episode_id=ep_id, terminated=term,
                        truncated=trunc, termination_reason=sinfo.termination_reason)
                    ep += 1
                    break
                if n >= steps:
                    self.replay.finish_episode(
                        game=self.game, episode_id=ep_id, terminated=False,
                        truncated=True, termination_reason="none")
                    break
            self.stats["random_steps"] = n
            if n % 500 == 0:
                print(f"[collect] {n}/{steps} 步", flush=True)
        counts: Dict[int, int] = {}
        for r in self.first_data:
            counts[r["global_token"]] = counts.get(r["global_token"], 0) + 1
        self.log({"event": "first_data_done", "game": self.game, "steps": n,
                  "episodes": ep, "token_counts": counts})
        print(f"[collect] 完成 {n} 步，动作分布 {counts}", flush=True)

    # ---- 表示适配 ----
    def adapt(self, updates: int) -> int:
        x = torch.tensor(np.stack([r["x"] for r in self.first_data]), dtype=torch.float32)
        nx = torch.tensor(np.stack([r["next_x"] for r in self.first_data]), dtype=torch.float32)
        tok = torch.tensor([int(r["global_token"]) for r in self.first_data], dtype=torch.long)
        eps = [int(r["episode_id"]) for r in self.first_data]
        tr_idx, diag_idx = split_by_episode(eps, val_fraction=0.2, seed=self.seed)

        if self.game not in self.model.phi.registered_games():
            if self.model.phi.num_tokens == 0:
                self.model.register_game_as_first(self.game)
            else:
                self.model.register_game(self.game)

        batches = make_batches(
            x=x[tr_idx], next_x=nx[tr_idx], token=tok[tr_idx], game=self.game,
            batch_size=int(self.cfg.replay["batch_size"]), seed=self.seed)
        diag = adapt_representation(
            self.model, batches, updates=updates,
            lr=float(self.cfg.continuous["repr_lr"]),
            kl_weight=float(self.cfg.continuous["kl_weight"]),
            grad_clip=float(self.cfg.train["grad_clip"]))
        coverage = self.model.phi.coverage_grid(self.game, res=41)
        self.mapping_version += 1
        self.stats["repr_updates"] = updates
        rec = {
            "event": "adapt_done", "game": self.game,
            "mapping_version": self.mapping_version,
            "train_accuracy": diag.accuracy, "train_nll": diag.nll, "kl": diag.kl,
            "per_action_accuracy": diag.per_action_accuracy,
            "coverage_counts": coverage["counts"],
            "all_actions_reachable": coverage["all_actions_reachable"],
        }
        self.log(rec)
        print(f"[adapt] {json.dumps(rec, ensure_ascii=False)}", flush=True)
        return self.mapping_version

    # ---- 初始离线（首个任务一次）----
    def initial_offline(self, updates: int) -> int:
        x = torch.tensor(np.stack([r["x"] for r in self.first_data]), dtype=torch.float32)
        nx = torch.tensor(np.stack([r["next_x"] for r in self.first_data]), dtype=torch.float32)
        e = torch.tensor(np.stack([r["e_behavior"] for r in self.first_data]), dtype=torch.float32)
        tok = torch.tensor([int(r["global_token"]) for r in self.first_data], dtype=torch.long)
        rew = torch.tensor([float(r["reward_train"]) for r in self.first_data], dtype=torch.float32)
        logb = torch.tensor([float(r["log_b_action"]) for r in self.first_data], dtype=torch.float32)
        task = F.task_id(self.game)
        batch = {
            "x": x, "next_x": nx, "e_behavior": e, "global_token": tok,
            "local_action": torch.tensor([F.local_action(self.game, int(t)) for t in tok]),
            "reward": rew, "log_b_action": logb,
            "reference_return": torch.zeros_like(rew),
            "ref_valid": torch.zeros_like(rew, dtype=torch.bool),
            "task_id": torch.full((len(rew),), task, dtype=torch.long),
            "mapping_version": torch.full((len(rew),), self.mapping_version, dtype=torch.long),
            "terminated": torch.zeros(len(rew), dtype=torch.bool),
            "truncated": torch.zeros(len(rew), dtype=torch.bool),
            "games": [self.game] * len(rew),
        }
        set_trainable(self.model, SubPhase.INITIAL_OFFLINE)
        assert_frozen(self.model, SubPhase.INITIAL_OFFLINE, game=self.game)
        for i in range(updates):
            self.learner.update(batch)
            if (i + 1) % 200 == 0:
                print(f"[offline] {i+1}/{updates}", flush=True)
        self.stats["offline_updates"] = updates
        self.log({"event": "initial_offline_done", "game": self.game, "updates": updates})
        return updates

    # ---- 在线改进 ----
    def _reconnect(self, need_game: str, timeout_s: float = 300.0) -> None:
        """掉线后重连（§4.2 规则 4）：新 session_id、清空旧队列、reset 重开。

        关键：**丢弃没有确认 next_obs 的最后一步**，绝不盲目重发 step（会执行两次）。
        """
        print(f"[reconnect] 等待模拟器重新连接（最多 {timeout_s:.0f}s）…", flush=True)
        self.log({"event": "disconnected", "game": need_game,
                  "online_steps": self.stats["online_steps"]})
        conn = self.bundle.conn
        # 等旧连接彻底消失，再等新连接
        t0 = time.monotonic()
        while conn.has_client and time.monotonic() - t0 < 10.0:
            time.sleep(0.2)
        conn.waiting_client(timeout_s=timeout_s)
        # 重建 session（新 session_id），并清空残留消息
        self.bundle.session.reconnect()
        self.env = WemiGameEnv(need_game, self.bundle.session, horizon=self.cfg.horizon)
        self.bundle.env = self.env
        print("[reconnect] 已重新握手，回合从 reset 重新开始", flush=True)
        self.log({"event": "reconnected", "game": need_game})

    def _periodic_save(self, tag: str) -> None:
        try:
            self.save(self.run_root / f"ckpt_latest.pt", stage=tag)
            self.log({"event": "periodic_checkpoint", "step": self.stats["online_steps"]})
        except Exception as exc:      # 落盘失败不能中断训练，但必须记录
            print(f"[warn] 周期存盘失败: {exc}", flush=True)
            self.log({"event": "periodic_checkpoint_failed", "error": str(exc)})

    def online(self, steps: int, *, update_every: int, eval_env: Any = None,
               eval_every: int = 10000, eval_episodes: int = 10,
               stop_at: Optional[int] = None, save_every: int = 5000,
               reconnect_timeout_s: float = 300.0) -> None:
        set_trainable(self.model, SubPhase.ONLINE)
        assert_frozen(self.model, SubPhase.ONLINE, game=self.game)
        mix = float(self.cfg.continuous["decoder_behavior_uniform_mix"])
        done = 0
        ep_id = None
        x = None
        next_eval = eval_every
        while done < steps:
            if x is None:
                try:
                    x, info = self.env.reset(seed=int(self.rng.integers(0, 1 << 30)))
                except (NoClientError, EnvTimeout) as exc:
                    self.log({"event": "reset_failed", "error": type(exc).__name__,
                              "step": done})
                    print(f"[warn] reset 失败（{type(exc).__name__}），尝试重连…", flush=True)
                    self._reconnect(self.game, timeout_s=reconnect_timeout_s)
                    continue
                ep_id = int(info["episode_id"])
            xb = torch.as_tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0)
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
            local = F.local_action(self.game, token)
            try:
                nx, r, term, trunc, sinfo = self.env.step(local)
            except (NoClientError, EnvTimeout) as exc:
                # 掉线：丢弃这一条未确认的 transition，重连后从新回合继续（不重发动作）
                self.log({"event": "step_failed", "error": type(exc).__name__,
                          "detail": str(exc)[:200], "step": done})
                print(f"[warn] 交互失败（{type(exc).__name__}），尝试重连…", flush=True)
                self._reconnect(self.game, timeout_s=reconnect_timeout_s)
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
                env_hash=ENV_HASH[self.game], clock_mode=self.cfg.env["clock_mode"],
            ), game=self.game)
            x = nx

            if term or trunc:
                self.replay.finish_episode(
                    game=self.game, episode_id=ep_id, terminated=term, truncated=trunc,
                    termination_reason=sinfo.termination_reason)
                self.stats["episodes"].append({
                    "episode_id": ep_id, "score_raw": sinfo.score_raw,
                    "return_train": sinfo.episode_return_train,
                    "length": sinfo.steps_taken,
                    "reason": sinfo.termination_reason})
                x = None
                if sinfo.score_raw > 0:
                    print(f"[online] ep{ep_id} score={sinfo.score_raw:.0f} "
                          f"len={sinfo.steps_taken} reason={sinfo.termination_reason} "
                          f"step={done}/{steps}", flush=True)

            if done % update_every == 0 and self.replay.total_transitions() >= 8:
                batch = self._sample_batch()
                st = self.learner.update(batch)
                self.stats["learner_updates"] += 1
                if self.stats["learner_updates"] % 50 == 0:
                    self.log({
                        "event": "learner_update", "game": self.game, "step": done,
                        "td_loss": st.td_loss, "cal": st.cal_regularizer,
                        "actor_loss": st.actor_loss, "w_max": st.w_max,
                        "ess_over_batch": st.ess_over_batch})
                    print(f"[online] step={done}/{steps} upd={self.stats['learner_updates']} "
                          f"td={st.td_loss:.4f} cal={st.cal_regularizer:.4f} "
                          f"actor={st.actor_loss:.4f} w_max={st.w_max:.2f} "
                          f"ess={st.ess_over_batch:.2f}", flush=True)

            if save_every > 0 and done % save_every == 0:
                self._periodic_save(f"{self.game}@{done}")

            if done >= next_eval and eval_env is not None:
                rep = EV.evaluate_game(
                    cfg=self.cfg, model=self.model, game=self.game,
                    env_reset=eval_env.reset, env_step=eval_env.step,
                    policy_mode=self.cfg.eval["primary_policy_mode"],
                    episodes=eval_episodes,
                    seed_start=int(self.cfg.eval["validation_seed_start"]))
                self.log({"event": "validation", "game": self.game, "step": done,
                          "score_mean": rep.score_mean, "pass_count": rep.pass_count,
                          "episodes": rep.episodes, "return_mean": rep.return_mean})
                print(f"[eval] step={done} F/T={self.game} greedy score_mean="
                      f"{rep.score_mean:.2f} pass={rep.pass_count}/{rep.episodes}", flush=True)
                next_eval += eval_every
                x = None   # 评测已经 reset 过环境，必须重新开局
            if stop_at is not None and done >= stop_at:
                break

    def _sample_batch(self) -> Dict[str, Any]:
        trs = self.replay.sample(F.task_id(self.game))
        batch = self.replay.stack(trs, game_of_task={0: "Flappybird", 1: "Timberman"})
        batch["games"] = [GAMES[int(t)] for t in batch["task_id"].tolist()]
        return batch

    # ---- 保存 ----
    def save(self, path: Path, *, stage: str) -> Path:
        p = CK.save_checkpoint(
            path, cfg=self.cfg, model=self.model,
            optimizers={"q1": self.learner.opt_q1, "q2": self.learner.opt_q2,
                        "beta": self.learner.opt_beta},
            replay=self.replay,
            trainer_state={"game": self.game, "stage": stage,
                           "mapping_version": self.mapping_version,
                           "stats": {k: v for k, v in self.stats.items()
                                     if k not in ("latencies", "episodes")}},
            extra={"stats": {k: v for k, v in self.stats.items() if k != "latencies"},
                   "latency_summary": self._latency_summary()},
        )
        return p

    def _latency_summary(self) -> Dict[str, float]:
        if not self.stats["latencies"]:
            return {}
        arr = np.array(self.stats["latencies"], dtype=np.float64)
        return {"p50_ms": float(np.percentile(arr, 50) * 1000),
                "p95_ms": float(np.percentile(arr, 95) * 1000),
                "mean_ms": float(arr.mean() * 1000),
                "n": int(arr.size)}
