"""Gymnasium 风格的环境契约（技术方案 §13，任务 T3/T7）。

一个 ``WemiGameEnv`` 对应一个游戏 + 一个端口。它负责把 session 的原始响应
翻译成策略需要的 29 维状态、训练奖励与终止语义，并维护 H=1000 的有限时域。

关键约定：
- ``reset(seed)`` 返回 ``(x[29], info)``；新回合剩余时域=1。
- ``step(local_action)`` 返回 ``(next_x, r, terminated, truncated, info)``，
  且只返回"真实已确认"的结果。
- horizon 终止不是死亡；外部截断（阶段预算耗尽）保留 bootstrap。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import features as F
from .session import Session


@dataclass
class EnvStepInfo:
    """一步的完整诊断信息；日志与 replay 都从这里取。"""

    game: str
    steps_taken: int
    remaining_budget: float
    score_raw: float
    reward_train: float
    terminated: bool
    truncated: bool
    termination_reason: str
    died: bool
    sim_ticks: int
    latency_s: float
    episode_return_train: float = 0.0
    episode_id: int = -1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "game": self.game,
            "steps_taken": self.steps_taken,
            "remaining_budget": self.remaining_budget,
            "score_raw": self.score_raw,
            "reward_train": self.reward_train,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
            "died": self.died,
            "sim_ticks": self.sim_ticks,
            "latency_s": self.latency_s,
            "episode_return_train": self.episode_return_train,
            "episode_id": self.episode_id,
        }


class WemiGameEnv:
    """单游戏 lockstep 环境。"""

    def __init__(
        self,
        game: str,
        session: Session,
        *,
        horizon: int = 1000,
        obs_dtype: Any = np.float32,
    ) -> None:
        if game not in F.GAMES:
            raise F.FeatureError(f"未知游戏: {game!r}")
        if horizon <= 0:
            raise F.FeatureError("horizon 必须为正")
        self.game = game
        self.session = session
        self.horizon = int(horizon)
        self.obs_dtype = obs_dtype

        self.steps_taken = 0
        self.score_raw = 0.0
        self.episode_return_train = 0.0
        self.episode_id = -1
        self.ended = True
        self._last_obs: Optional[Dict[str, Any]] = None

    # ---------- 生命周期 ----------
    @property
    def action_mask(self) -> np.ndarray:
        return F.action_mask(self.game)

    @property
    def legal_tokens(self) -> Tuple[int, ...]:
        return F.legal_tokens(self.game)

    def reset(self, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        resp = self.session.reset(seed)
        # §5.3：reset 后剩余时域=1；这里的 steps_taken=0 表示"还没执行过动作"
        self.steps_taken = 0
        self.score_raw = resp.score
        self.episode_return_train = 0.0
        self.episode_id = int(resp.raw["episode_id"])
        self.ended = False
        self._last_obs = resp.obs
        x = F.encode_observation(
            self.game, resp.obs, steps_taken=self.steps_taken, horizon=self.horizon,
            dtype=self.obs_dtype,
        )
        return x, {
            "game": self.game,
            "episode_id": self.episode_id,
            "steps_taken": self.steps_taken,   # reset 为 0
            "score_raw": self.score_raw,
            "env_hash": resp.info.get("env_hash"),
            "sim_ticks": resp.sim_ticks,
        }

    def step(
        self, local_action: int, *, externally_truncated: bool = False
    ) -> Tuple[np.ndarray, float, bool, bool, EnvStepInfo]:
        """执行一次本地动作，对应一个 agent step。

        H=1000 表示一回合最多 1000 个动作：第 1000 次 step 后预算到 0，
        该步 terminated=True、reason='horizon'、truncated=False（§4.5）。

        ``externally_truncated=True`` 用于阶段预算耗尽：本步结论标记为操作性截断，
        调用方不应再多采一步。
        """
        if self.ended:
            raise RuntimeError("回合已经结束，必须先 reset")
        score_before = self.score_raw
        resp = self.session.step(local_action)

        # 先按已确认的响应判定终止语义，再算奖励（死亡步的奖励要用 died）
        steps_after = self.steps_taken + 1
        term = F.classify_termination(
            response=resp,
            steps_taken=steps_after,
            horizon=self.horizon,
            externally_truncated=externally_truncated,
        )
        reward = F.train_reward(self.game, resp, score_before=score_before, died=term.died)

        self.steps_taken = steps_after
        self.score_raw = resp.score
        self.episode_return_train += reward
        self._last_obs = resp.obs
        self.ended = term.terminated or term.truncated

        x = F.encode_observation(
            self.game, resp.obs, steps_taken=self.steps_taken, horizon=self.horizon,
            dtype=self.obs_dtype,
        )
        info = EnvStepInfo(
            game=self.game,
            steps_taken=self.steps_taken,
            remaining_budget=(self.horizon - self.steps_taken) / self.horizon,
            score_raw=self.score_raw,
            reward_train=reward,
            terminated=term.terminated,
            truncated=term.truncated,
            termination_reason=term.reason,
            died=term.died,
            sim_ticks=resp.sim_ticks,
            latency_s=resp.latency_s,
            episode_return_train=self.episode_return_train,
            episode_id=self.episode_id,
        )
        return x, reward, term.terminated, term.truncated, info

    # ---------- 诊断 ----------
    @property
    def last_observation(self) -> Optional[Dict[str, Any]]:
        return self._last_obs

    def close(self) -> None:
        self.session.close()
