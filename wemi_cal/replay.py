"""Replay：事实字段、episode 管理、MC 回填、容量淘汰与两 schedule 采样（§7.2/§8.1/§8.4，任务 T5）。

必须保留的采样事实（§8.1）——任何一条都不能缺，missing/非法一律报错：

``sample_id, task_id, episode_id, request_id, x, e_behavior, global_token, local_action,
reward_train, next_x, score_raw, terminated, truncated, termination_reason,
behavior_id, behavior_kernel, behavior_mapping_version, log_b_action,
reference_return, ref_valid, env_hash, obs_schema, reward_schema, collection_split``

设计要点：
- 同一 batch 含旧 F 数据和新 T 数据 ≠ 两个游戏同时在线采样；桶按任务分开。
- 数据按 episode 管理；只有**完整结束**的回合才算 MC；外部截断保持无效。
- 淘汰以完整 episode 为单位，绝不把被截断的尾部补 0 冒充 MC。
- 不同 env_hash/clock/obs_schema/reward_schema 拒绝合并；
  同一游戏不同 mapping_version 允许共存，但必须保留行为概率并修正。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .envs import features as F

# ---- 契约字段（顺序即规范；缺失即报错）----
REQUIRED_FIELDS: Tuple[str, ...] = (
    "sample_id", "task_id", "episode_id", "request_id",
    "x", "e_behavior", "global_token", "local_action",
    "reward_train", "next_x", "score_raw",
    "terminated", "truncated", "termination_reason",
    "behavior_id", "behavior_kernel", "behavior_mapping_version", "log_b_action",
    "reference_return", "ref_valid",
    "env_hash", "obs_schema", "reward_schema", "collection_split",
)

BEHAVIOR_KERNELS = ("uniform", "decoder_exploration")
SCHEMAS = ("compact29-v1",)

# 与 mapping_version 无关的兼容性字段：不一致就拒绝合并（§8.4）
COMPATIBILITY_FIELDS = ("env_hash", "obs_schema", "reward_schema", "clock_mode")


class ReplayError(ValueError):
    """字段缺失、概率未知、非法 token、schema 不兼容。"""


@dataclass
class Transition:
    """一条已确认的 transition。"""

    sample_id: int
    task_id: int
    episode_id: int
    request_id: int
    x: np.ndarray
    e_behavior: np.ndarray
    global_token: int
    local_action: int
    reward_train: float
    next_x: np.ndarray
    score_raw: float
    terminated: bool
    truncated: bool
    termination_reason: str
    behavior_id: int
    behavior_kernel: str
    behavior_mapping_version: int
    log_b_action: float
    clock_mode: str
    reference_return: float = 0.0
    ref_valid: bool = False
    env_hash: Optional[str] = None
    obs_schema: str = "compact29-v1"
    reward_schema: str = "bounded-v1"
    collection_split: str = "train_only"

    # ---- 校验与转换 ----
    def validate(self, *, game: str) -> None:
        if self.clock_mode not in ("lockstep-v1", "realtime-v1"):
            raise ReplayError(f"未知 clock_mode: {self.clock_mode!r}")
        if self.behavior_kernel not in BEHAVIOR_KERNELS:
            raise ReplayError(f"未知 behavior_kernel: {self.behavior_kernel!r}")
        if self.obs_schema not in SCHEMAS:
            raise ReplayError(f"未知 obs_schema: {self.obs_schema!r}")
        if self.collection_split != "train_only":
            raise ReplayError(
                f"replay 只收训练数据，得到 collection_split={self.collection_split!r}"
            )
        expect_task = F.task_id(game)
        if self.task_id != expect_task:
            raise ReplayError(f"task_id 与游戏不符: {self.task_id} vs {expect_task}")
        if F.local_action(game, self.global_token) != self.local_action:
            raise ReplayError(
                f"token {self.global_token} 在 {game} 上的本地动作应为 "
                f"{F.local_action(game, self.global_token)}，得到 {self.local_action}"
            )
        if self.x.shape != (F.OBS_DIM,) or self.next_x.shape != (F.OBS_DIM,):
            raise ReplayError(f"x/next_x 必须是 [{F.OBS_DIM}]")
        if self.e_behavior.shape != (2,):
            raise ReplayError("e_behavior 必须是 [2]")
        if not np.isfinite(self.x).all() or not np.isfinite(self.next_x).all():
            raise ReplayError("x/next_x 含 NaN/Inf")
        if not np.isfinite(self.e_behavior).all():
            raise ReplayError("e_behavior 含 NaN/Inf")
        if not np.isfinite(self.log_b_action):
            raise ReplayError("log_b_action 不是有限值：行为概率记录有误")
        if self.log_b_action > 1e-9:
            raise ReplayError(f"log_b_action 必须 ≤ 0（概率 ≤ 1），得到 {self.log_b_action}")
        if not np.isfinite(self.reward_train):
            raise ReplayError("reward_train 含 NaN/Inf")
        if abs(self.reward_train) > 1.0 + 1e-9:
            raise ReplayError(f"reward_train 必须在 [-1,1]，得到 {self.reward_train}")
        if self.termination_reason not in ("none", "collision", "ground", "blood", "horizon"):
            raise ReplayError(f"未知 termination_reason: {self.termination_reason!r}")
        if self.terminated and self.truncated:
            raise ReplayError("terminated 与 truncated 不能同时为真（§4.5）")
        if self.ref_valid and not self.terminated:
            raise ReplayError("ref_valid=True 只允许出现在完整结束的回合末（§7.2）")

    @property
    def died(self) -> bool:
        return self.terminated and self.termination_reason in ("collision", "ground", "blood")


@dataclass
class Episode:
    """一个回合的所有已确认 transition（按时间顺序）。"""

    episode_id: int
    game: str
    transitions: List[Transition] = field(default_factory=list)
    completed: bool = False
    termination_reason: str = "none"
    mc_filled: bool = False

    @property
    def task_id(self) -> int:
        return F.task_id(self.game)

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def first_sample_id(self) -> int:
        return self.transitions[0].sample_id if self.transitions else -1


class ReplayBuffer:
    """按任务分桶的 episode 存储；支持 strict / rehearsal 两种组织方式。"""

    def __init__(
        self,
        *,
        per_task_capacity: int = 20000,
        max_stored_transitions: int = 40000,
        gamma: float = 0.99,
        batch_size: int = 128,
        active_task_history_fraction: float = 0.5,
        seed: int = 0,
    ) -> None:
        if per_task_capacity <= 0 or max_stored_transitions <= 0:
            raise ReplayError("容量必须为正")
        if not (0.0 < active_task_history_fraction < 1.0):
            raise ReplayError("active_task_history_fraction 必须在 (0,1)")
        self.per_task_capacity = int(per_task_capacity)
        self.max_stored_transitions = int(max_stored_transitions)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.active_task_history_fraction = float(active_task_history_fraction)
        self.rng = np.random.default_rng(seed)

        # task_id → 完整回合桶 / 活跃在线回合桶
        self.historical: Dict[int, List[Episode]] = {0: [], 1: []}
        self.online: Dict[int, List[Episode]] = {0: [], 1: []}
        # (task_id, episode_id) → 尚未结束的回合
        self.pending: Dict[Tuple[int, int], Episode] = {}
        self.compat: Dict[str, Any] = {}
        self._next_sample_id = 0
        self.evictions = 0
        self.rejected_finish = 0

    def get_rng_state(self) -> Dict[str, Any]:
        """返回 replay 抽样器状态，供完整训练 checkpoint 使用。"""
        return self.rng.bit_generator.state

    def set_rng_state(self, state: Dict[str, Any]) -> None:
        """恢复 replay 抽样器状态。"""
        self.rng.bit_generator.state = state

    # ---------------------------------------------------------------- 写入
    def add_confirmed(self, tr: Transition, *, game: str) -> None:
        """只收已确认的 transition；先以 ref_valid=False 入账（§13）。"""
        tr.validate(game=game)
        self._check_compat(tr)
        key = (tr.task_id, tr.episode_id)
        ep = self.pending.get(key)
        if ep is None:
            if tr.episode_id in [e.episode_id for e in self.online[tr.task_id]]:
                raise ReplayError(
                    f"回合 {tr.episode_id} 已结束，不能再追加 transition"
                )
            ep = Episode(episode_id=tr.episode_id, game=game)
            self.pending[key] = ep
        elif ep.completed:
            raise ReplayError(f"回合 {tr.episode_id} 已完成，拒绝追加")
        ep.transitions.append(tr)

    def finish_episode(
        self, *, game: str, episode_id: int, terminated: bool, truncated: bool,
        termination_reason: str,
    ) -> Episode:
        """回合结束：完整结束才回填 G；外部截断保持 invalid（§13）。"""
        task = F.task_id(game)
        key = (task, episode_id)
        ep = self.pending.pop(key, None)
        if ep is None:
            raise ReplayError(f"没有待结束的回合 {(task, episode_id)}")
        if terminated and truncated:
            raise ReplayError("terminated 与 truncated 不能同时为真")
        ep.completed = True
        ep.termination_reason = termination_reason
        complete = bool(terminated) and not truncated
        if complete:
            self._fill_reference_returns(ep)
        else:
            ep.mc_filled = False
            self.rejected_finish += 1
        # 结束后进入"在线"桶；阶段切换时由 commit_online_to_history 固化
        self.online[task].append(ep)
        self._evict_if_needed(task)
        return ep

    def _fill_reference_returns(self, ep: Episode) -> None:
        """G_T=0、G_t = r_t + γ G_{t+1}；逐条写 reference_return 与 ref_valid（§7.2）。

        只对**完整结束**（自然死亡或到 H）的回合做；奖励必须是训练奖励。
        """
        if not ep.transitions:
            raise ReplayError("空回合不能回填 MC")
        g = 0.0
        for tr in reversed(ep.transitions):
            g = tr.reward_train + self.gamma * g
            tr.reference_return = g
            tr.ref_valid = True
        ep.mc_filled = True

    # ---------------------------------------------------------------- 兼容性
    def _check_compat(self, tr: Transition) -> None:
        """跨游戏 buffer 的兼容性检查。

        ``env_hash`` 必须**按任务**分别记录：两个不同游戏的 env_hash 天然不同，
        用单一全局值会错误地拒绝合法数据；而同一游戏内部 hash 变了（改了 JS 逻辑/
        lockstep 参数/特征或奖励版本）仍然必须拒绝合并。
        """
        for f in COMPATIBILITY_FIELDS:
            val = getattr(tr, f, None)
            if val is None:
                continue
            if f == "env_hash":
                key = f"env_hash:{tr.task_id}"
                if key in self.compat and self.compat[key] != val:
                    raise ReplayError(
                        f"task {tr.task_id} 的 env_hash 不一致"
                        f"（{self.compat[key]!r} vs {val!r}）：拒绝合并不同环境版本的 replay"
                    )
                self.compat.setdefault(key, val)
                continue
            if f in self.compat and self.compat[f] != val:
                raise ReplayError(
                    f"{f} 不一致（{self.compat[f]!r} vs {val!r}）：拒绝合并不同环境的 replay"
                )
            self.compat.setdefault(f, val)

    # ---------------------------------------------------------------- 容量
    def _task_transitions(self, task: int) -> int:
        return sum(len(e) for e in self.historical[task]) + sum(len(e) for e in self.online[task])

    def _evict_if_needed(self, task: int) -> None:
        """淘汰最旧的**完整** episode；先历史后在线（§8.4）。"""
        while self._task_transitions(task) > self.per_task_capacity:
            removed = False
            for bucket in (self.historical[task], self.online[task]):
                idx = next((i for i, e in enumerate(bucket) if e.completed), None)
                if idx is not None:
                    bucket.pop(idx)
                    self.evictions += 1
                    removed = True
                    break
            if not removed:
                break   # 只剩未完成回合：由 pending 上限另行处理

    def total_transitions(self) -> int:
        n = sum(len(e) for b in (self.historical[0], self.historical[1],
                                 self.online[0], self.online[1]) for e in b)
        n += sum(len(e) for e in self.pending.values())
        return n

    def pending_transitions(self) -> int:
        return sum(len(e) for e in self.pending.values())

    # ---------------------------------------------------------------- 阶段切换
    def commit_online_to_history(self, task: int) -> None:
        """阶段结束：把该任务 online 合入历史并清空 online（§8.4 rehearsal）。"""
        self.historical[task].extend(self.online[task])
        self.online[task] = []
        self._evict_if_needed(task)

    def freeze_historical(self, task: int) -> None:
        """strict 的 S1→S2：F 的在线数据固化为历史；此后不得再补写（§8.4）。"""
        if any(k[0] == task for k in self.pending):
            raise ReplayError("还有未完成回合（pending）：固化前必须先把 pending 处理掉")
        if any(not e.completed for e in self.online[task]):
            raise ReplayError("online 桶里还有未完成回合")
        self.commit_online_to_history(task)

    # ---------------------------------------------------------------- 采样
    def _pool(self, task: int, *, use_history: bool, use_online: bool) -> List[Transition]:
        out: List[Transition] = []
        if use_history:
            for e in self.historical[task]:
                out.extend(e.transitions)
        if use_online:
            for e in self.online[task]:
                out.extend(e.transitions)
        return out

    def sample_plan(self, active_task: int) -> List[Tuple[int, str, int]]:
        """返回 [(task, bucket, count)]，bucket ∈ {'historical','online'}。

        规则（§8.4）：
        - strict S1（只有一个任务有数据）：全给活跃任务。
        - 双任务：非活跃任务取历史 64；活跃任务历史/当前同时存在时各 32，否则取 64。
        """
        other = 1 - active_task
        other_pool = self._pool(other, use_history=True, use_online=False)
        if not other_pool:
            # 单任务阶段：batch 全部给活跃任务
            if not self._pool(active_task, use_history=True, use_online=True):
                return []
            n = self.batch_size
            hp = self._pool(active_task, use_history=True, use_online=False)
            op = self._pool(active_task, use_history=False, use_online=True)
            if hp and op:
                h = int(round(n * self.active_task_history_fraction))
                return [(active_task, "historical", h), (active_task, "online", n - h)]
            return [(active_task, "historical" if hp else "online", n)]

        half = self.batch_size // 2
        plan: List[Tuple[int, str, int]] = [(other, "historical", half)]
        hp = self._pool(active_task, use_history=True, use_online=False)
        op = self._pool(active_task, use_history=False, use_online=True)
        if hp and op:
            h = int(round(half * self.active_task_history_fraction))
            plan.append((active_task, "historical", h))
            plan.append((active_task, "online", half - h))
        elif hp:
            plan.append((active_task, "historical", half))
        elif op:
            plan.append((active_task, "online", half))
        return plan

    def sample(self, active_task: int) -> List[Transition]:
        """按 plan 采样；非空小桶可有放回采样（§8.4）。"""
        plan = self.sample_plan(active_task)
        if not plan:
            raise ReplayError("replay 为空，无法采样")
        out: List[Transition] = []
        for task, bucket, count in plan:
            pool = self._pool(
                task,
                use_history=(bucket == "historical"),
                use_online=(bucket == "online"),
            )
            if not pool:
                raise ReplayError(f"计划要求 {count} 条 {task}/{bucket}，但该桶为空")
            replace = len(pool) < count
            idx = self.rng.choice(len(pool), size=count, replace=replace)
            out.extend(pool[int(i)] for i in idx)
        return out

    def sample_bucket(self, task: int, bucket: str, count: int) -> List[Transition]:
        """从指定桶无放回采样；严格配比不能用替补或跨桶样本凑数。"""
        if bucket not in ("historical", "online"):
            raise ReplayError(f"未知 replay 桶: {bucket!r}")
        pool = self._pool(task, use_history=bucket == "historical",
                          use_online=bucket == "online")
        if len(pool) < count:
            raise ReplayError(
                f"replay 来源不足: task={task}/{bucket} 需要 {count} 条，实际 {len(pool)} 条"
            )
        idx = self.rng.choice(len(pool), size=count, replace=False)
        return [pool[int(i)] for i in idx]

    @staticmethod
    def task_counts(transitions: Sequence[Transition]) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for tr in transitions:
            counts[tr.task_id] = counts.get(tr.task_id, 0) + 1
        return counts

    def assert_task_counts(
        self, transitions: Sequence[Transition], expected: Mapping[int, int]
    ) -> None:
        actual = self.task_counts(transitions)
        wanted = dict(expected)
        if actual != wanted:
            raise ReplayError(f"batch 任务构成错误: 期望 {wanted}，得到 {actual}")

    # ---------------------------------------------------------------- 诊断
    def diagnostics(self) -> Dict[str, Any]:
        return {
            "historical": {t: len(self.historical[t]) for t in (0, 1)},
            "online": {t: len(self.online[t]) for t in (0, 1)},
            "pending_episodes": len(self.pending),
            "total_transitions": self.total_transitions(),
            "pending_transitions": self.pending_transitions(),
            "evictions": self.evictions,
            "truncated_episodes_without_mc": self.rejected_finish,
            "compat": dict(self.compat),
        }

    # ---------------------------------------------------------------- 批次堆叠
    def stack(self, transitions: Sequence[Transition], *, game_of_task: Dict[int, str],
              device: Any = "cpu") -> Dict[str, Any]:
        """把 transition 列表堆成张量 dict。不重写行为概率字段（§13）。"""
        import torch

        n = len(transitions)
        x = torch.zeros(n, F.OBS_DIM, dtype=torch.float32, device=device)
        nx = torch.zeros(n, F.OBS_DIM, dtype=torch.float32, device=device)
        e = torch.zeros(n, 2, dtype=torch.float32, device=device)
        tok = torch.zeros(n, dtype=torch.long, device=device)
        loc = torch.zeros(n, dtype=torch.long, device=device)
        rew = torch.zeros(n, dtype=torch.float32, device=device)
        logb = torch.zeros(n, dtype=torch.float32, device=device)
        ref = torch.zeros(n, dtype=torch.float32, device=device)
        ref_valid = torch.zeros(n, dtype=torch.bool, device=device)
        tasks = torch.zeros(n, dtype=torch.long, device=device)
        mver = torch.zeros(n, dtype=torch.long, device=device)
        term = torch.zeros(n, dtype=torch.bool, device=device)
        trunc = torch.zeros(n, dtype=torch.bool, device=device)
        sample_ids = []
        games: List[str] = []
        for i, tr in enumerate(transitions):
            x[i] = torch.as_tensor(tr.x, dtype=torch.float32)
            nx[i] = torch.as_tensor(tr.next_x, dtype=torch.float32)
            e[i] = torch.as_tensor(tr.e_behavior, dtype=torch.float32)
            tok[i] = tr.global_token
            loc[i] = tr.local_action
            rew[i] = tr.reward_train
            logb[i] = tr.log_b_action
            ref[i] = tr.reference_return
            ref_valid[i] = tr.ref_valid
            tasks[i] = tr.task_id
            mver[i] = tr.behavior_mapping_version
            term[i] = tr.terminated
            trunc[i] = tr.truncated
            sample_ids.append(tr.sample_id)
            games.append(game_of_task[tr.task_id])
        return {
            "x": x, "next_x": nx, "e_behavior": e, "global_token": tok,
            "local_action": loc, "reward": rew, "log_b_action": logb,
            "reference_return": ref, "ref_valid": ref_valid, "task_id": tasks,
            "mapping_version": mver, "terminated": term, "truncated": trunc,
            "sample_ids": sample_ids, "games": games, "size": n,
        }

    def next_sample_id(self) -> int:
        s = self._next_sample_id
        self._next_sample_id += 1
        return s


def make_transition(
    *,
    buffer: ReplayBuffer,
    game: str,
    episode_id: int,
    request_id: int,
    x: np.ndarray,
    next_x: np.ndarray,
    e_behavior: np.ndarray,
    global_token: int,
    reward_train: float,
    log_b_action: float,
    behavior_kernel: str,
    behavior_mapping_version: int,
    behavior_id: int = 0,
    score_raw: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    termination_reason: str = "none",
    env_hash: Optional[str] = None,
    clock_mode: str,
) -> Transition:
    """便捷构造：自动补 sample_id/task_id/local_action 等派生字段。"""
    tr = Transition(
        sample_id=buffer.next_sample_id(),
        task_id=F.task_id(game),
        episode_id=int(episode_id),
        request_id=int(request_id),
        x=np.asarray(x, dtype=np.float32),
        e_behavior=np.asarray(e_behavior, dtype=np.float32),
        global_token=int(global_token),
        local_action=F.local_action(game, int(global_token)),
        reward_train=float(reward_train),
        next_x=np.asarray(next_x, dtype=np.float32),
        score_raw=float(score_raw),
        terminated=bool(terminated),
        truncated=bool(truncated),
        termination_reason=str(termination_reason),
        behavior_id=int(behavior_id),
        behavior_kernel=str(behavior_kernel),
        behavior_mapping_version=int(behavior_mapping_version),
        log_b_action=float(log_b_action),
        env_hash=env_hash,
        clock_mode=str(clock_mode),
    )
    return tr
