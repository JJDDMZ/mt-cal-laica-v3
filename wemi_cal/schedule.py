"""训练状态机：首次采集 → 表示适配 → 初始离线 → 在线改进 → 阶段结束（§9，任务 T7）。

阶段内流程（每个阶段 = 一个 game + 一段步数预算）：

    collect_first_data(前 2000 步均匀动作，不调用 φ)
      → adapt(φ/ψ 各 1000 updates，发布新 mapping_version)
      → initial_offline(仅首个任务：β/Q 1000 updates，且只做一次)
      → online(每 4 个新交互做一次 learner update；回合内冻结 rollout β)

冻结表（§9.3）在 ``FROZEN_BY_SUBPHASE`` 里显式声明并由 ``assert_frozen`` 强制检查：
这不是注释，而是运行时会报错的不变量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .envs import bridge, features as F
from .envs.session import Session
from .models.continuous import ContinuousModel, freeze_parameters
from .replay import ReplayBuffer

# ---------------------------------------------------------------- 子阶段
class SubPhase(str, Enum):
    FIRST_DATA = "first_data"
    ADAPT = "adapt"
    INITIAL_OFFLINE = "initial_offline"
    ONLINE = "online"
    EVAL = "eval"
    DONE = "done"


# §9.3 冻结表：每个子阶段哪些模块可更新 / 可交互。
# C 与 D 的表示模块名不同（C: phi/psi；D: actor/codebook 的当前游戏行 + psi），
# 因此按 variant 分别给出模块名；语义完全一致（§16.2 末尾说明 D 沿用同一套冻结规则）。
FROZEN_BY_SUBPHASE: Dict[SubPhase, Dict[str, Any]] = {
    SubPhase.FIRST_DATA: {
        "trainable": (), "trainable_discrete": (),
        "new_game_interaction": "uniform",
    },
    SubPhase.ADAPT: {
        "trainable": ("phi", "psi"), "trainable_discrete": ("codebook", "psi"),
        "new_game_interaction": "none",
    },
    SubPhase.INITIAL_OFFLINE: {
        "trainable": ("beta", "q"), "trainable_discrete": ("actor", "q"),
        "new_game_interaction": "none",
    },
    SubPhase.ONLINE: {
        "trainable": ("beta", "q"), "trainable_discrete": ("actor", "q"),
        "new_game_interaction": "current_stage_game_only",
    },
    SubPhase.EVAL: {
        "trainable": (), "trainable_discrete": (),
        "new_game_interaction": "eval_only",
    },
}


def trainable_modules(subphase: SubPhase, variant: str) -> Tuple[str, ...]:
    """取该 variant 在该子阶段的可训练模块名。"""
    spec = FROZEN_BY_SUBPHASE[subphase]
    key = "trainable_discrete" if variant == "discrete" else "trainable"
    return tuple(spec[key])

# 模块名 → 参数名前缀。C 与 D 的参数命名不同，必须分开，
# 否则 D 的 actor 会因为不在 C 的前缀表里而被整体冻结（曾真实发生）。
MODULE_PREFIXES = {
    "continuous": {
        "beta": ("beta.",),
        "phi": ("phi.",),
        "psi": ("psi.",),
        "q": ("q1.", "q2.", "target_q1.", "target_q2."),
    },
    "discrete": {
        "actor": ("actor.",),
        "codebook": ("codebook.",),
        "psi": ("psi.",),
        "q": ("q1.", "q2.", "target_q1.", "target_q2."),
    },
}


def _variant_of(model: Any) -> str:
    """按模型自带的属性判断 variant，不靠猜类名。"""
    v = getattr(model, "variant_name", None)
    if v in MODULE_PREFIXES:
        return v
    # 兜底：有 phi 就是 C，有 actor/codebook 就是 D
    if hasattr(model, "phi"):
        return "continuous"
    if hasattr(model, "actor"):
        return "discrete"
    raise ScheduleError(f"无法判断 model 的 variant: {type(model).__name__}")


def _prefixes(model: Any, names: Sequence[str]) -> Tuple[str, ...]:
    table = MODULE_PREFIXES[_variant_of(model)]
    out: list = []
    for n in names:
        if n not in table:
            raise ScheduleError(
                f"冻结表里的模块 {n!r} 在 {_variant_of(model)} 的命名表里不存在；"
                f"可用: {sorted(table)}"
            )
        out.extend(table[n])
    return tuple(out)


class ScheduleError(RuntimeError):
    """状态机被违反（阶段顺序、冻结表、预算）。"""


def assert_frozen(model: Any, subphase: SubPhase, *, game: str) -> None:
    """按冻结表断言：该阶段禁止更新的模块不能带 requires_grad=True。"""
    names = trainable_modules(subphase, _variant_of(model))
    trainable_prefixes = _prefixes(model, names)
    for name, param in model.named_parameters():
        # 关键：前缀为空时**不能**用 startswith(())（它对任何字符串都为真，
        # 会把整个模型设成可训练）。
        should_train = bool(trainable_prefixes) and name.startswith(trainable_prefixes)
        if param.requires_grad and not should_train:
            raise ScheduleError(
                f"{subphase.value} 阶段 {name} 不应可训练（冻结表：trainable={names}）"
            )


def set_trainable(model: Any, subphase: SubPhase) -> None:
    """按冻结表设置 requires_grad（C 与 D 各自一套前缀）。"""
    names = trainable_modules(subphase, _variant_of(model))
    trainable_prefixes = _prefixes(model, names)
    for name, param in model.named_parameters():
        param.requires_grad_(
            bool(trainable_prefixes) and name.startswith(trainable_prefixes))


# ---------------------------------------------------------------- 阶段结果
@dataclass
class StageResult:
    game: str
    first_data_steps: int = 0
    adapt_updates: int = 0
    initial_offline_updates: int = 0
    online_steps: int = 0
    learner_updates: int = 0
    mapping_version: int = 0
    score_raw_mean: float = 0.0
    episode_returns: List[float] = field(default_factory=list)
    terminations: Dict[str, int] = field(default_factory=dict)
    truncated_tail: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "game": self.game,
            "first_data_steps": self.first_data_steps,
            "adapt_updates": self.adapt_updates,
            "initial_offline_updates": self.initial_offline_updates,
            "online_steps": self.online_steps,
            "learner_updates": self.learner_updates,
            "mapping_version": self.mapping_version,
            "score_raw_mean": self.score_raw_mean,
            "episodes": len(self.episode_returns),
            "terminations": dict(self.terminations),
            "truncated_tail": self.truncated_tail,
        }


@dataclass
class TrainerCallbacks:
    """把"具体怎么学"注入状态机；测试时可替换为假实现。"""

    collect_first_data: Callable[[str, int, int], Any]
    adapt_representation: Callable[[str, int], int]
    learner_update: Callable[[int], Dict[str, Any]]
    initial_offline: Callable[[str, int], int]
    env_reset: Callable[[str, int], Tuple[np.ndarray, Dict[str, Any]]]
    env_step: Callable[[str, int], Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]]
    choose_action: Callable[[str, int, Any], Tuple[int, int, float, np.ndarray, int]]
    on_episode_end: Callable[[str, int, bool, bool, str], None]
    on_transition: Callable[..., None]


class Trainer:
    """执行 strict / rehearsal 的阶段序列。"""

    def __init__(
        self,
        *,
        cfg: Any,
        model: ContinuousModel,
        callbacks: TrainerCallbacks,
        horizon: int,
        logger: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_stage_end: Optional[Callable[[str, int, StageResult], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.cb = callbacks
        self.horizon = int(horizon)
        self.log = logger or (lambda rec: None)
        # 阶段收尾钩子（例如把在线数据固化进历史桶 §8.4）
        self.on_stage_end = on_stage_end
        self.subphase = SubPhase.FIRST_DATA
        self.mapping_version = 0
        self.offline_done = False
        self.log_records: List[Dict[str, Any]] = []
        self.counters = {
            "random_steps": 0, "online_steps": 0, "learner_updates": 0,
            "repr_updates": 0, "offline_updates": 0, "adapt_events": 0,
        }

    # ---- 日志 ----
    def _record(self, **kw: Any) -> None:
        rec = {"subphase": self.subphase.value, **self.counters, **kw}
        self.log_records.append(rec)
        self.log(rec)

    def _enter(self, phase: SubPhase, *, game: str) -> None:
        self.subphase = phase
        set_trainable(self.model, phase)
        assert_frozen(self.model, phase, game=game)

    # ---- 单个阶段 ----
    def run_stage(self, game: str, steps: int, *, first_visit: bool) -> StageResult:
        """跑一个阶段：首次接入含随机采集+适配（+首个任务的离线），否则直接在线。"""
        if steps <= 0:
            raise ScheduleError("阶段步数必须为正")
        res = StageResult(game=game)
        cfg = self.cfg
        random_steps = int(cfg.train["random_steps_first_visit"])
        repr_updates = int(cfg.train["repr_updates_first_visit"])
        offline_updates = int(cfg.train["initial_offline_updates"])
        update_every = int(cfg.train["update_every_online_steps"])

        online_target = steps
        if first_visit:
            if random_steps >= steps:
                raise ScheduleError(
                    f"阶段步数 {steps} 不足以覆盖首次随机采集 {random_steps} 步"
                )
            # 1) 首次随机采集：均匀动作，不调用尚未扩展的 φ
            self._enter(SubPhase.FIRST_DATA, game=game)
            self.cb.collect_first_data(game, random_steps, self.mapping_version)
            self.counters["random_steps"] += random_steps
            res.first_data_steps = random_steps
            self._record(event="first_data_done", game=game, steps=random_steps)

            # 2) 表示适配：φ/ψ 更新，发布新 mapping_version
            self._enter(SubPhase.ADAPT, game=game)
            self.mapping_version = int(self.cb.adapt_representation(game, repr_updates))
            self.counters["repr_updates"] += repr_updates
            self.counters["adapt_events"] += 1
            res.adapt_updates = repr_updates
            res.mapping_version = self.mapping_version
            self._record(event="adapt_done", game=game, updates=repr_updates,
                         mapping_version=self.mapping_version)

        # 3) 仅首个任务做一次初始离线 β/Q 更新（§6.5）
        if first_visit and not self.offline_done:
            self._enter(SubPhase.INITIAL_OFFLINE, game=game)
            n = int(self.cb.initial_offline(game, offline_updates))
            self.counters["offline_updates"] += n
            res.initial_offline_updates = n
            self.offline_done = True
            self._record(event="initial_offline_done", game=game, updates=n)

        # 4) 在线改进
        self._enter(SubPhase.ONLINE, game=game)
        online_target = steps - res.first_data_steps
        if online_target <= 0:
            raise ScheduleError("在线阶段没有剩余步数")
        used = self._run_online(game, online_target, update_every, res)
        res.online_steps = used
        res.mapping_version = self.mapping_version
        self._record(event="stage_end", game=game, online_steps=used)
        if self.on_stage_end is not None:
            self.on_stage_end(game, self.mapping_version, res)
        return res

    def _run_online(self, game: str, budget: int, update_every: int, res: StageResult) -> int:
        """在线采样循环：每 4 个新交互做一次 learner update；回合内冻结 rollout β。

        预算耗尽时若回合未结束：标操作性截断并**不多采**（§9.3/§4.5）。
        """
        steps_done = 0
        while steps_done < budget:
            seed = 100000 + steps_done
            x, info = self.cb.env_reset(game, seed)
            # 回合内冻结 rollout β 由 callbacks.choose_action 负责（传入 snapshot id）
            rollout_id = steps_done
            episode_done = False
            while steps_done < budget and not episode_done:
                token, local, log_b, e_beh, behavior_id = self.cb.choose_action(
                    game, self.mapping_version, rollout_id
                )
                next_x, r, term, trunc, step_info = self.cb.env_step(game, local)
                steps_done += 1
                self.counters["online_steps"] += 1
                self.cb.on_transition(
                    game=game, x=x, next_x=next_x, token=token, local=local,
                    r=r, log_b=log_b, e_beh=e_beh, behavior_id=behavior_id,
                    mapping_version=self.mapping_version,
                    terminated=term, truncated=trunc,
                    termination_reason=step_info.get("termination_reason", "none"),
                )
                x = next_x
                if term or trunc:
                    episode_done = True
                    res.terminations[step_info.get("termination_reason", "none")] = (
                        res.terminations.get(step_info.get("termination_reason", "none"), 0) + 1
                    )
                    res.episode_returns.append(float(step_info.get("episode_return_train", 0.0)))
                    self.cb.on_episode_end(
                        game, step_info.get("episode_id", -1), term, trunc,
                        step_info.get("termination_reason", "none"),
                    )
                # learner update：按新交互步数计
                if steps_done % update_every == 0:
                    stats = self.cb.learner_update(self.mapping_version)
                    self.counters["learner_updates"] += 1
                    res.learner_updates += 1
                    self._record(event="learner_update", game=game, **{
                        k: v for k, v in stats.items() if isinstance(v, (int, float))
                    })
            if not episode_done and steps_done >= budget:
                # 预算到点：截断，不补采
                res.truncated_tail = True
                self.cb.on_episode_end(game, -1, False, True, "none")
                self._record(event="budget_truncated_episode", game=game, steps=steps_done)
        res.score_raw_mean = (
            float(np.mean(res.episode_returns)) if res.episode_returns else 0.0
        )
        return steps_done

    # ---- 整个 schedule ----
    def run(self) -> Dict[str, Any]:
        stages = list(self.cfg.stage_schedule)
        seen_games: set = set()
        results: List[StageResult] = []
        for i, stage in enumerate(stages):
            game = stage["game"]
            first_visit = game not in seen_games
            res = self.run_stage(game, int(stage["steps"]), first_visit=first_visit)
            seen_games.add(game)
            results.append(res)
            # 阶段结束：把该任务的在线数据固化进历史（§8.4）
            self._record(event="commit_stage", game=game, stage_index=i)
        self.subphase = SubPhase.DONE
        return {
            "stages": [r.as_dict() for r in results],
            "counters": dict(self.counters),
            "mapping_version": self.mapping_version,
            "total_online_steps": sum(r.online_steps for r in results),
            "total_first_data_steps": sum(r.first_data_steps for r in results),
            "adapt_events": self.counters["adapt_events"],
            "offline_updates": self.counters["offline_updates"],
        }
