"""纯评测（技术方案 §12.1/§12.2，任务 T8）。

两种明确区分的推理模式：

- **greedy，主部署指标**：C 用 ``e=tanh(μβ(x))``，再取合法集合内 ``argmax φ(j|e,g)``；
  D 取 ``argmax π(j|x)``。并列取最小 token。**没有** 2% 探索。
- **stochastic_target，机制诊断**：C 从 β 采 e、从纯 φ 采 j；D 从 π 采 j；
  仍不加入行为探索。策略 RNG 使用 ``300000+episode_index``。

硬约束：评测不写 replay、不更新任何权重、不改变参数 hash（§12.5/§14 T8）。
测试/验证数据永不进入任何 buffer（§12.1）。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import Config
from .envs import features as F
from .envs.features import PolicyMode

POLICY_RNG_BASE = 300000
VALIDATION_SEED_START = 100000
TEST_SEED_START = 200000


class EvaluateError(RuntimeError):
    """非法评测请求。"""


@dataclass
class EpisodeResult:
    game: str
    episode_index: int
    reset_seed: int
    policy_mode: str
    score_raw: float
    return_train: float
    length: int
    termination_reason: str
    truncated: bool

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class GameReport:
    game: str
    policy_mode: str
    episodes: int = 0
    score_mean: float = 0.0
    score_median: float = 0.0
    score_std: float = 0.0
    score_max: float = 0.0
    pass_count: int = 0
    pass_rate: float = 0.0
    return_mean: float = 0.0
    length_mean: float = 0.0
    terminations: Dict[str, int] = field(default_factory=dict)
    truncated_count: int = 0
    # 每个 global token 被 greedy 选中多少次（诊断动作坍缩）
    token_counts: Dict[int, int] = field(default_factory=dict)
    token_share: Dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------- 动作选择
def select_token(
    *,
    variant: str,
    model: Any,
    x: np.ndarray,
    game: str,
    policy_mode: str,
    rng: Optional[torch.Generator] = None,
) -> int:
    """按推理模式给出 global token。绝不使用行为探索混合（§12.1）。"""
    if policy_mode not in (PolicyMode.GREEDY.value, PolicyMode.STOCHASTIC_TARGET.value):
        raise EvaluateError(f"未知 policy_mode: {policy_mode!r}（§12.1 只有两种）")
    xb = torch.as_tensor(np.asarray(x, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        if variant == "continuous":
            if policy_mode == PolicyMode.GREEDY.value:
                e = model.beta.deterministic_e(xb)          # tanh(μβ(x))
                tok = model.phi.greedy_token(e, game)
            else:
                s = model.beta.rsample(xb, samples=1)
                e = s.e_squeezed
                tok = model.phi.sample_token(e, game, uniform_mix=0.0)   # 纯 φ
        elif variant == "discrete":
            if policy_mode == PolicyMode.GREEDY.value:
                tok = model.actor.greedy_token(xb, model.codebook, game)
            else:
                tok = model.actor.sample_token(xb, model.codebook, game)
        else:
            raise EvaluateError(f"未知 variant: {variant!r}")
    return int(tok.item())


# ---------------------------------------------------------------- 评测循环
def _info_get(info: Any, key: str, default: Any = None) -> Any:
    """step 的 info 既可能是 dict，也可能是 EnvStepInfo 之类的对象；两种都支持。"""
    if info is None:
        return default
    if isinstance(info, dict):
        return info.get(key, default)
    return getattr(info, key, default)


def evaluate_game(
    *,
    cfg: Config,
    model: Any,
    game: str,
    env_reset: Any,
    env_step: Any,
    policy_mode: str,
    episodes: int,
    seed_start: int,
    device: str = "cpu",
    verbose: bool = False,
) -> GameReport:
    """跑 ``episodes`` 个回合并汇总。调用方负责保证 env 不把评测数据写进 replay。

    reset seed 与策略 RNG seed **分开**：前者用 ``seed_start..``，后者用
    ``300000+episode_index``（§12.1）。

    ``verbose=True`` 时每回合打印一行：真机评测一跑就是几十分钟，没有进度输出时
    操作者完全无法区分"在跑"和"卡住了"（这个坑真实踩过）。
    """
    if episodes <= 0:
        raise EvaluateError("episodes 必须为正")
    results: List[EpisodeResult] = []
    report = GameReport(game=game, policy_mode=policy_mode)
    gen = torch.Generator().manual_seed(POLICY_RNG_BASE)
    # 动作直方图：诊断"策略从不选某个动作"这类坍缩最直接的证据
    token_counts: Dict[int, int] = {}
    pass_score = (cfg.eval["flappy_pass_score"] if game == "Flappybird"
                  else cfg.eval["timber_pass_score"])

    for i in range(episodes):
        reset_seed = seed_start + i
        gen.manual_seed(POLICY_RNG_BASE + i)
        torch.manual_seed(POLICY_RNG_BASE + i)
        np.random.seed(POLICY_RNG_BASE + i)
        x, info = env_reset(game, reset_seed)
        ep_return = 0.0
        length = 0
        reason = "none"
        truncated = False
        while True:
            tok = select_token(variant=cfg.variant, model=model, x=x, game=game,
                               policy_mode=policy_mode, rng=gen)
            token_counts[int(tok)] = token_counts.get(int(tok), 0) + 1
            local = F.local_action(game, tok)
            x, r, term, trunc, step_info = env_step(game, local)
            ep_return += float(r)
            length += 1
            reason = _info_get(step_info, "termination_reason", "none")
            truncated = bool(trunc)
            if term or trunc:
                break
            if length >= cfg.horizon * 2:
                raise EvaluateError("评测回合超过 2H 仍未结束：终止语义有错")
        results.append(EpisodeResult(
            game=game, episode_index=i, reset_seed=reset_seed, policy_mode=policy_mode,
            score_raw=float(_info_get(step_info, "score_raw", 0.0)),
            return_train=ep_return, length=length, termination_reason=reason,
            truncated=truncated,
        ))
        if verbose:
            r0 = results[-1]
            running = statistics.fmean([q.score_raw for q in results])
            print(f"[eval] {game} 第 {i + 1}/{episodes} 回合: 分={r0.score_raw:.1f} "
                  f"步数={r0.length} 终止={reason} | 累计均值={running:.2f}",
                  flush=True)

    scores = [r.score_raw for r in results]
    report.episodes = len(results)
    report.score_mean = float(statistics.fmean(scores))
    report.score_median = float(statistics.median(scores))
    report.score_std = float(statistics.pstdev(scores)) if len(scores) > 1 else 0.0
    report.score_max = float(max(scores))
    report.pass_count = sum(1 for s in scores if s >= pass_score)
    report.pass_rate = report.pass_count / len(scores)
    report.return_mean = float(statistics.fmean([r.return_train for r in results]))
    report.length_mean = float(statistics.fmean([r.length for r in results]))
    for r in results:
        report.terminations[r.termination_reason] = report.terminations.get(r.termination_reason, 0) + 1
    report.truncated_count = sum(1 for r in results if r.truncated)
    report.token_counts = dict(sorted(token_counts.items()))
    report.token_share = {int(k): v / max(1, sum(token_counts.values()))
                          for k, v in report.token_counts.items()}
    return report


# ---------------------------------------------------------------- 达标判定
@dataclass
class AcceptanceResult:
    game: str
    passed_absolute: bool
    passed_relative: bool
    score_mean: float
    pass_rate: float
    required_pass_count: int
    relative_evidence_available: bool
    st_baseline_mean: Optional[float] = None

    @property
    def passed(self) -> bool:
        return self.passed_absolute and self.passed_relative

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["passed"] = self.passed
        return d


def acceptance(
    *,
    cfg: Config,
    report: GameReport,
    st_baseline_mean: Optional[float],
) -> AcceptanceResult:
    """§12.2 的绝对门槛 + 相对门槛（N_g = J_MT,g / J_ST,g ≥ 0.8）。"""
    required = int(round(cfg.eval["required_pass_rate"] * report.episodes))
    passed_abs = report.pass_count >= required
    if st_baseline_mean is None:
        passed_rel = False
        note = None
    elif st_baseline_mean <= 0:
        # §12.2：单游戏基线均值 ≤0 时比例无效，应先排查基线
        passed_rel = False
        note = st_baseline_mean
    else:
        passed_rel = (report.score_mean / st_baseline_mean) >= 0.8
        note = st_baseline_mean
    return AcceptanceResult(
        game=report.game,
        passed_absolute=passed_abs,
        passed_relative=passed_rel,
        score_mean=report.score_mean,
        pass_rate=report.pass_rate,
        required_pass_count=required,
        relative_evidence_available=st_baseline_mean is not None,
        st_baseline_mean=note,
    )
