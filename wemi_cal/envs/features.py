"""特征、动作注册表、奖励与终止语义（技术方案 §5，任务 T3）。

三件容易混淆的东西在这里被显式分开：
- ``feature_mask``：区分“真实的 0”和 pad 填充值；
- ``action_mask``：限制当前游戏的合法动作；
- ``global token``：4 个动作身份，由注册表翻译成本地动作编号（不是 φ）。

不启用动态 running normalization：固定尺度避免历史数据/回报/checkpoint 之间悄悄改变含义。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# ---- §5.2/§5.3 维度 ----
D_LOCAL = 13          # 游戏局部特征最大长度
OBS_DIM = 29          # 13 + 13 + 2 + 1
FEATURE_SLICE = slice(0, 13)
MASK_SLICE = slice(13, 26)
TASK_SLICE = slice(26, 28)
BUDGET_INDEX = 28

GAMES: Tuple[str, ...] = ("Flappybird", "Timberman")
TASK_ID: Dict[str, int] = {"Flappybird": 0, "Timberman": 1}
N_LOCAL_FEATURES: Dict[str, int] = {"Flappybird": 4, "Timberman": 13}

OBS_SCHEMA = "compact29-v1"
REWARD_SCHEMA = "bounded-v1"

# 主 schema 的命名字段（§5.2）。旧 36 维 pad 包装保留为对照，不作主 schema。
FLAPPY_FIELDS = ("bird_y", "gap_top", "gap_bottom", "pipe_x")
TIMBER_FIELDS = ("tree_sides", "npc_side", "blood")

# 编码常量（§5.2）
SIDE_LEFT = -1.0
SIDE_CENTER = 0.0
SIDE_RIGHT = 1.0
BLOOD_MAX = 6000.0

WORLD_WIDTH = 400.0
WORLD_HEIGHT = 600.0

class PolicyMode(str, Enum):
    """§12.1 两种明确区分的推理模式。"""

    GREEDY = "greedy"                    # 主部署指标：C 用 e=tanh(μβ)，再 argmax φ
    STOCHASTIC_TARGET = "stochastic_target"   # 机制诊断：从 β 采 e、从纯 φ 采 j


# ---- §5.4 四个动作身份 ----
# global token → (game, local action)
ACTION_REGISTRY: Tuple[Tuple[str, int], ...] = (
    ("Flappybird", 0),
    ("Flappybird", 1),
    ("Timberman", 0),
    ("Timberman", 1),
)
TOKEN_TO_LOCAL: Dict[str, Dict[int, int]] = {
    "Flappybird": {0: 0, 1: 1},
    "Timberman": {2: 0, 3: 1},
}
ACTION_MASK: Dict[str, np.ndarray] = {
    "Flappybird": np.array([1, 1, 0, 0], dtype=np.float32),
    "Timberman": np.array([0, 0, 1, 1], dtype=np.float32),
}
LEGAL_TOKENS: Dict[str, Tuple[int, ...]] = {
    "Flappybird": (0, 1),
    "Timberman": (2, 3),
}


class FeatureError(ValueError):
    """命名字段非法：缺字段、未知枚举、维度错误。"""


def task_id(game: str) -> int:
    if game not in TASK_ID:
        raise FeatureError(f"未知游戏: {game!r}，允许 {GAMES}")
    return TASK_ID[game]


def legal_tokens(game: str) -> Tuple[int, ...]:
    if game not in LEGAL_TOKENS:
        raise FeatureError(f"未知游戏: {game!r}")
    return LEGAL_TOKENS[game]


def local_action(game: str, token: int) -> int:
    """global token → 本地动作编号；非法 token 直接报错（不 clamp 到最后一个）。"""
    if game not in TOKEN_TO_LOCAL:
        raise FeatureError(f"未知游戏: {game!r}")
    mapping = TOKEN_TO_LOCAL[game]
    if token not in mapping:
        raise FeatureError(
            f"非法 token {token} for {game}；合法集合 {sorted(mapping)}（不 clamp）"
        )
    return mapping[token]


def token_for_local(game: str, action: int) -> int:
    for tok, local in TOKEN_TO_LOCAL.get(game, {}).items():
        if local == action:
            return tok
    raise FeatureError(f"{game} 没有本地动作 {action}")


def action_mask(game: str) -> np.ndarray:
    """返回副本，避免调用方改到全局常量。"""
    if game not in ACTION_MASK:
        raise FeatureError(f"未知游戏: {game!r}")
    return ACTION_MASK[game].copy()


# ---------------------------------------------------------------- 特征编码
def encode_flappy(obs: Mapping[str, Any]) -> List[float]:
    """§5.2 F 四维：bird_y / gap_top / gap_bottom / pipe_x，按 world 尺度归一。

    终止时坐标可略超边界，不裁剪到 [0,1]。
    """
    missing = [k for k in FLAPPY_FIELDS if k not in obs]
    if missing:
        raise FeatureError(f"Flappybird obs 缺少字段: {missing}")
    vals = []
    for k in FLAPPY_FIELDS:
        v = obs[k]
        if v is None:
            raise FeatureError(f"Flappybird obs.{k} 是 null")
        try:
            f = float(v)
        except (TypeError, ValueError) as exc:
            raise FeatureError(f"Flappybird obs.{k} 不是数值: {v!r}") from exc
        if f != f:
            raise FeatureError(f"Flappybird obs.{k} 是 NaN")
        vals.append(f)
    # 观测已由 JS 归一；这里不再二次缩放，避免尺度悄悄变化
    return vals


def encode_timberman(obs: Mapping[str, Any]) -> List[float]:
    """§5.2 T 十三维：[11 棵树方位, npc_side, blood/6000]。

    JS 侧已按 false→-1 / "center"→0 / true→+1 编码并拒绝未知字符串；
    这里只做形状与范围校验，不从字符串反推。
    """
    if "features" not in obs:
        raise FeatureError("Timberman obs 缺少 features")
    feats = obs["features"]
    if isinstance(feats, (str, bytes)):
        raise FeatureError("Timberman obs.features 不能是字符串（bool/center 混合列表）")
    out: List[float] = []
    for i, v in enumerate(feats):
        if v is None:
            raise FeatureError(f"Timberman features[{i}] 是 null")
        try:
            f = float(v)
        except (TypeError, ValueError) as exc:
            raise FeatureError(f"Timberman features[{i}] 不是数值: {v!r}") from exc
        if f != f:
            raise FeatureError(f"Timberman features[{i}] 是 NaN")
        out.append(f)
    if len(out) != D_LOCAL:
        raise FeatureError(
            f"Timberman features 长度必须是 {D_LOCAL}（11 树 + npc + blood），得到 {len(out)}"
        )
    sides = out[:11]
    for i, s in enumerate(sides):
        if s not in (SIDE_LEFT, SIDE_CENTER, SIDE_RIGHT):
            raise FeatureError(
                f"Timberman 树方位[{i}] 必须是 -1/0/+1，得到 {s!r}（拒绝未知枚举）"
            )
    if out[11] not in (SIDE_LEFT, SIDE_RIGHT):
        raise FeatureError(f"Timberman npc_side 必须是 -1/+1，得到 {out[11]!r}")
    if not (0.0 <= out[12] <= 1.0):
        # blood 允许短暂为负（源码 blood-16 后判 <0.017），但不允许超出比例上界
        if out[12] > 1.0:
            raise FeatureError(f"Timberman blood 比例越界: {out[12]!r}")
    return out


def local_features(game: str, obs: Mapping[str, Any]) -> List[float]:
    if game == "Flappybird":
        return encode_flappy(obs)
    if game == "Timberman":
        return encode_timberman(obs)
    raise FeatureError(f"未知游戏: {game!r}")


def encode_observation(
    game: str,
    obs: Mapping[str, Any],
    *,
    steps_taken: int,
    horizon: int,
    dtype: Any = np.float32,
) -> np.ndarray:
    """把命名字段观测编码成 §5.3 的 29 维向量。

    ``[0:13]`` 归一化局部特征（不足补零）；``[13:26]`` 有效特征掩码；
    ``[26:28]`` 任务 one-hot；``[28]`` 剩余回合预算 (H-t)/H。

    ``steps_taken`` 是"已经执行过的 agent step 数"：reset 后的初始观测为 0
    （方案 §5.3 写明 reset=1，即剩余预算恰为 1），随后每次 step 加一。
    因此 H=1000 表示一回合有 1000 个可用动作，最后一步把预算推到 0（horizon 终止）。
    """
    if steps_taken < 0:
        raise FeatureError("steps_taken 不能为负")
    if steps_taken > horizon:
        raise FeatureError(f"steps_taken 不能超过 horizon: {steps_taken} > {horizon}")

    feats = local_features(game, obs)
    n = N_LOCAL_FEATURES[game]
    if len(feats) != n:
        raise FeatureError(f"{game} 局部特征长度必须是 {n}，得到 {len(feats)}")

    x = np.zeros(OBS_DIM, dtype=np.float64)
    x[FEATURE_SLICE][: len(feats)] = feats
    x[MASK_SLICE][:n] = 1.0
    x[26 + task_id(game)] = 1.0
    x[BUDGET_INDEX] = (horizon - steps_taken) / horizon
    return x.astype(dtype)


# ---------------------------------------------------------------- 奖励
def flappy_reward(*, died: bool, bird_y: float, gap_top: float, gap_bottom: float) -> float:
    """§5.5：死亡 -1；否则 clip(0.01*min(y-gap_top, gap_bottom-y), -1, 1)，边界相等取 0。

    ``bird_y/gap_top/gap_bottom`` 使用与观测相同的世界尺度（已归一化也无妨，比例一致，
    但这里按 world 坐标传入以匹配方案公式）。
    """
    if died:
        return -1.0
    d = min(bird_y - gap_top, gap_bottom - bird_y)
    return float(np.clip(0.01 * d, -1.0, 1.0))


def timberman_reward(*, died: bool, score_t: float, score_next: float) -> float:
    """§5.5：clip(score_{t+1} - score_t - 1[本步死亡], -1, 1)。

    原始累计分数不能逐步求和当奖励；这里的增量才是训练奖励。
    """
    delta = float(score_next) - float(score_t) - (1.0 if died else 0.0)
    return float(np.clip(delta, -1.0, 1.0))


def train_reward(
    game: str,
    response: Any,
    *,
    score_before: float,
    died: bool,
) -> float:
    """统一入口：由 obs/score/termination_reason 在 Python 单一位置算训练奖励（§4.2）。

    horizon 不算死亡：达到 H 的最后一步仍用正常奖励。
    """
    if game == "Flappybird":
        obs = response.obs
        try:
            by = float(obs["bird_y"])
            gt = float(obs["gap_top"])
            gb = float(obs["gap_bottom"])
        except KeyError as exc:
            raise FeatureError(f"Flappybird 奖励需要 bird_y/gap_top/gap_bottom: {exc}") from exc
        # 观测是归一化值；公式要求同一世界尺度下的距离，缩放 1/600 一致，乘回去以匹配方案数值
        return flappy_reward(
            died=died,
            bird_y=by * WORLD_HEIGHT,
            gap_top=gt * WORLD_HEIGHT,
            gap_bottom=gb * WORLD_HEIGHT,
        )
    if game == "Timberman":
        return timberman_reward(
            died=died, score_t=score_before, score_next=float(response.score)
        )
    raise FeatureError(f"未知游戏: {game!r}")


# ---------------------------------------------------------------- 终止语义
@dataclass(frozen=True)
class Termination:
    """把游戏死亡、任务定义的 horizon、外部操作性截断三者分开（§4.5）。"""

    terminated: bool
    truncated: bool
    reason: str

    @property
    def died(self) -> bool:
        """游戏自身死亡（不含 horizon、不含外部截断）→ 训练奖励里的 -1。"""
        return self.terminated and self.reason in ("collision", "ground", "blood")


def classify_termination(
    *,
    response: Any,
    steps_taken: int,
    horizon: int,
    externally_truncated: bool = False,
) -> Termination:
    """判定本步的终止语义。

    - 游戏报 terminated 且 reason 是死亡类 → terminated=True, reason 照抄；
    - 达到 H → terminated=True, reason='horizon'（不算死亡、不加死亡罚分）；
    - 阶段预算耗尽等外部中止 → truncated=True（保留 bootstrap，且不产生 MC 标签）。
    """
    if steps_taken > horizon:
        raise FeatureError(f"steps_taken 超过 horizon: {steps_taken} > {horizon}")
    if externally_truncated:
        return Termination(terminated=False, truncated=True, reason="none")

    if response.terminated:
        reason = response.termination_reason
        if reason == "horizon":
            return Termination(terminated=True, truncated=False, reason="horizon")
        if reason not in ("collision", "ground", "blood"):
            raise FeatureError(f"未知终止原因: {reason!r}")
        return Termination(terminated=True, truncated=False, reason=reason)

    if steps_taken >= horizon:
        # 有限时域到达：终止但不是死亡（§4.5）
        return Termination(terminated=True, truncated=False, reason="horizon")

    return Termination(terminated=False, truncated=False, reason="none")


def bootstrap_allowed(term: Termination) -> bool:
    """§4.5：真正 terminal（含 horizon）不 bootstrap；外部截断保留 bootstrap。"""
    return term.truncated and not term.terminated


def reference_label_valid(term: Termination) -> bool:
    """§7.2：完整结束（自然死亡或到 H）才有有效 MC 标签；操作性截断没有。"""
    return term.terminated and not term.truncated


# ---------------------------------------------------------------- 旧包装（对照）
def legacy_pad_observation(obs: Sequence[float], game: str, *, dim: int = 34) -> np.ndarray:
    """旧的 ``pad(原观测,34)+task_onehot(2)`` 36 维包装，仅供对照（§5.1）。

    故意不做归一化：它保留原始数值尺度，用来演示与 compact29 的差异。
    """
    flat = np.asarray(list(obs), dtype=np.float32).ravel()
    if flat.size > dim:
        raise FeatureError(f"原观测长度 {flat.size} 超过 pad 维度 {dim}")
    out = np.zeros(dim + 2, dtype=np.float32)
    out[: flat.size] = flat
    out[dim + task_id(game)] = 1.0
    return out
