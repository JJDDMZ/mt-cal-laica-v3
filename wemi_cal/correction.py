"""φ/b 单步 TD 修正（技术方案 §8.2，任务 T5）。

概率映射下同一 ``(x,e)`` 对应的真实转移是各动作转移按 φ 的混合：

    P_k(x'|x,e) = Σ_j φ_k(j|e,g) P_g(x'|x,j)

旧数据由 b_old 选动作，代表的是另一种混合。因此当前映射版本 k 下用

    w = sg[ exp{ log φ_k(j_executed|e_behavior,g) - log b_old(j_executed|e_behavior,g) } ]

校正**单步条件转移**。依据是

    Σ_j b_old(j|e,g) · [φ_k(j|e,g)/b_old(j|e,g)] · P_g(x'|x,j) = P_k(x'|x,e)

成立条件（必须写进日志与报告）：游戏条件转移未变、合法动作有行为支持、概率记录正确。
这**不是**消除所有离策略误差的定理，也不修正游戏版本变化、状态混叠、整条轨迹回报。

**当前版本的新数据也要修正**（行为是 b=0.98φ+0.01，critic 的目标映射是 φ）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .envs import features as F

# 行为探索下限使 b ≥ 0.01/|A|（§8.2）：|A|=2 时 b ≥ 0.005，故 w ≤ φ/b ≤ 200。
# 方案里写的"理论上≤100"来自 b ≥ 0.01 的粗估；这里按实际混合常数算出精确上界，
# 并在越界时报错而不是悄悄裁剪。
UNIFORM_MIX = 0.02
MAX_WEIGHT = 1.0 / (UNIFORM_MIX / 2)      # = 100.0，b 的最小值 0.01（=0.02/2）


class CorrectionError(ValueError):
    """概率记录或掩码有错：必须中止排查，不能用裁剪掩盖。"""


@dataclass
class CorrectionReport:
    weights: torch.Tensor            # [B] detached
    mean: float
    max: float
    min: float
    ess: float                       # (Σw)² / Σw²
    ess_over_batch: float
    n_above_max: int
    task: str


def mapping_correction(
    *,
    phi: Any,
    x: torch.Tensor,
    e_behavior: torch.Tensor,
    token: torch.Tensor,
    log_b_action: torch.Tensor,
    game: str,
    check_bounds: bool = True,
) -> torch.Tensor:
    """返回 detached 的 ``[B]`` 校正权重 w。

    输入都是**执行时记录的事实**：``e_behavior`` 与 ``log_b_action`` 不得按当前 φ 重写。
    """
    if e_behavior.dim() != 2:
        raise CorrectionError(f"e_behavior 必须是 [B,2]，得到 {tuple(e_behavior.shape)}")
    if e_behavior.shape[1] != 2:
        raise CorrectionError(
            f"e_behavior 第二维必须是 latent_dim=2，得到 {e_behavior.shape[1]}"
        )
    if token.shape != (e_behavior.shape[0],):
        raise CorrectionError(f"token 形状应为 [B]，得到 {tuple(token.shape)}")
    if log_b_action.shape != token.shape:
        raise CorrectionError(f"log_b_action 形状应为 [B]，得到 {tuple(log_b_action.shape)}")

    with torch.no_grad():
        log_phi_all = phi.log_probs(e_behavior, game)            # [B, total_tokens]
        rows = torch.arange(token.shape[0], device=token.device)
        cols = token.long()
        if cols.max().item() >= log_phi_all.shape[1] or cols.min().item() < 0:
            raise CorrectionError(f"token 越界: {cols.tolist()}")
        log_phi = log_phi_all[rows, cols]

        # 非法 token（未注册/不属于该游戏）→ -inf → 比值无意义
        if torch.isinf(log_phi).any() or torch.isnan(log_phi).any():
            bad = torch.nonzero(torch.isinf(log_phi) | torch.isnan(log_phi)).flatten().tolist()
            raise CorrectionError(f"token 在当前 φ 下非法（行 {bad}）：概率记录或掩码有错")

        if torch.isnan(log_b_action).any() or torch.isinf(log_b_action).any():
            raise CorrectionError("log_b_action 含 NaN/Inf：行为概率记录有错")

        log_w = log_phi - log_b_action
        # 用 float64 做差再转回学习 dtype，避免大数相减的精度损失
        if log_w.dtype != torch.float64:
            log_w = log_w.double()
        if not torch.isfinite(log_w).all():
            raise CorrectionError("log 差出现非有限值：概率记录有错")
        w = torch.exp(log_w).to(e_behavior.dtype)

        if check_bounds:
            if not torch.isfinite(w).all():
                raise CorrectionError("校正权重出现非有限值")
            if (w <= 0).any():
                raise CorrectionError("校正权重必须为正")
            wmax = float(w.max())
            if wmax > MAX_WEIGHT * (1 + 1e-6):
                raise CorrectionError(
                    f"校正权重上界被突破: max={wmax:.6g} > {MAX_WEIGHT:.6g}；"
                    "说明行为概率或掩码记录有误，应中止排查而不是裁剪（§8.2）"
                )
    return w


def correction_report(w: torch.Tensor, game: str) -> CorrectionReport:
    with torch.no_grad():
        wd = w.detach().double()
        s = float(wd.sum())
        ss = float((wd ** 2).sum())
        ess = (s * s / ss) if ss > 0 else 0.0
        b = int(wd.numel())
        return CorrectionReport(
            weights=w.detach(),
            mean=float(wd.mean()),
            max=float(wd.max()),
            min=float(wd.min()),
            ess=ess,
            ess_over_batch=(ess / b if b else 0.0),
            n_above_max=int((wd > MAX_WEIGHT * (1 + 1e-6)).sum().item()),
            task=game,
        )


def enumerate_correction_check() -> Dict[str, float]:
    """§8.5 的枚举单元测试，作为可执行的自检函数（不走随机采样）。

    固定同一个 ``(x,e)``：

    - 旧行为分布 b_old = [0.9, 0.1]
    - 当前 φ 概率 = [0.2, 0.8]
    - 两个动作的奖励 = [1, 3]

    期望：
    - 不修正的旧分布期望奖励 = 0.9*1 + 0.1*3 = 1.2
    - 用 w = φ/b 加权后的期望       = 0.9*(0.2/0.9)*1 + 0.1*(0.8/0.1)*3 = 0.2 + 2.4 = 2.6
    - 直接按新映射算的期望           = 0.2*1 + 0.8*3 = 2.6
    """
    b_old = [0.9, 0.1]
    phi = [0.2, 0.8]
    rewards = [1.0, 3.0]

    naive = sum(b_old[j] * rewards[j] for j in range(2))
    corrected = sum(
        b_old[j] * (phi[j] / b_old[j]) * rewards[j] for j in range(2)
    )
    direct = sum(phi[j] * rewards[j] for j in range(2))
    return {
        "naive_expectation": naive,
        "corrected_expectation": corrected,
        "direct_new_mapping_expectation": direct,
        "weights": [phi[j] / b_old[j] for j in range(2)],
    }


def uniform_behavior_weight_bound(n_legal: int = 2, uniform_mix: float = UNIFORM_MIX) -> float:
    """均匀热身数据（b=1/|A|）下的权重上界，用于断言 "≤ 1/(ε/|A|)"。"""
    b_min = uniform_mix / n_legal
    return 1.0 / b_min
