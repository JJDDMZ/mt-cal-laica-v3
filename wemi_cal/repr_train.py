"""表示适配：φ/ψ 的联合训练、输出层扩展与 optimizer 状态继承（技术方案 §6.3/§6.4，任务 T4）。

适配阶段只更新 φ/ψ；β/Q/target 及其 optimizer 必须完全冻结，并先保存适配前快照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .envs import features as F
from .models.continuous import Phi, gaussian_kl_to_standard


# ---------------------------------------------------------------- optimizer 继承
def expand_decoder_optimizer(
    optimizer: torch.optim.Optimizer,
    phi: Phi,
    old_params: Sequence[nn.Parameter],
) -> None:
    """把 Adam 状态从旧的 φ 参数迁移到扩展后的新参数。

    - 旧行的 ``exp_avg`` / ``exp_avg_sq`` 逐行复制；
    - 新行置零；
    - ``step`` 等技术量保留，不重置整个优化器（§6.4 规则 2）。

    调用时机：``phi.register_actions(...)`` **之后**，且 ``old_params`` 是扩展前的
    ``[weight, bias]`` 对象（它们在 register_actions 里被替换掉了）。
    """
    old_weight, old_bias = old_params[0], old_params[1]
    new_weight, new_bias = phi.linear.weight, phi.linear.bias
    if not isinstance(old_weight, nn.Parameter) or not isinstance(old_bias, nn.Parameter):
        raise TypeError("old_params 必须是扩展前的 nn.Parameter（weight, bias）")
    if new_weight.shape[0] < old_weight.shape[0]:
        raise ValueError("扩展后的参数行数不应少于旧参数")

    for old_p, new_p in ((old_weight, new_weight), (old_bias, new_bias)):
        old_state = optimizer.state.pop(old_p, None)
        if old_state is None:
            # 没有历史矩：显式建立零矩，保证后续 step 的形状正确
            optimizer.state[new_p] = {
                "step": torch.tensor(0.0),
                "exp_avg": torch.zeros_like(new_p, memory_format=torch.preserve_format),
                "exp_avg_sq": torch.zeros_like(new_p, memory_format=torch.preserve_format),
            }
            continue
        n_old = old_p.shape[0]
        new_state: Dict[str, Any] = {}
        for key, val in old_state.items():
            if torch.is_tensor(val) and val.shape[:1] == (n_old,):
                pad = torch.zeros(
                    (new_p.shape[0] - n_old,) + tuple(val.shape[1:]),
                    dtype=val.dtype, device=val.device,
                )
                new_state[key] = torch.cat([val.clone(), pad], dim=0)
            elif torch.is_tensor(val):
                new_state[key] = val.clone()
            else:
                new_state[key] = val
        optimizer.state[new_p] = new_state

    # 确保 param_groups 指向新参数对象
    for group in optimizer.param_groups:
        group["params"] = [
            phi.linear.weight if p is old_weight else
            phi.linear.bias if p is old_bias else p
            for p in group["params"]
        ]


# ---------------------------------------------------------------- 表示损失
@dataclass
class ReprBatch:
    """表示适配的一个 minibatch。"""

    x: torch.Tensor            # [B,29]
    next_x: torch.Tensor       # [B,29]
    token: torch.Tensor        # [B] global token
    game: str

    def __len__(self) -> int:  # pragma: no cover - 便捷
        return int(self.x.shape[0])


@dataclass
class ReprDiagnostics:
    loss: float = 0.0
    nll: float = 0.0
    kl: float = 0.0
    accuracy: float = 0.0
    per_action_accuracy: Dict[int, float] = field(default_factory=dict)
    per_action_nll: Dict[int, float] = field(default_factory=dict)
    grad_norm: float = 0.0
    n_updates: int = 0


def repr_loss(
    model: Any,
    batch: ReprBatch,
    *,
    kl_weight: float = 1e-3,
    samples: int = 1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """§6.3：``L_repr = mean[-log φ_k(j|e_ψ,g) + λ_KL·KL(q_ψ(u|x,x')‖N(0,I))]``。

    每条 transition 采一个重参数化样本即可。ψ 不读 j/reward；它只通过动作重构目标
    看到 j。tanh 在 prior/posterior 两边共同使用，因此 KL 在 u 空间计算。
    """
    if isinstance(batch.token, list):
        token = torch.as_tensor(batch.token)
    else:
        token = batch.token
    s = model.psi.rsample(batch.x, batch.next_x, samples=samples)
    # samples>1 时取第一个样本（每条 transition 一个样本即可）
    e_psi = s.e[:, 0] if s.e.dim() == 3 else s.e

    logp_all = model.phi.finite_log_probs(e_psi, batch.game)   # [B,K]，非法项有限下界
    logp = logp_all[torch.arange(e_psi.shape[0]), token]
    if not torch.isfinite(logp).all():
        raise ValueError("重构项出现非有限值：token 或掩码有错（非法 token 会得到下界）")

    mean, log_std = model.psi.distribution_params(batch.x, batch.next_x)
    kl = gaussian_kl_to_standard(mean, log_std)

    nll = -logp.mean()
    kl_mean = kl.mean()
    loss = nll + kl_weight * kl_mean

    with torch.no_grad():
        pred = logp_all.argmax(dim=-1)
        acc = (pred == token).float().mean()
    stats = {
        "loss": loss,
        "nll": nll,
        "kl": kl_mean,
        "accuracy": acc,
        "logp_all": logp_all,
    }
    return loss, stats


def _per_action_stats(logp_all: torch.Tensor, token: torch.Tensor) -> Tuple[Dict[int, float], Dict[int, float]]:
    acc: Dict[int, float] = {}
    nll: Dict[int, float] = {}
    pred = logp_all.argmax(dim=-1)
    for tok in torch.unique(token).tolist():
        sel = token == tok
        acc[int(tok)] = float((pred[sel] == token[sel]).float().mean().item())
        nll[int(tok)] = float(-logp_all[sel][torch.arange(int(sel.sum())), token[sel]].mean().item())
    return acc, nll


# ---------------------------------------------------------------- 适配循环
def adapt_representation(
    model: Any,
    batches: Iterable[ReprBatch],
    *,
    updates: int,
    lr: float = 1e-3,
    kl_weight: float = 1e-3,
    grad_clip: float = 10.0,
    model_name: str = "continuous",
) -> ReprDiagnostics:
    """跑 ``updates`` 个 minibatch 的 φ/ψ 适配。

    约束（§6.4）：
    - β/Q/target 及其 optimizer 全程冻结；
    - 只更新 φ/ψ（或 D 的 ψD + 当前游戏码本行）；
    - 适配结束后调用方负责冻结 φ/ψ。
    """
    if updates <= 0:
        raise ValueError("updates 必须为正")
    phi_params = list(model.phi.parameters())
    psi_params = list(model.psi.parameters())
    optimizer = torch.optim.Adam(phi_params + psi_params, lr=lr)

    frozen = [p for name, p in model.named_parameters()
              if not (name.startswith("phi.") or name.startswith("psi."))]
    saved = [(p, p.requires_grad) for p in frozen]
    for p, _ in saved:
        p.requires_grad_(False)

    diag = ReprDiagnostics()
    data = list(batches)
    if not data:
        raise ValueError("适配数据为空")
    try:
        for i in range(updates):
            batch = data[i % len(data)]
            loss, stats = repr_loss(model, batch, kl_weight=kl_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(phi_params + psi_params, grad_clip)
            optimizer.step()
            diag.loss = float(stats["loss"].item())
            diag.nll = float(stats["nll"].item())
            diag.kl = float(stats["kl"].item())
            diag.accuracy = float(stats["accuracy"].item())
            diag.grad_norm = float(gn)
            diag.n_updates = i + 1
            with torch.no_grad():
                acc, nll = _per_action_stats(stats["logp_all"], batch.token)
            diag.per_action_accuracy = acc
            diag.per_action_nll = nll
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)
    return diag


def split_by_episode(
    episode_ids: Sequence[int], *, val_fraction: float = 0.2, seed: int = 0
) -> Tuple[List[int], List[int]]:
    """按 episode 划分 80/20（§6.4：全部属于训练数据，不是保留测试集）。

    返回 (train_indices, diag_indices)。划分以 episode 为单位，避免同一回合被拆开。
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction 必须在 (0,1)")
    eps = sorted(set(int(e) for e in episode_ids))
    if len(eps) < 2:
        return list(range(len(episode_ids))), []
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(len(eps), generator=g).tolist()
    n_val = max(1, int(round(len(eps) * val_fraction)))
    val_eps = {eps[i] for i in perm[:n_val]}
    train_idx, val_idx = [], []
    for i, e in enumerate(episode_ids):
        (val_idx if int(e) in val_eps else train_idx).append(i)
    if not train_idx:  # pragma: no cover - 极端划分保护
        raise ValueError("划分后训练集为空")
    return train_idx, val_idx


def make_batches(
    *,
    x: torch.Tensor,
    next_x: torch.Tensor,
    token: torch.Tensor,
    game: str,
    episode_ids: Optional[Sequence[int]] = None,
    batch_size: int = 128,
    seed: int = 0,
) -> List[ReprBatch]:
    """把适配数据切成固定 batch。episode_ids 给定时会按 episode 顺序分组。"""
    n = x.shape[0]
    if n == 0:
        raise ValueError("适配数据为空")
    g = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(n, generator=g).tolist()
    batches: List[ReprBatch] = []
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        if len(idx) < 2:
            break
        t = torch.as_tensor(idx)
        batches.append(ReprBatch(x=x[t], next_x=next_x[t], token=token[t], game=game))
    if not batches:
        raise ValueError("batch 划分后没有可用数据")
    return batches
