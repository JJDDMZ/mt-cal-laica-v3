"""C 主方案的学习目标：TD、校准正则、连续 actor 与梯度边界（技术方案 §7，任务 T6）。

价值定义（§7.1）：Qᵏ(x,e) 表示**先按当前映射 φk 抽出真实动作、随后由当前 β/φ 控制**
得到的环境奖励折扣回报。概率映射下它是各真实动作价值的混合，不能把单个实际动作的 Q
与 Q(x,e) 无条件写成逐点相等。

Bellman target（无熵项）：

    y = sg[ r + γ(1-terminated) · (1/N) Σ_n min_i Q̄_i(x', e'_n) ]

以 §8 的映射校正权重 w 做加权回归：

    L_TD,i = ½ mean[ w · (Q_i(x, e_behavior) - y)² ]

校准（§7.3，只在 ref_valid=true 行）：

    R_i^cal = mean[ (1/M) Σ_m max{Q_i(x, sg(e^m)), sg(G_t)} - Q_i(x, e_behavior) ]
    L_Qi = L_TD,i + λcal · R_i^cal

actor（§7.4）：

    L_β = E[ α·log β_θ(e|x) - min_i Q_i(x,e) ]，α=0.02

**梯度边界（最容易写错的地方）**：
- 校准/目标里的 e 采样与 G 都要 stop-gradient，Q 保留梯度；
- actor 更新时冻结 Q 的**参数**（``freeze_parameters``，不是 ``no_grad``），
  但 Q 对 e 的导数必须保留，否则 "loss 会变但 actor 不会学"；
- 形状：[B,N,latent] → Q/logp [B,N]；N=1 时显式 squeeze，禁止广播出 [B,B]。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .correction import CorrectionReport, correction_report, mapping_correction
from .models.continuous import freeze_parameters, polyak_update


@dataclass
class LearnerConfig:
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_alpha: float = 0.02
    lambda_cal: float = 0.1
    tau: float = 0.005
    grad_clip: float = 10.0
    target_samples: int = 2
    calibration_samples: int = 10
    calibration_mode: str = "max"        # max / linear / off
    backup_entropy: bool = False

    def __post_init__(self) -> None:
        if self.backup_entropy:
            raise ValueError(
                "本版 critic 不含熵 backup（§3）；改用 soft backup 必须另立实验版本并重定义 reference"
            )
        if self.calibration_mode not in ("max", "linear", "off"):
            raise ValueError(f"未知 calibration_mode: {self.calibration_mode!r}")
        if self.target_samples < 1 or self.calibration_samples < 1:
            raise ValueError("采样数必须 >= 1")


@dataclass
class UpdateStats:
    td_loss: float = 0.0
    cal_regularizer: float = 0.0
    q1_loss: float = 0.0
    q2_loss: float = 0.0
    actor_loss: float = 0.0
    actor_entropy: float = 0.0
    grad_norm_q1: float = 0.0
    grad_norm_q2: float = 0.0
    grad_norm_actor: float = 0.0
    w_mean: float = 0.0
    w_max: float = 0.0
    ess_over_batch: float = 0.0
    reference_coverage: Dict[int, float] = field(default_factory=dict)
    q_minus_reference_frac: Dict[int, float] = field(default_factory=dict)
    target_mean: float = 0.0
    q_data_mean: float = 0.0
    n_transitions: int = 0

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        return d


def validate_update_stats(
    stats: Any, *, expected_batch_size: int, variant: str,
) -> None:
    """Fail closed on invalid learner diagnostics before training can continue."""
    common_fields = (
        "td_loss", "cal_regularizer", "q1_loss", "q2_loss", "actor_loss",
        "grad_norm_q1", "grad_norm_q2", "grad_norm_actor", "target_mean",
        "q_data_mean",
    )
    if variant == "continuous":
        numeric_fields = common_fields + ("w_mean", "w_max", "ess_over_batch")
    elif variant == "discrete":
        numeric_fields = common_fields
    else:
        raise ValueError(f"未知 learner variant: {variant!r}")

    for name in numeric_fields:
        if not hasattr(stats, name):
            raise RuntimeError(f"更新统计缺少必需字段: {name}")
        value = getattr(stats, name)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"更新统计字段非法: {name}={value!r}（必须是有限数值）") from exc
        if not math.isfinite(numeric_value):
            raise RuntimeError(f"更新统计字段非法: {name}={value!r}（必须是有限数值）")

    if not hasattr(stats, "n_transitions"):
        raise RuntimeError("更新统计缺少必需字段: n_transitions")
    if stats.n_transitions != expected_batch_size:
        raise RuntimeError(
            f"更新 batch size 不匹配: stats.n_transitions={stats.n_transitions}, "
            f"expected={expected_batch_size}"
        )
    if variant == "continuous" and (float(stats.w_mean) <= 0 or float(stats.w_max) <= 0):
        raise RuntimeError(
            f"importance weights 必须为正: w_mean={stats.w_mean}, w_max={stats.w_max}"
        )
    if variant == "continuous" and float(stats.ess_over_batch) <= 0:
        raise RuntimeError(f"ESS 必须为正: ess_over_batch={stats.ess_over_batch}")


# ---------------------------------------------------------------- Bellman target
@torch.no_grad()
def compute_target(
    model: Any,
    batch: Dict[str, torch.Tensor],
    *,
    gamma: float,
    target_samples: int,
) -> torch.Tensor:
    """y = sg[r + γ(1-terminated) · mean_n min_i Q̄_i(x', e'_n)]；无熵项。

    ``terminated`` 为真（含任务定义的 horizon）时不做 bootstrap；
    外部截断 truncated 保留 bootstrap（§4.5/§7.1）。
    """
    x_next = batch["next_x"]
    b = x_next.shape[0]
    s = model.beta.rsample(x_next, samples=target_samples)
    e_next = s.e                                   # [B,N,latent]
    q1 = model.target_q1(x_next, e_next)           # [B,N,1]
    q2 = model.target_q2(x_next, e_next)
    q_min = torch.minimum(q1, q2)
    # N 在 dim=1；末维是长度 1 的标量维，必须显式 squeeze
    q_next = q_min.mean(dim=1).squeeze(-1)         # [B]
    if q_next.shape != (b,):
        raise RuntimeError(f"target 形状错误：期望 [{(b,)}]，得到 {tuple(q_next.shape)}（禁止 [B,B] 广播）")
    not_terminal = (~batch["terminated"]).to(q_next.dtype)
    return batch["reward"] + gamma * not_terminal * q_next


# ---------------------------------------------------------------- 校准正则
def calibration_regularizer_by_task(
    *,
    q_net: Any,
    model: Any,
    batch: Dict[str, torch.Tensor],
    mode: str,
    samples: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """按任务分别对 ref_valid 行求 mean，再加权求和（§7.3）。

    - 一个任务没有 valid 行时贡献 0（并记录覆盖率）；
    - 任务权重固定为双任务各 1/2，**不**由 valid 数量决定；
    - ``mode='off'`` 直接返回 0（replay 消融）；
    - ``mode='linear'`` 用 Q 代替 max(Q, G)（uncalibrated 消融）。
    """
    if mode not in ("max", "linear", "off"):
        raise ValueError(f"未知校准模式: {mode!r}")
    zero = torch.zeros((), dtype=batch["x"].dtype, device=batch["x"].device)
    if mode == "off":
        return zero, {"q_minus_reference_frac": {}, "coverage": {}}

    x = batch["x"]
    e_behavior = batch["e_behavior"]
    ref_valid = batch["ref_valid"]
    total = zero
    coverage: Dict[int, float] = {}
    below_frac: Dict[int, float] = {}
    task_ids = batch["task_id"]

    # 采样 e^m：对 β 停止梯度（校准项里的候选 latent 不作为策略梯度路径）
    with torch.no_grad():
        s = model.beta.rsample(x, samples=samples)
        e_candidates_sg = s.e                                 # [B,M,latent]，已 detach

    q_data = q_net(x, e_behavior)                             # [B,1]，保留 Q 的梯度
    present = sorted(set(int(t) for t in task_ids[ref_valid].tolist()))
    if not present:
        return zero, {"q_minus_reference_frac": {}, "coverage": {}}
    # 任务权重固定：双任务各 1/2，单任务 1。**不**由 valid 行数决定（§7.3）。
    task_weight = 0.5 if len(present) == 2 else 1.0

    for task in present:
        sel = ref_valid & (task_ids == task)
        n_valid = int(sel.sum().item())
        coverage[task] = n_valid / max(1, int((task_ids == task).sum().item()))
        if n_valid == 0:
            continue
        q_cand = q_net(x[sel], e_candidates_sg[sel])          # [n,M,1]
        g = batch["reference_return"][sel].detach()           # [n]
        if mode == "max":
            inside = torch.maximum(q_cand, g.reshape(-1, 1, 1))
        else:
            inside = q_cand
        # 先对该任务的 valid 行求 mean，再乘固定任务权重求和（§7.3）
        bracket = inside.mean(dim=1).squeeze(-1) - q_data[sel].squeeze(-1)
        total = total + task_weight * bracket.mean()
        with torch.no_grad():
            below_frac[task] = float(
                (q_data[sel].squeeze(-1).detach() < g).float().mean().item()
            )
    return total, {"q_minus_reference_frac": below_frac, "coverage": coverage}


def td_loss(
    *,
    q_net: Any,
    batch: Dict[str, torch.Tensor],
    target: torch.Tensor,
    weights: torch.Tensor,
    task_weighting: str = "mean_per_task",
) -> torch.Tensor:
    """L_TD = ½ · [w · (Q(x,e_behavior) - y)²] 的回归项（§7.1）。

    ``task_weighting='mean_per_task'``（默认）：先在每个任务内部求 mean，再按固定的
    任务权重（出现过的任务等权）求和。TD 因此是"按任务均衡"的，与校准正则一致，
    也让 §14 T6 要求的导数为 -1。

    ``task_weighting='batch_mean'``：整批直接 mean（§7.1 的字面写法），保留供对照。

    w 不按 batch 总和重新归一化，也不默认裁剪。
    """
    q_data = q_net(batch["x"], batch["e_behavior"]).squeeze(-1)
    if q_data.shape != target.shape:
        raise RuntimeError(f"Q 与 target 形状不一致: {tuple(q_data.shape)} vs {tuple(target.shape)}")
    if weights.shape != target.shape:
        raise RuntimeError(f"w 与 target 形状不一致: {tuple(weights.shape)} vs {tuple(target.shape)}")
    per_sample = weights * (q_data - target).pow(2)

    if task_weighting == "batch_mean":
        return 0.5 * per_sample.mean()
    if task_weighting != "mean_per_task":
        raise ValueError(f"未知 task_weighting: {task_weighting!r}")

    task_ids = batch["task_id"]
    present = sorted(set(int(t) for t in task_ids.tolist()))
    if len(present) <= 1:
        return 0.5 * per_sample.mean()
    weight = 1.0 / len(present)
    total = torch.zeros((), dtype=per_sample.dtype, device=per_sample.device)
    for task in present:
        sel = task_ids == task
        total = total + weight * per_sample[sel].mean()
    return 0.5 * total


# ---------------------------------------------------------------- actor
def actor_loss(
    *,
    model: Any,
    q1: Any,
    q2: Any,
    x: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L_β = E[α·log β(e|x) - min_i Q_i(x,e)]（§7.4）。

    返回 (loss, log_prob, q_actor)。e 用 rsample 重参数化；Q 对 e 的导数必须保留。
    """
    s = model.beta.rsample(x, samples=1)
    e = s.e_squeezed                       # [B,latent]
    logp = s.log_prob_squeezed             # [B]
    q = torch.minimum(q1(x, e), q2(x, e)).squeeze(-1)   # [B]
    if q.shape != logp.shape:
        raise RuntimeError(
            f"actor 路径形状不一致: q={tuple(q.shape)} logp={tuple(logp.shape)}（禁止 [B,B] 广播）"
        )
    return (alpha * logp - q).mean(), logp, q


# ---------------------------------------------------------------- learner
class ContinuousLearner:
    """持有 C 主方案的全部优化器，并暴露 ``update(batch)``。"""

    def __init__(self, model: Any, cfg: Optional[LearnerConfig] = None) -> None:
        self.model = model
        self.cfg = cfg or LearnerConfig()
        self.opt_q1 = torch.optim.Adam(model.q1.parameters(), lr=self.cfg.critic_lr)
        self.opt_q2 = torch.optim.Adam(model.q2.parameters(), lr=self.cfg.critic_lr)
        self.opt_beta = torch.optim.Adam(model.beta.parameters(), lr=self.cfg.actor_lr)
        self.update_count = 0

    # ---- 校正权重 ----
    def compute_weights(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """按每条 transition 自己的游戏计算 w（同一 batch 可混合两个任务）。"""
        w = torch.ones(batch["x"].shape[0], dtype=batch["x"].dtype, device=batch["x"].device)
        games: List[str] = batch["games"]
        for task in sorted(set(int(t) for t in batch["task_id"].tolist())):
            sel = batch["task_id"] == task
            game = games[int(torch.nonzero(sel)[0].item())]
            w_sel = mapping_correction(
                phi=self.model.phi,
                x=batch["x"][sel],
                e_behavior=batch["e_behavior"][sel],
                token=batch["global_token"][sel],
                log_b_action=batch["log_b_action"][sel],
                game=game,
            )
            w[sel] = w_sel
        return w

    # ---- 一次 learner 更新 ----
    def update(self, batch: Dict[str, torch.Tensor]) -> UpdateStats:
        """只更新允许的参数（φ/ψ 必须在调用前已冻结，§6.4 规则 4）。"""
        cfg = self.cfg
        stats = UpdateStats(n_transitions=int(batch["x"].shape[0]))

        # 0) φ/ψ 必须已冻结：训练期漂移解码器是被禁止的
        for name, p in self.model.named_parameters():
            if name.startswith(("phi.", "psi.")) and p.requires_grad:
                raise RuntimeError(
                    f"{name} 在改进阶段仍可训练：适配后必须冻结 φ/ψ（§6.4 规则 4）"
                )

        # 1) 校正权重 + target（target 用 sg，不建图）
        w = self.compute_weights(batch)
        rep: CorrectionReport = correction_report(w, "batch")
        stats.w_mean, stats.w_max = rep.mean, rep.max
        stats.ess_over_batch = rep.ess_over_batch

        target = compute_target(
            self.model, batch, gamma=cfg.gamma, target_samples=cfg.target_samples
        )
        stats.target_mean = float(target.mean().item())

        # 2) 两个 critic 各自更新（校准项里 Q 保留梯度，e/G 停止梯度）
        for q_net, opt, tag in (
            (self.model.q1, self.opt_q1, "q1"),
            (self.model.q2, self.opt_q2, "q2"),
        ):
            with torch.no_grad():
                q_before = q_net(batch["x"], batch["e_behavior"]).squeeze(-1).mean().item()
            td = td_loss(q_net=q_net, batch=batch, target=target, weights=w)
            reg, diag = calibration_regularizer_by_task(
                q_net=q_net, model=self.model, batch=batch,
                mode=cfg.calibration_mode, samples=cfg.calibration_samples,
            )
            loss = td + cfg.lambda_cal * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(q_net.parameters(), cfg.grad_clip)
            opt.step()
            if tag == "q1":
                stats.q1_loss = float(loss.item())
                stats.grad_norm_q1 = float(gn)
            else:
                stats.q2_loss = float(loss.item())
                stats.grad_norm_q2 = float(gn)
            # TD 项与校准项分别记录：便于报告区分"回归"与"保守正则"
            stats.td_loss = float(td.item())
            stats.cal_regularizer = float(reg.item())
            stats.reference_coverage = diag["coverage"]
            stats.q_minus_reference_frac = diag["q_minus_reference_frac"]
            stats.q_data_mean = q_before

        # 3) actor：critic 参数冻结，但 Q→e 的梯度必须保留
        with freeze_parameters(self.model.q1, self.model.q2):
            loss_a, logp, q_actor = actor_loss(
                model=self.model, q1=self.model.q1, q2=self.model.q2,
                x=batch["x"], alpha=cfg.entropy_alpha,
            )
            self.opt_beta.zero_grad(set_to_none=True)
            loss_a.backward()
            gn_a = torch.nn.utils.clip_grad_norm_(self.model.beta.parameters(), cfg.grad_clip)
            self.opt_beta.step()
        stats.actor_loss = float(loss_a.item())
        stats.actor_entropy = float(-logp.mean().item())
        stats.grad_norm_actor = float(gn_a)

        # 4) target 只做 Polyak
        polyak_update(self.model.q1, self.model.target_q1, cfg.tau)
        polyak_update(self.model.q2, self.model.target_q2, cfg.tau)
        self.update_count += 1
        return stats


# ---------------------------------------------------------------- 离线预训练
def offline_pretrain(
    model: Any,
    learner: ContinuousLearner,
    batches: Sequence[Dict[str, torch.Tensor]],
    *,
    updates: int,
) -> List[UpdateStats]:
    """§6.5：仅在首个任务表示适配完成后做一次 1000 次 β/Q 更新。"""
    if updates <= 0:
        raise ValueError("updates 必须为正")
    if not batches:
        raise ValueError("离线数据为空")
    out: List[UpdateStats] = []
    for i in range(updates):
        out.append(learner.update(batches[i % len(batches)]))
    return out
