"""D（离散码本）的学习目标（技术方案 §16.2）。

与 C 的关键区别（不能混写）：

- **w = 1**：token 与真实动作是固定翻译，离散 TD 已按实际 token 定义动作价值，
  所以不需要 C 的映射概率比。不能把 C 的规则照搬到 D，也不能反过来。
- **枚举合法动作**：target 与校准正则都在当前游戏的两个合法动作上枚举，
  不做连续采样。
- actor 通过离散概率 π 的梯度更新，**Q 输出必须 detach**；
  而 C 依赖 Q→e 的梯度，绝不能 detach。这是两者最容易写错的分歧点。

公式（§16.2）：

    y_D = sg[ r + γ(1-terminated) Σ_{j∈A_g} π(j|x') min_i Q̄_i(x', j) ]
    R_D,i^cal = mean_valid[ Σ_j sg(π(j|x)) max(q_i(x,j), sg(G)) - q_i(x, j_executed) ]
    L_actor,D = mean Σ_j π(j|x) [ α log π(j|x) - sg(min_i q_i(x,j)) ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from .envs import features as F
from .models.discrete import DiscreteModel


@dataclass
class DiscreteLearnerConfig:
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    entropy_alpha: float = 0.02
    lambda_cal: float = 0.1
    tau: float = 0.005
    grad_clip: float = 10.0
    calibration_mode: str = "max"      # max / linear / off

    def __post_init__(self) -> None:
        if self.calibration_mode not in ("max", "linear", "off"):
            raise ValueError(f"未知 calibration_mode: {self.calibration_mode!r}")


@dataclass
class DiscreteUpdateStats:
    td_loss: float = 0.0
    cal_regularizer: float = 0.0
    q1_loss: float = 0.0
    q2_loss: float = 0.0
    actor_loss: float = 0.0
    grad_norm_q1: float = 0.0
    grad_norm_q2: float = 0.0
    grad_norm_actor: float = 0.0
    reference_coverage: Dict[int, float] = None
    target_mean: float = 0.0
    q_data_mean: float = 0.0
    n_transitions: int = 0

    def __post_init__(self) -> None:
        if self.reference_coverage is None:
            self.reference_coverage = {}


def _legal_action_slice(game: str, device: torch.device) -> torch.Tensor:
    return torch.tensor(list(F.legal_tokens(game)), dtype=torch.long, device=device)


@torch.no_grad()
def discrete_target(
    model: DiscreteModel,
    *,
    next_x: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    games: List[str],
    gamma: float,
) -> torch.Tensor:
    """y_D = sg[r + γ(1-term) Σ_{j∈A_g} π(j|x') min_i Q̄_i(x',j)]。

    两个游戏的动作集合不同，因此按游戏分组计算后写回各自位置。
    """
    b = next_x.shape[0]
    out = torch.zeros(b, dtype=reward.dtype, device=reward.device)
    term_flag = (~terminated).to(reward.dtype)
    for game in sorted(set(games)):
        sel = torch.tensor([g == game for g in games], device=next_x.device)
        if not bool(sel.any()):
            continue
        xg = next_x[sel]
        legal = _legal_action_slice(game, xg.device)
        pi = model.actor.probs(xg, model.codebook, game)[:, legal]        # [n,|A|]
        q1 = model.target_q1(xg, model.codebook)[:, legal]                # [n,|A|]
        q2 = model.target_q2(xg, model.codebook)[:, legal]
        qmin = torch.minimum(q1, q2)                                      # [n,|A|]
        boot = (pi * qmin).sum(dim=-1)                                    # [n]
        out[sel] = reward[sel] + gamma * term_flag[sel] * boot
    if out.shape != (b,):
        raise RuntimeError(f"target 形状错误: {tuple(out.shape)}")
    return out


def discrete_td_loss(
    model: DiscreteModel,
    *,
    x: torch.Tensor,
    token: torch.Tensor,
    target: torch.Tensor,
    q_net: Any,
) -> torch.Tensor:
    """L_TD = ½ mean[(q(x, token) - y)²]，**w = 1**（§16.2）。"""
    rows = torch.arange(x.shape[0], device=x.device)
    q_all = q_net(x, model.codebook)
    q_data = q_all[rows, token]
    if q_data.shape != target.shape:
        raise RuntimeError(f"Q 与 target 形状不一致: {tuple(q_data.shape)} vs {tuple(target.shape)}")
    return 0.5 * (q_data - target).pow(2).mean()


def discrete_calibration(
    model: DiscreteModel,
    *,
    q_net: Any,
    x: torch.Tensor,
    token: torch.Tensor,
    reference: torch.Tensor,
    ref_valid: torch.Tensor,
    games: List[str],
    mode: str,
) -> Tuple[torch.Tensor, Dict[int, float]]:
    """R_D,i^cal = mean_valid[ Σ_j sg(π(j|x)) max(q_i(x,j), sg(G)) - q_i(x, j_exec) ]。

    §7.3 的严格性同样适用：任务权重固定、按 valid 行求 mean、无 valid 则贡献 0。
    """
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    if mode == "off":
        return zero, {}
    coverage: Dict[int, float] = {}
    present_games = sorted(set(g for g, v in zip(games, ref_valid.tolist()) if v))
    if not present_games:
        return zero, {}
    task_weight = 0.5 if len(present_games) == 2 else 1.0
    total = zero

    for game in present_games:
        sel = torch.tensor([(g == game) and v for g, v in zip(games, ref_valid.tolist())],
                           device=x.device)
        n_valid = int(sel.sum().item())
        total_task = int(sum(1 for g in games if g == game))
        coverage[F.task_id(game)] = n_valid / max(1, total_task)
        if n_valid == 0:
            continue
        xg = x[sel]
        legal = _legal_action_slice(game, xg.device)
        with torch.no_grad():
            pi = model.actor.probs(xg, model.codebook, game)[:, legal]     # [n,|A|] → sg
            g_ref = reference[sel].detach()
        q_all = q_net(xg, model.codebook)[:, legal]                        # [n,|A|]，保留梯度
        if mode == "max":
            inside = torch.maximum(q_all, g_ref.unsqueeze(-1))
        else:
            inside = q_all
        rows = torch.arange(xg.shape[0], device=xg.device)
        # 已执行动作在全局 token 空间里的下标
        tok_local = torch.tensor(
            [list(F.legal_tokens(game)).index(int(t)) for t in token[sel].tolist()],
            device=xg.device)
        q_exec = q_all[rows, tok_local]
        bracket = (pi * inside).sum(dim=-1) - q_exec
        total = total + task_weight * bracket.mean()
    return total, coverage


def discrete_actor_loss(
    model: DiscreteModel,
    *,
    x: torch.Tensor,
    games: List[str],
    alpha: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """L_actor,D = mean Σ_j π(j|x)[α log π(j|x) - sg(min_i q_i(x,j))]。

    **Q 必须 detach**（与 C 相反）。
    """
    loss_terms = []
    entropies = []
    for game in sorted(set(games)):
        sel = torch.tensor([g == game for g in games], device=x.device)
        if not bool(sel.any()):
            continue
        xg = x[sel]
        legal = _legal_action_slice(game, xg.device)
        logp = model.actor.log_probs(xg, model.codebook, game)[:, legal]    # [n,|A|]
        pi = torch.exp(logp)
        with torch.no_grad():
            q1 = model.q1(xg, model.codebook)[:, legal]
            q2 = model.q2(xg, model.codebook)[:, legal]
            qmin = torch.minimum(q1, q2)                                    # detach 正确
        per_sample = (pi * (alpha * logp - qmin)).sum(dim=-1)
        loss_terms.append(per_sample.mean())
        with torch.no_grad():
            entropies.append(float(-(pi * logp).sum(dim=-1).mean().item()))
    if not loss_terms:
        raise ValueError("batch 里没有任何已知游戏")
    loss = torch.stack(loss_terms).mean()
    return loss, {"entropy": float(sum(entropies) / len(entropies))}


class DiscreteLearner:
    """D 的 learner：更新 actor 与双 Q；码本在适配后冻结（§16.1）。"""

    def __init__(self, model: DiscreteModel, cfg: Optional[DiscreteLearnerConfig] = None) -> None:
        self.model = model
        self.cfg = cfg or DiscreteLearnerConfig()
        self.opt_q1 = torch.optim.Adam(model.q1.parameters(), lr=self.cfg.critic_lr)
        self.opt_q2 = torch.optim.Adam(model.q2.parameters(), lr=self.cfg.critic_lr)
        self.opt_actor = torch.optim.Adam(model.actor.parameters(), lr=self.cfg.actor_lr)
        self.update_count = 0

    def update(self, batch: Dict[str, Any]) -> DiscreteUpdateStats:
        cfg = self.cfg
        model = self.model
        # 改进阶段：码本必须已冻结（§16.1），否则等价于偷偷学表示
        for i, row in enumerate(model.codebook.rows):
            if row.requires_grad:
                raise RuntimeError(
                    f"码本第 {i} 行在改进阶段仍可训练：适配后必须冻结 E（§16.1）")
        stats = DiscreteUpdateStats(n_transitions=int(batch["x"].shape[0]))
        games = batch["games"]
        target = discrete_target(
            model, next_x=batch["next_x"], reward=batch["reward"],
            terminated=batch["terminated"], games=games, gamma=cfg.gamma)
        stats.target_mean = float(target.mean().item())

        for q_net, opt, tag in ((model.q1, self.opt_q1, "q1"), (model.q2, self.opt_q2, "q2")):
            td = discrete_td_loss(model, x=batch["x"], token=batch["global_token"],
                                  target=target, q_net=q_net)
            reg, cov = discrete_calibration(
                model, q_net=q_net, x=batch["x"], token=batch["global_token"],
                reference=batch["reference_return"], ref_valid=batch["ref_valid"],
                games=games, mode=cfg.calibration_mode)
            loss = td + cfg.lambda_cal * reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(q_net.parameters(), cfg.grad_clip)
            opt.step()
            if tag == "q1":
                stats.q1_loss, stats.grad_norm_q1 = float(loss.item()), float(gn)
            else:
                stats.q2_loss, stats.grad_norm_q2 = float(loss.item()), float(gn)
            stats.td_loss = float(td.item())
            stats.cal_regularizer = float(reg.item())
            stats.reference_coverage = cov

        loss_a, info = discrete_actor_loss(model, x=batch["x"], games=games,
                                           alpha=cfg.entropy_alpha)
        self.opt_actor.zero_grad(set_to_none=True)
        loss_a.backward()
        gn_a = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), cfg.grad_clip)
        self.opt_actor.step()
        stats.actor_loss = float(loss_a.item())
        stats.grad_norm_actor = float(gn_a)

        with torch.no_grad():
            from .models.continuous import polyak_update
            polyak_update(model.q1, model.target_q1, cfg.tau)
            polyak_update(model.q2, model.target_q2, cfg.tau)
        self.update_count += 1
        return stats
