"""D：离散码本对照（技术方案 §16，任务 T8/T9）。

与 C 的实际区别（§16.1）：

===========  ==========================================================
动作码本 E    4×8，每 token 一行；初始化为 8 维空间的前四个单位基向量
actor        x[29]→64→64→h[8]；合法 logits = h·E_j，π = masked softmax
ψD           [x,x'][58]→64→64→z[8]；用 z·E_j 重构真实 token
Q1/Q2        [x,E_j][37]→64→64→1；各自独立 target
===========  ==========================================================

D **不是**完整 LAICA：它保留共享表示、表示适配、历史回放与价值校准，但 token 与游戏
动作之间是固定翻译，不学习连续 e→动作概率分区。TD 因此 w=1（§16.2）。

表示适配只更新 ψD **和当前游戏的两行 E**；已学游戏的 E 严格冻结——用按行独立的
``nn.Parameter`` 实现，只把梯度置零并不能保证 Adam 的旧行不被动量改到（§16.1）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as tnf

from ..envs import features as F
from .continuous import mlp, polyak_update

EMBEDDING_DIM = 8
TOTAL_TOKENS = len(F.ACTION_REGISTRY)     # 4


class Codebook(nn.Module):
    """4×8 动作码本；每行是独立的 ``nn.Parameter``，便于按行冻结。"""

    def __init__(self, token_ids: Sequence[int], embedding_dim: int = EMBEDDING_DIM,
                 seed: int = 0) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.gen = torch.Generator().manual_seed(int(seed))
        self.rows = nn.ParameterList()
        self._token_ids: List[int] = []
        basis = torch.eye(self.embedding_dim)
        # 初始化为 8 维空间的前四个单位基向量（§16.1）
        for i, tok in enumerate(sorted(token_ids)):
            if i >= self.embedding_dim:
                raise ValueError(f"token 数 {len(token_ids)} 超过 embedding_dim {self.embedding_dim}")
            self.rows.append(nn.Parameter(basis[i].clone()))
            self._token_ids.append(int(tok))

    @property
    def token_ids(self) -> Tuple[int, ...]:
        return tuple(self._token_ids)

    def row_for(self, token: int) -> int:
        if token not in self._token_ids:
            raise ValueError(f"token {token} 不在码本中（已注册 {self._token_ids}）")
        return self._token_ids.index(token)

    def forward(self) -> torch.Tensor:
        """返回 [total_tokens, embedding_dim]，未注册的行为零向量。"""
        out = torch.zeros(TOTAL_TOKENS, self.embedding_dim, dtype=self.rows[0].dtype
                          if len(self.rows) else torch.float32)
        for i, tok in enumerate(self._token_ids):
            out[tok] = self.rows[i]
        return out

    @torch.no_grad()
    def normalise_current(self, game: str) -> None:
        """§16.1：每次表示更新后把当前游戏的两行归一化到单位范数；范数为 0 报错。"""
        for tok in F.legal_tokens(game):
            i = self.row_for(tok)
            norm = float(self.rows[i].norm().item())
            if norm < 1e-12:
                raise ValueError(f"token {tok} 的行范数为 0，无法归一化")
            self.rows[i].div_(norm)

    def set_trainable_tokens(self, trainable: Sequence[int]) -> None:
        """只让指定 token 行可训练（旧游戏的 E 严格冻结，§16.1）。"""
        for i, tok in enumerate(self._token_ids):
            self.rows[i].requires_grad_(int(tok) in set(int(t) for t in trainable))


class DiscreteActor(nn.Module):
    """x[29]→64→64→h[8]；合法 logits = h·E_j（§16.1）。"""

    def __init__(self, obs_dim: int = F.OBS_DIM, hidden: Sequence[int] = (64, 64),
                 embedding_dim: int = EMBEDDING_DIM, seed: int = 0) -> None:
        super().__init__()
        self.trunk = mlp((obs_dim, *hidden, embedding_dim))
        self.gen = torch.Generator().manual_seed(int(seed))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def logits(self, x: torch.Tensor, codebook: Codebook) -> torch.Tensor:
        h = self.hidden(x)                       # [B,8]
        E = codebook()                           # [4,8]
        return h @ E.t()                         # [B,4]

    def log_probs(self, x: torch.Tensor, codebook: Codebook, game: str) -> torch.Tensor:
        """masked softmax：非法项为 -inf（合法项有限且和为 1）。"""
        legal = F.legal_tokens(game)
        logits = self.logits(x, codebook)
        mask = torch.full_like(logits, float("-inf"))
        mask[:, list(legal)] = 0.0
        return tnf.log_softmax(logits + mask, dim=-1)

    def probs(self, x: torch.Tensor, codebook: Codebook, game: str) -> torch.Tensor:
        return torch.exp(self.log_probs(x, codebook, game))

    def greedy_token(self, x: torch.Tensor, codebook: Codebook, game: str) -> torch.Tensor:
        """argmax π(j|x)，并列取最小 token（§12.1）。"""
        legal = list(F.legal_tokens(game))
        logits = self.logits(x, codebook)
        masked = torch.full_like(logits, float("-inf"))
        masked[:, legal] = logits[:, legal]
        return torch.argmax(masked, dim=-1)

    def sample_token(self, x: torch.Tensor, codebook: Codebook, game: str) -> torch.Tensor:
        p = self.probs(x, codebook, game)
        return torch.multinomial(p.clamp_min(0.0), 1, generator=self.gen).squeeze(1)


class PsiDiscrete(nn.Module):
    """[x,x'][58]→64→64→z[8]；用 z·E_j 重构真实 token（§16.1）。"""

    def __init__(self, obs_dim: int = F.OBS_DIM, hidden: Sequence[int] = (64, 64),
                 embedding_dim: int = EMBEDDING_DIM, seed: int = 0) -> None:
        super().__init__()
        self.trunk = mlp((2 * obs_dim, *hidden, embedding_dim))

    def forward(self, x: torch.Tensor, next_x: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([x, next_x], dim=-1))


class QDiscrete(nn.Module):
    """[x, E_j][37]→64→64→1（§16.1）。"""

    def __init__(self, obs_dim: int = F.OBS_DIM, embedding_dim: int = EMBEDDING_DIM,
                 hidden: Sequence[int] = (64, 64), seed: int = 0) -> None:
        super().__init__()
        self.net = mlp((obs_dim + embedding_dim, *hidden, 1))

    def forward(self, x: torch.Tensor, codebook: Codebook) -> torch.Tensor:
        """返回 [B, total_tokens] 的每个合法 token 价值。"""
        E = codebook()                                  # [4,8]
        x_rep = x.unsqueeze(1).expand(-1, E.shape[0], -1)
        e_rep = E.unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.net(torch.cat([x_rep, e_rep], dim=-1)).squeeze(-1)


class DiscreteModel(nn.Module):
    """D 的完整模块集合。"""

    def __init__(self, *, obs_dim: int = F.OBS_DIM, hidden: Sequence[int] = (64, 64),
                 embedding_dim: int = EMBEDDING_DIM, seed: int = 0) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.codebook = Codebook(list(range(TOTAL_TOKENS)), embedding_dim, seed=seed)
        self.actor = DiscreteActor(obs_dim, hidden, embedding_dim, seed=seed + 1)
        self.psi = PsiDiscrete(obs_dim, hidden, embedding_dim, seed=seed + 2)
        self.q1 = QDiscrete(obs_dim, embedding_dim, hidden, seed=seed + 3)
        self.q2 = QDiscrete(obs_dim, embedding_dim, hidden, seed=seed + 4)
        self.target_q1 = QDiscrete(obs_dim, embedding_dim, hidden, seed=seed + 5)
        self.target_q2 = QDiscrete(obs_dim, embedding_dim, hidden, seed=seed + 6)
        self.hard_update_targets()
        self.registered_games: List[str] = []
        # 供 schedule 的冻结表分派使用（不用靠类名猜）
        self.variant_name = "discrete"

    def hard_update_targets(self) -> None:
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

    # ---- 注册与冻结 ----
    def register_game(self, game: str) -> None:
        """登记该游戏的动作行；已学游戏的行保持冻结（§16.1）。"""
        if game in self.registered_games:
            raise ValueError(f"{game} 已经注册过")
        if not self.registered_games and game == "Timberman":
            # ST-T：首个游戏就是 T
            pass
        self.registered_games.append(game)
        self.codebook.set_trainable_tokens([])

    def freeze_embeddings_except(self, game: str) -> None:
        """表示适配时只放开当前游戏的两行。"""
        self.codebook.set_trainable_tokens(F.legal_tokens(game))

    def freeze_all_embeddings(self) -> None:
        self.codebook.set_trainable_tokens([])

    def parameter_hash(self) -> str:
        import hashlib

        h = hashlib.sha256()
        for name, p in sorted(self.named_parameters()):
            h.update(name.encode())
            h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def subset_hash(self, prefixes: Sequence[str]) -> str:
        import hashlib

        h = hashlib.sha256()
        for name, p in sorted(self.named_parameters()):
            if any(name.startswith(pre) for pre in prefixes):
                h.update(name.encode())
                h.update(p.detach().cpu().numpy().tobytes())
        return h.hexdigest()


# ---------------------------------------------------------------- 表示损失
@dataclass
class DiscreteReprDiag:
    loss: float = 0.0
    ce: float = 0.0
    sep: float = 0.0
    accuracy: float = 0.0
    n_updates: int = 0


def discrete_repr_loss(
    model: DiscreteModel,
    *,
    x: torch.Tensor,
    next_x: torch.Tensor,
    token: torch.Tensor,
    game: str,
    separation_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """§16.1：当前游戏动作重构交叉熵 + ``0.1·max(0, 0.5-||E_a-E_b||₂)²``。"""
    if token.shape[0] != x.shape[0]:
        raise ValueError("token 与 x 的 batch 不一致")
    z = model.psi(x, next_x)                       # [B,8]
    E = model.codebook()                           # [4,8]
    logits = z @ E.t()                             # [B,4]
    legal = list(F.legal_tokens(game))
    mask = torch.full_like(logits, float("-inf"))
    mask[:, legal] = 0.0
    logp = tnf.log_softmax(logits + mask, dim=-1)
    rows = torch.arange(token.shape[0])
    ce = -logp[rows, token].mean()

    a, b = legal
    dist = (E[a] - E[b]).norm(p=2)
    sep = separation_weight * torch.clamp(0.5 - dist, min=0.0) ** 2
    loss = ce + sep

    with torch.no_grad():
        pred = logp.argmax(dim=-1)
        acc = (pred == token).float().mean()
    return loss, {"loss": loss, "ce": ce, "sep": sep, "accuracy": acc}


def adapt_discrete_representation(
    model: DiscreteModel,
    *,
    data: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    game: str,
    updates: int,
    lr: float = 1e-3,
    separation_weight: float = 0.1,
    grad_clip: float = 10.0,
) -> DiscreteReprDiag:
    """只更新 ψD 与当前游戏的两行 E；actor/Q/target 冻结（§16.1）。"""
    if updates <= 0:
        raise ValueError("updates 必须为正")
    if not data:
        raise ValueError("适配数据为空")
    model.freeze_embeddings_except(game)
    params = list(model.psi.parameters()) + [
        model.codebook.rows[model.codebook.row_for(tok)] for tok in F.legal_tokens(game)
    ]
    opt = torch.optim.Adam(params, lr=lr)
    frozen = [p for n, p in model.named_parameters()
              if not (n.startswith("psi.") or n.startswith("codebook."))]
    saved = [(p, p.requires_grad) for p in frozen]
    for p, _ in saved:
        p.requires_grad_(False)

    diag = DiscreteReprDiag()
    try:
        for i in range(updates):
            x, nx, tok = data[i % len(data)]
            loss, stats = discrete_repr_loss(
                model, x=x, next_x=nx, token=tok, game=game,
                separation_weight=separation_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            # 每次更新后把当前两行归一化
            model.codebook.normalise_current(game)
            diag.loss = float(stats["loss"].item())
            diag.ce = float(stats["ce"].item())
            diag.sep = float(stats["sep"].item())
            diag.accuracy = float(stats["accuracy"].item())
            diag.n_updates = i + 1
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)
        model.freeze_all_embeddings()
    return diag
