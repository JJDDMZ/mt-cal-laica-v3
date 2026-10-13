"""C 主方案的连续网络：β（actor）、φ（可学习映射）、ψ（逆动力学）、双 Q（技术方案 §6，任务 T4）。

模块与张量（§6.1）：

===========  ==========================  =========================  ==========
模块          输入 → 输出                  初始结构                    训练阶段
===========  ==========================  =========================  ==========
β, 共享 actor  x[29] → μ[2], logσ[2]        29→64→64→4                 离线预训练、在线改进
φ, 共享映射    e[2] → 动作 logits[K]        线性层 2→K，含 bias         表示适配
ψ, 逆动力学    [x,x'][58] → μψ[2], logσψ[2]  58→64→64→4                 表示适配
Q1、Q2        [x,e][31] → 标量             31→64→64→1，各自独立        离线预训练、在线改进
===========  ==========================  =========================  ==========

φ 是**真正训练出来的网络**，不是固定查表；β 输出任意连续向量，不是从码本里选。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as tnf

from ..envs import features as F

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
OBS_DIM = F.OBS_DIM              # 29
LATENT_DIM = 2
MIN_LOG_PROB = -1e30             # 非法动作的 log 概率（有限但极低，便于检查）


# ---------------------------------------------------------------- 稳定 tanh log_prob
def tanh_log_det_jacobian(u: torch.Tensor) -> torch.Tensor:
    """log|det J| 的稳定形式（§7.4）。

    ``log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))``。
    直接用 ``log(1 - tanh(u)^2)`` 在 |u| 大时会因 1-tanh²→0 而失去精度甚至得到 -inf。
    """
    return 2.0 * (math.log(2.0) - u - tnf.softplus(-2.0 * u))


def tanh_squash_log_prob(
    u: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
) -> torch.Tensor:
    """tanh 压缩后 e=tanh(u) 的对数密度（u 空间，§7.4）。

    ``log β(e|x) = Σ_d [ log N(u_d; μ_d, σ_d) - log(1 - tanh²u_d) ]``。

    这里在 u 空间计算（未做逐元素的 -log(1-e²) 减法），因为噪声是加在 u 上；
    实现与方案公式逐项对应，并且在 e 接近 ±1 时仍然有限。
    """
    var = torch.exp(2.0 * log_std)
    log_norm = -0.5 * (((u - mean) ** 2) / var + 2.0 * log_std + math.log(2.0 * math.pi))
    return (log_norm - tanh_log_det_jacobian(u)).sum(dim=-1)


def gaussian_kl_to_standard(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """KL(N(μ,σ²) ‖ N(0,I)) = ½ Σ (μ² + σ² - 1 - 2 log σ)（§6.3）。"""
    return 0.5 * (mean.pow(2) + torch.exp(2.0 * log_std) - 1.0 - 2.0 * log_std).sum(dim=-1)


def mlp(sizes: Sequence[int], *, activate_last: bool = False) -> nn.Sequential:
    """Linear 堆叠 + ReLU。

    隐藏层一律 ReLU（§6.1）；**输出层默认不加激活**——Q 必须能取任意实数。
    早期版本在这里给最后一层也加了 ReLU，导致 Q 被压成非负、Bellman 回归失去
    表达能力，而且校准正则的符号测试会被"ReLU 把负值截成 0"掩盖。
    """
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        is_last = (i == len(sizes) - 2)
        if not is_last or activate_last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------- β
@dataclass
class Sample:
    """一次采样结果：e 与它在 u 空间的 log 概率。"""

    e: torch.Tensor                     # [B, N, latent_dim]
    log_prob: torch.Tensor              # [B, N]
    u: torch.Tensor                     # [B, N, latent_dim]
    mean: torch.Tensor                  # [B, N, latent_dim]
    std: torch.Tensor                   # [B, N, latent_dim]

    @property
    def e_squeezed(self) -> torch.Tensor:
        """N=1 时显式 squeeze 成 [B, latent_dim]，禁止依赖广播。"""
        if self.e.shape[1] != 1:
            raise ValueError(f"只有 N=1 才能 squeeze，当前 N={self.e.shape[1]}")
        return self.e[:, 0]

    @property
    def log_prob_squeezed(self) -> torch.Tensor:
        if self.log_prob.shape[1] != 1:
            raise ValueError(f"只有 N=1 才能 squeeze，当前 N={self.log_prob.shape[1]}")
        return self.log_prob[:, 0]


class Beta(nn.Module):
    """共享连续策略：x[29] → μ[2], logσ[2]，再 e = tanh(u)。"""

    def __init__(self, obs_dim: int = OBS_DIM, latent_dim: int = LATENT_DIM,
                 hidden: Sequence[int] = (64, 64), seed: int = 0) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.trunk = mlp((obs_dim, *hidden))
        self.head = nn.Linear(hidden[-1], 2 * latent_dim)
        self.gen = torch.Generator().manual_seed(int(seed))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """用**本模块自己的 generator** 初始化，保证同 seed 构建完全一致。

        不能依赖全局 RNG：两次构造若全局 seed 不同，参数就会不同，
        于是"同 seed 可重放"在模型层面就失效了（T4 复现性要求）。
        """
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0), generator=self.gen)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01, generator=self.gen)
        nn.init.zeros_(self.head.bias)

    def distribution_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        out = self.head(h)
        mean, log_std = out[..., : self.latent_dim], out[..., self.latent_dim:]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def rsample(
        self, x: torch.Tensor, samples: int = 1, *, explore: bool = True
    ) -> Sample:
        """重参数化采样；``samples`` 维放在中间（[B,N,latent]）。

        ``explore=False`` 时噪声置零（即 e=tanh(μ)），用于确定性诊断。
        """
        if samples < 1:
            raise ValueError("samples 必须 >= 1")
        mean, log_std = self.distribution_params(x)          # [B, latent]
        std = torch.exp(log_std)
        mean_n = mean.unsqueeze(1).expand(-1, samples, -1)   # [B,N,latent]
        std_n = std.unsqueeze(1).expand(-1, samples, -1)
        if explore:
            eps = torch.randn(mean_n.shape, generator=self.gen, dtype=mean_n.dtype)
        else:
            eps = torch.zeros_like(mean_n)
        u = mean_n + std_n * eps
        e = torch.tanh(u)
        logp = tanh_squash_log_prob(u, mean_n, log_std.unsqueeze(1).expand_as(mean_n))
        return Sample(e=e, log_prob=logp, u=u, mean=mean_n, std=std_n)

    def log_prob(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """给定 e 求 log β(e|x)（u=atanh(e)，数值稳定形式）。"""
        mean, log_std = self.distribution_params(x)
        e_clamped = e.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        u = torch.atanh(e_clamped)
        return tanh_squash_log_prob(u, mean, log_std)

    def deterministic_e(self, x: torch.Tensor) -> torch.Tensor:
        """greedy 部署用的 e = tanh(μ_β(x))（§12.1）。"""
        mean, _ = self.distribution_params(x)
        return torch.tanh(mean)


# ---------------------------------------------------------------- φ
class Phi(nn.Module):
    """可学习动作映射：e[2] → 动作 logits[K]。

    同时维护 **global token 注册表**（token → game/local_action），
    保证网络输出行与游戏动作的对应关系是显式的，ST-T 也适用（§13）。
    首次接入新游戏时按 §6.4 追加行，继承旧行权重/bias/Adam 矩。
    """

    def __init__(self, latent_dim: int = LATENT_DIM, seed: int = 0,
                 total_tokens: int = 4) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        # 固定 token 空间大小：§5.4 一共只有 4 个动作身份（F 两个 + T 两个）。
        # 输出层行号恒等于全局 token 号，因此 ST-T（token=2/3）也能直接按行取值，
        # "行号 == token 号" 这个不变量在整个生命周期内成立。
        if total_tokens < 2:
            raise ValueError("total_tokens 至少为 2")
        self.total_tokens = int(total_tokens)
        self.linear = nn.Linear(latent_dim, self.total_tokens, bias=True)
        # 未注册的行必须从"干净的零"开始：否则它们带着默认 kaiming 初始化，
        # 一旦哪次掩码写错就会给出看似合理的概率，掩码 bug 会被掩盖。
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.zero_()
        # 用一个 buffer 显式记录哪些 token 已注册
        self.register_buffer("_registered", torch.zeros(self.total_tokens, dtype=torch.bool))
        self.register_buffer("_token_ids", torch.zeros(0, dtype=torch.long))
        self._token_meta: List[Tuple[int, str, int]] = []   # (token, game, local)
        self.gen = torch.Generator().manual_seed(int(seed))

    # ---- 注册表 ----
    @property
    def num_tokens(self) -> int:
        return len(self._token_meta)

    @property
    def tokens(self) -> Tuple[int, ...]:
        return tuple(t for t, _, _ in self._token_meta)

    @property
    def registered(self) -> Dict[int, Tuple[str, int]]:
        """已注册 token → (game, local_action)。未注册的行不参与任何归一化。"""
        return {t: (g, a) for t, g, a in self._token_meta}

    def registered_games(self) -> Tuple[str, ...]:
        seen: List[str] = []
        for _, g, _ in self._token_meta:
            if g not in seen:
                seen.append(g)
        return tuple(seen)

    def register_actions(
        self,
        entries: Iterable[Tuple[int, str, int]],
        *,
        expect_contiguous: bool = True,
    ) -> None:
        """按全局 token 顺序注册动作；只允许在既有 token 之后追加（回合边界，§2）。

        ``expect_contiguous=True``（默认）要求 token 号连续递增，保证网络输出行号与
        token 号一致。ST-T 场景（首个任务就是 T，token=2/3）用
        ``expect_contiguous=False``，此时行号 ≠ token 号，注册表才是唯一的映射来源。
        """
        entries = list(entries)
        if not entries:
            return
        existing = list(self.tokens)
        expected_next = len(existing)
        for token, game, local in entries:
            if game not in F.GAMES:
                raise ValueError(f"未知游戏: {game!r}")
            if F.local_action(game, token) != local:
                raise ValueError(
                    f"token {token} 在 {game} 上应是本地动作 "
                    f"{F.local_action(game, token)}，得到 {local}"
                )
            if expect_contiguous:
                if token != expected_next:
                    raise ValueError(
                        f"token 必须按全局顺序追加：期望 {expected_next}，得到 {token}"
                    )
            else:
                if token <= (existing[-1] if existing else -1):
                    raise ValueError(
                        f"非连续注册也要求 token 严格递增：得到 {token}"
                    )
            expected_next += 1

        for token, _, _ in entries:
            if token >= self.total_tokens:
                raise ValueError(
                    f"token {token} 超出固定 token 空间 {self.total_tokens}"
                )
            if bool(self._registered[token]):
                raise ValueError(f"token {token} 已经注册过")

        # 只给新注册的行做随机初始化；已注册行保持原值（扩展前后概率不变）
        with torch.no_grad():
            for token, _, _ in entries:
                nn.init.normal_(self.linear.weight[token], std=0.01, generator=self.gen)
                nn.init.zeros_(self.linear.bias[token])
                self._registered[token] = True
        self._token_meta.extend([(int(t), str(g), int(a)) for t, g, a in entries])
        self._token_ids = torch.tensor([t for t, _, _ in self._token_meta], dtype=torch.long)

    # ---- 概率 ----
    def logits(self, e: torch.Tensor) -> torch.Tensor:
        return self.linear(e)

    def _mask_from_tokens(self, token_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """把 [B,L] 的合法 token 集合展开成 [B,total_tokens] 的 0/-inf 掩码。

        掩码按**全局 token 号**索引，因此对 ST-T（token=2/3）同样成立。
        """
        if self.num_tokens == 0:
            raise RuntimeError("φ 还没有注册任何动作")
        k = self.total_tokens
        mask = torch.full((token_ids.shape[0], k), float("-inf"), dtype=dtype)
        rows = torch.arange(token_ids.shape[0], device=token_ids.device)
        for i in range(token_ids.shape[1]):
            col = token_ids[:, i]
            if col.max().item() >= k or col.min().item() < 0:
                raise ValueError(f"合法 token 越界: {col.tolist()}（token 空间 {k}）")
            mask[rows, col] = 0.0
        return mask

    def legal_token_ids(self, game: str, batch: int, device: torch.device) -> torch.Tensor:
        toks = F.legal_tokens(game)
        return torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0).expand(batch, -1)

    def registered_only(self, logp: torch.Tensor, game: str) -> torch.Tensor:
        """把完整 token 空间的输出裁剪成"当前游戏合法 token"两列，便于调用方使用。"""
        toks = torch.tensor(F.legal_tokens(game), device=logp.device)
        return logp[:, toks]

    def log_probs(self, e: torch.Tensor, game: str) -> torch.Tensor:
        """在**当前游戏的合法 token** 内归一化（§13）。

        严格语义：非法项为 ``-inf``，合法项的 log 概率有限（归一化后和恰为 1）。
        ``-inf`` 只用于诊断/掩码断言，**不要**直接进入带梯度的损失
        （``softmax`` 会把非法项压成精确 0，梯度可能变 NaN）；需要梯度时用
        :meth:`finite_log_probs`。
        """
        toks = self.legal_token_ids(game, e.shape[0], e.device)
        mask = self._mask_from_tokens(toks, e.dtype)
        logits = self.logits(e) + mask
        return tnf.log_softmax(logits, dim=-1)

    def finite_log_probs(self, e: torch.Tensor, game: str) -> torch.Tensor:
        """与 :meth:`log_probs` 同值，但把非法项的 -inf 换成有限下界，便于求梯度。

        ``-inf + 梯度`` 在 log_softmax 的反向里会产生 NaN，因此训练路径统一走这里。
        """
        out = self.log_probs(e, game)
        illegal = torch.isinf(out) & (out < 0)
        return torch.where(illegal, torch.full_like(out, MIN_LOG_PROB), out)

    def probs(self, e: torch.Tensor, game: str) -> torch.Tensor:
        return torch.exp(self.log_probs(e, game))

    def probability_mass(self, e: torch.Tensor, game: str) -> torch.Tensor:
        """合法 token 上的概率（和=1），用于裁剪与诊断。"""
        p = self.probs(e, game)
        legal = torch.tensor(F.legal_tokens(game), device=e.device)
        return p[:, legal]

    # ---- 行为分布 b = (1-ε)φ + ε/|A|（§6.2）----
    def behavior_log_probs(self, e: torch.Tensor, game: str, *, uniform_mix: float) -> torch.Tensor:
        """混合分布 b 的 log 概率；只在实际会抽到的合法 token 上有效。

        注意：这里返回的是**混合分布**的 log 概率，训练时必须存这个值而不是 φ 的概率。
        取 log 时用 logsumexp 保持数值稳定：
            log b(k) = log( (1-ε)·φ(k) + ε/|A| )，仅当 k 合法；非法 token 为 -inf 语义。
        """
        if not (0.0 <= uniform_mix < 1.0):
            raise ValueError("uniform_mix 必须在 [0,1)")
        toks = self.legal_token_ids(game, e.shape[0], e.device)
        mask = self._mask_from_tokens(toks, e.dtype)
        logits = self.logits(e) + mask
        log_phi = tnf.log_softmax(logits, dim=-1)
        phi = torch.exp(log_phi)
        n_legal = len(F.legal_tokens(game))
        mixed = (1.0 - uniform_mix) * phi + uniform_mix / n_legal
        # 非法项：phi=0 → mixed = ε/|A| > 0，必须显式压回 -inf 语义
        valid = torch.isfinite(log_phi)
        log_mixed = torch.log(torch.clamp(mixed, min=1e-45))
        return torch.where(valid, log_mixed, torch.full_like(log_mixed, float("-inf")))

    def behavior_log_prob_of(self, e: torch.Tensor, game: str, token: torch.Tensor,
                             *, uniform_mix: float) -> torch.Tensor:
        """取 b(指定 token | e, game)，用于记录 log_b_action（§6.2）。"""
        logb = self.behavior_log_probs(e, game, uniform_mix=uniform_mix)
        idx = token.reshape(-1, 1)
        gathered = torch.gather(logb, 1, idx).squeeze(1)
        if torch.isinf(gathered).any():
            raise ValueError("请求了 b 在非法 token 上的概率：行为概率记录有错")
        return gathered

    def greedy_token(self, e: torch.Tensor, game: str) -> torch.Tensor:
        """argmax φ(j|e,g) over 合法 token；并列取最小 token（§12.1）。"""
        toks = self.legal_token_ids(game, e.shape[0], e.device)
        mask = self._mask_from_tokens(toks, e.dtype)
        logits = self.logits(e) + mask
        # argmax 天然取第一个最大值 → 即最小 token
        return torch.argmax(logits, dim=-1)

    def sample_token(self, e: torch.Tensor, game: str, *, uniform_mix: float = 0.0) -> torch.Tensor:
        """从 b（uniform_mix>0）或纯 φ（uniform_mix=0）抽一个真实 token。"""
        if uniform_mix > 0.0:
            logb = self.behavior_log_probs(e, game, uniform_mix=uniform_mix)
        else:
            logb = self.finite_log_probs(e, game)
        # 非法项是 -inf / MIN_LOG_PROB；用 gumbel 技巧从 logits 采样
        probs = torch.softmax(logb, dim=-1)
        return torch.multinomial(probs.clamp_min(0.0), 1, generator=self.gen).squeeze(1)

    # ---- 诊断 ----
    def coverage_grid(self, game: str, res: int = 41) -> Dict[str, Any]:
        """§6.4：41×41 个 e 网格点上 φ 的 argmax 覆盖；每个合法动作应至少一次成为 argmax。"""
        lin = torch.linspace(-0.99, 0.99, res)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        e = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        with torch.no_grad():
            tok = self.greedy_token(e, game).reshape(res, res)
        counts = {t: int((tok == t).sum().item()) for t in F.legal_tokens(game)}
        return {
            "grid": tok.cpu().numpy(),
            "counts": counts,
            "all_actions_reachable": all(c > 0 for c in counts.values()),
        }


# ---------------------------------------------------------------- ψ
class Psi(nn.Module):
    """共享逆动力学：[x,x'][58] → μψ[2], logσψ[2]。

    **不读取 j、reward 或未来回报**；只通过动作重构的监督目标看到 j（§6.3）。
    """

    def __init__(self, obs_dim: int = OBS_DIM, latent_dim: int = LATENT_DIM,
                 hidden: Sequence[int] = (64, 64), seed: int = 0) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.trunk = mlp((2 * obs_dim, *hidden))
        self.head = nn.Linear(hidden[-1], 2 * latent_dim)
        self.gen = torch.Generator().manual_seed(int(seed))
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0), generator=self.gen)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head.weight, gain=0.01, generator=self.gen)
        nn.init.zeros_(self.head.bias)

    def distribution_params(self, x: torch.Tensor, next_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([x, next_x], dim=-1))
        out = self.head(h)
        mean, log_std = out[..., : self.latent_dim], out[..., self.latent_dim:]
        return mean, torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

    def rsample(self, x: torch.Tensor, next_x: torch.Tensor, samples: int = 1) -> Sample:
        mean, log_std = self.distribution_params(x, next_x)
        std = torch.exp(log_std)
        mean_n = mean.unsqueeze(1).expand(-1, samples, -1)
        std_n = std.unsqueeze(1).expand(-1, samples, -1)
        eps = torch.randn(mean_n.shape, generator=self.gen, dtype=mean_n.dtype)
        u = mean_n + std_n * eps
        e = torch.tanh(u)
        logp = tanh_squash_log_prob(u, mean_n, log_std.unsqueeze(1).expand_as(mean_n))
        return Sample(e=e, log_prob=logp, u=u, mean=mean_n, std=std_n)


# ---------------------------------------------------------------- Q
class QNet(nn.Module):
    """[x,e][31] → 标量。Q1/Q2 各自独立实例（§6.1）。"""

    def __init__(self, obs_dim: int = OBS_DIM, latent_dim: int = LATENT_DIM,
                 hidden: Sequence[int] = (64, 64), seed: int = 0) -> None:
        super().__init__()
        self.net = mlp((obs_dim + latent_dim, *hidden, 1))
        self.gen = torch.Generator().manual_seed(int(seed))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0), generator=self.gen)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """支持 [B, latent] 与 [B, N, latent]；输出 [B,1] 或 [B,N,1]（广播只在特征维）。"""
        if e.dim() == 2:
            x_in = x
        elif e.dim() == 3:
            x_in = x.unsqueeze(1).expand(-1, e.shape[1], -1)
        else:
            raise ValueError(f"e 必须是 [B,latent] 或 [B,N,latent]，得到 {tuple(e.shape)}")
        return self.net(torch.cat([x_in, e], dim=-1))


class ContinuousModel(nn.Module):
    """把 β/φ/ψ/Q1/Q2 与 target 打包，方便 checkpoint 与冻结管理。"""

    def __init__(self, *, obs_dim: int = OBS_DIM, latent_dim: int = LATENT_DIM,
                 hidden: Sequence[int] = (64, 64), seed: int = 0) -> None:
        super().__init__()
        self.beta = Beta(obs_dim, latent_dim, hidden, seed=seed)
        self.phi = Phi(latent_dim, seed=seed + 1, total_tokens=len(F.ACTION_REGISTRY))
        self.psi = Psi(obs_dim, latent_dim, hidden, seed=seed + 2)
        self.q1 = QNet(obs_dim, latent_dim, hidden, seed=seed + 3)
        self.q2 = QNet(obs_dim, latent_dim, hidden, seed=seed + 4)
        self.target_q1 = QNet(obs_dim, latent_dim, hidden, seed=seed + 5)
        self.target_q2 = QNet(obs_dim, latent_dim, hidden, seed=seed + 6)
        self.hard_update_targets()
        # 供 schedule 的冻结表分派使用
        self.variant_name = "continuous"

    def hard_update_targets(self) -> None:
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

    def generator_state(self) -> Dict[str, torch.Tensor]:
        """返回模块采样器的私有 RNG 状态。"""
        return {
            "beta": self.beta.gen.get_state().clone(),
            "phi": self.phi.gen.get_state().clone(),
            "psi": self.psi.gen.get_state().clone(),
        }

    def set_generator_state(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = {"beta", "phi", "psi"}
        if set(state) != expected:
            raise ValueError(f"component RNG 字段错误: {sorted(state)}")
        self.beta.gen.set_state(state["beta"])
        self.phi.gen.set_state(state["phi"])
        self.psi.gen.set_state(state["psi"])

    def register_game(self, game: str) -> None:
        """按 §5.4 的固定表追加该游戏的 token（首次接入时调用一次）。"""
        if game in self.phi.registered_games():
            raise ValueError(f"{game} 已经注册过，重复注册会破坏 mapping_version 语义")
        entries = [(F.token_for_local(game, a), game, a) for a in (0, 1)]
        self.phi.register_actions(entries)

    def register_game_as_first(self, game: str) -> None:
        """ST-T 专用：直接注册 T，不要求先训练 F（§10 的 ST-T）。

        全局 token 仍是 2/3，因此网络输出行号与 token 号不相等——
        这正是"不能假设 token == 输出行号"的验收场景。
        """
        if self.phi.num_tokens != 0:
            raise ValueError("register_game_as_first 只能在尚未注册任何动作时使用")
        entries = [(F.token_for_local(game, a), game, a) for a in (0, 1)]
        # ST-T：允许首个 token 不是 0
        self.phi.register_actions(entries, expect_contiguous=False)

    def parameter_hash(self) -> str:
        """用于断言"适配期间 β/Q 不变"等冻结规则。"""
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


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for ps, pt in zip(source.parameters(), target.parameters()):
        pt.mul_(1.0 - tau).add_(ps, alpha=tau)
    for bs, bt in zip(source.buffers(), target.buffers()):
        bt.copy_(bs)


class freeze_parameters:
    """临时冻结参数（§7.5：不等于 no_grad，Q 对 e 的梯度必须保留）。

    用法::

        with freeze_parameters(q1, q2):
            actor_loss = ...
    """

    def __init__(self, *modules: nn.Module) -> None:
        self.modules = modules
        self._saved: List[Tuple[torch.nn.Parameter, bool]] = []

    def __enter__(self):
        for m in self.modules:
            for p in m.parameters():
                self._saved.append((p, p.requires_grad))
                p.requires_grad_(False)
        return self

    def __exit__(self, *exc):
        for p, flag in self._saved:
            p.requires_grad_(flag)
        self._saved.clear()
        return False


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
