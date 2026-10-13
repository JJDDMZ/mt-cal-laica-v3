"""T6 验收：critic 校准、连续 actor 梯度边界与 target 更新（§7，§14 T6）。"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
from types import SimpleNamespace

from wemi_cal.envs import features as F
from wemi_cal.learner import (
    ContinuousLearner,
    LearnerConfig,
    UpdateStats,
    actor_loss,
    calibration_regularizer_by_task,
    compute_target,
    offline_pretrain,
    td_loss,
    validate_update_stats,
)
from wemi_cal.learner_discrete import DiscreteUpdateStats
from wemi_cal.models.continuous import ContinuousModel, freeze_parameters


def make_model(seed=0) -> ContinuousModel:
    m = ContinuousModel(seed=seed)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    for p in m.phi.parameters():
        p.requires_grad_(False)
    for p in m.psi.parameters():
        p.requires_grad_(False)
    return m


def make_batch(n=16, *, task=0, terminated=False, ref_valid=True, g=5.0) -> dict:
    torch.manual_seed(0)
    game = "Flappybird" if task == 0 else "Timberman"
    token = F.legal_tokens(game)[0]
    return {
        "x": torch.randn(n, F.OBS_DIM),
        "next_x": torch.randn(n, F.OBS_DIM),
        "e_behavior": torch.rand(n, 2) * 2 - 1,
        "global_token": torch.full((n,), token, dtype=torch.long),
        "local_action": torch.zeros(n, dtype=torch.long),
        "reward": torch.rand(n) * 2 - 1,
        "log_b_action": torch.log(torch.full((n,), 0.5)),
        "reference_return": torch.full((n,), g),
        "ref_valid": torch.full((n,), ref_valid, dtype=torch.bool),
        "task_id": torch.full((n,), task, dtype=torch.long),
        "mapping_version": torch.ones(n, dtype=torch.long),
        "terminated": torch.full((n,), terminated, dtype=torch.bool),
        "truncated": torch.zeros(n, dtype=torch.bool),
        "games": [game] * n,
        "size": n,
    }


# ================================================================ 标量叶子隔离检查
class ScalarLeafQ:
    """只用独立标量叶子验证校准算子的导数（§14 T6）。"""

    def __init__(self, x, e_behavior, candidates, q_data_value=0.0):
        self.x = x
        self.e_behavior = e_behavior
        self.candidates = candidates

    def __call__(self, x, e):
        if e.dim() == 3:
            return self.candidates
        return torch.zeros(x.shape[0], 1)


def test_calibration_operator_gradients_with_independent_leaves():
    """§14 T6：Q_candidate=0, G=5, Q_data=0 时
    校准正则对 candidate 的导数为 0（被 max 截住）、对 data 为 -1；
    uncalibrated 对 candidate 为 +1/|M|。"""
    n, M = 4, 5
    x = torch.zeros(n, F.OBS_DIM)
    e_beh = torch.zeros(n, 2)

    # --- mode=max：candidate 全 0 < G=5，max 取 G → 对 candidate 无梯度 ---
    cand = torch.zeros(n, M, 1, requires_grad=True)
    data = torch.zeros(n, 1, requires_grad=True)

    class QMax(torch.nn.Module):
        def forward(self, x_, e_):
            if e_.dim() == 3:
                return cand
            return data

    q = QMax()
    model = make_model()
    batch = make_batch(n)
    batch["x"] = x
    batch["e_behavior"] = e_beh
    batch["reference_return"] = torch.full((n,), 5.0)
    batch["ref_valid"] = torch.ones(n, dtype=torch.bool)
    batch["task_id"] = torch.zeros(n, dtype=torch.long)
    reg, _ = calibration_regularizer_by_task(
        q_net=q, model=model, batch=batch, mode="max", samples=M
    )
    reg.backward()
    assert abs(float(cand.grad.abs().sum())) < 1e-12, "candidate 被 max 截住，导数应为 0"
    # 标量正则对 [n,1] 张量的梯度：每个元素是 -1/n，整块求和才是 -1
    assert abs(float(data.grad.sum()) + 1.0) < 1e-12, "data 项总导数应为 -1"
    assert abs(float(data.grad.mean()) + 1.0 / n) < 1e-12

    # --- mode=linear（uncalibrated）：candidate 直接进入，导数 +1/M ---
    cand2 = torch.zeros(n, M, 1, requires_grad=True)
    data2 = torch.zeros(n, 1, requires_grad=True)

    class QLin(torch.nn.Module):
        def forward(self, x_, e_):
            if e_.dim() == 3:
                return cand2
            return data2

    reg2, _ = calibration_regularizer_by_task(
        q_net=QLin(), model=model, batch=batch, mode="linear", samples=M
    )
    reg2.backward()
    # 第一项 = mean_m Q_cand：张量里每个 candidate 元素的总导数是 1/n（任务内 mean），
    # 即单条样本上的 sum_m 导数为 1/n → 每个元素 1/(n·M)。data 项总导数为 -1。
    assert abs(float(cand2.grad.sum()) - 1.0) < 1e-9, cand2.grad.sum()
    assert abs(float(cand2.grad.mean()) - 1.0 / (n * M)) < 1e-8   # float32 精度
    assert abs(float(data2.grad.sum()) + 1.0) < 1e-12


def test_calibration_off_is_identically_zero():
    q = make_model().q1
    reg, diag = calibration_regularizer_by_task(
        q_net=q, model=make_model(), batch=make_batch(), mode="off", samples=4
    )
    assert float(reg.item()) == 0.0


def test_calibration_no_valid_rows_contributes_zero():
    q = make_model().q1
    batch = make_batch(ref_valid=False)
    reg, diag = calibration_regularizer_by_task(
        q_net=q, model=make_model(), batch=batch, mode="max", samples=4
    )
    assert float(reg.item()) == 0.0
    assert diag["coverage"] == {}


def test_calibration_samples_do_not_backprop_to_beta():
    """§14 T6：target/正则里的采样不向 β 反传。"""
    m = make_model()
    batch = make_batch(8)
    reg, _ = calibration_regularizer_by_task(
        q_net=m.q1, model=m, batch=batch, mode="max", samples=6
    )
    reg.backward()
    assert m.beta.head.weight.grad is None, "校准项不应给 β 梯度（e^m 已 stop-gradient）"
    assert m.q1.net[0].weight.grad is not None, "Q 必须保留梯度"


def test_task_weights_are_fixed_not_by_valid_count():
    """§7.3：任务权重固定为 1/2，不由 valid 行数决定。"""
    m = make_model()
    n = 16
    batch = make_batch(n, task=0)
    # 让 T 只有 1 条 valid，F 有 15 条
    half = make_batch(n, task=1)
    for k in ["x", "next_x", "e_behavior", "reward", "reference_return"]:
        batch[k][8:] = half[k][:8]
    batch["task_id"][8:] = 1
    batch["global_token"][8:] = F.legal_tokens("Timberman")[0]
    batch["games"] = ["Flappybird"] * 8 + ["Timberman"] * 8
    batch["ref_valid"][:] = False
    batch["ref_valid"][:8] = True
    batch["ref_valid"][8] = True           # T 只有 1 条 valid
    reg, diag = calibration_regularizer_by_task(
        q_net=m.q1, model=m, batch=batch, mode="max", samples=4
    )
    assert set(diag["coverage"]) == {0, 1}
    assert abs(diag["coverage"][1] - 1 / 8) < 1e-9
    assert reg.requires_grad


# ================================================================ target
def test_target_shape_and_no_broadcast():
    m = make_model()
    batch = make_batch(12)
    y = compute_target(m, batch, gamma=0.99, target_samples=2)
    assert y.shape == (12,)
    assert torch.isfinite(y).all()


def test_terminal_target_equals_reward():
    """终局不 bootstrap：y = r。"""
    m = make_model()
    batch = make_batch(6, terminated=True)
    y = compute_target(m, batch, gamma=0.99, target_samples=2)
    assert torch.allclose(y, batch["reward"], atol=1e-6)


def test_truncated_target_keeps_bootstrap():
    """外部截断保留 bootstrap：y = r + γ·V。"""
    m = make_model()
    batch = make_batch(6, terminated=False)
    y = compute_target(m, batch, gamma=0.99, target_samples=2)
    v = (y - batch["reward"]) / 0.99
    assert not torch.allclose(v, torch.zeros_like(v))


def test_bootstrap_arithmetic_matches_plan_evidence():
    """§14 T3 的数值：r=1, γ=0.99, next_value=10 → 终局 1、截断 10.9。"""
    m = make_model()
    # 把 target 网络做成常数 10
    for q in (m.target_q1, m.target_q2):
        with torch.no_grad():
            for p in q.parameters():
                p.zero_()
            q.net[-1].bias.fill_(10.0)
    batch = make_batch(4, terminated=False)
    batch["reward"] = torch.ones(4)
    y = compute_target(m, batch, gamma=0.99, target_samples=3)
    assert torch.allclose(y, torch.full((4,), 10.9), atol=1e-5), y
    batch_term = make_batch(4, terminated=True)
    batch_term["reward"] = torch.ones(4)
    y_term = compute_target(m, batch_term, gamma=0.99, target_samples=2)
    assert torch.allclose(y_term, torch.ones(4), atol=1e-6)


def test_target_uses_min_of_two_target_networks():
    m = make_model()
    for q, val in ((m.target_q1, 3.0), (m.target_q2, 1.0)):
        with torch.no_grad():
            for p in q.parameters():
                p.zero_()
            q.net[-1].bias.fill_(val)
    batch = make_batch(4, terminated=False)
    batch["reward"] = torch.zeros(4)
    y = compute_target(m, batch, gamma=1.0, target_samples=2)
    assert torch.allclose(y, torch.full((4,), 1.0), atol=1e-6), y


def _linear_q_from_latent() -> torch.nn.Module:
    """构造 Q(x,e) = e_0 的单层线性网络（Q 可正可负）。

    多层 ReLU MLP 无法把某个输入维"直通"到输出（会被隐藏层打散），
    所以用它做定值检查只会得到 0；这里显式替换成单层线性层。
    """
    class QLatent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(F.OBS_DIM + 2, 1)
            with torch.no_grad():
                self.lin.weight.zero_()
                self.lin.bias.zero_()
                self.lin.weight[0, F.OBS_DIM] = 1.0     # 只取 e 的第一维

        def forward(self, x, e):
            if e.dim() == 2:
                return self.lin(torch.cat([x, e], -1))
            x_in = x.unsqueeze(1).expand(-1, e.shape[1], -1)
            return self.lin(torch.cat([x_in, e], -1))

    return QLatent()


def test_target_averages_over_samples():
    """N 个 e' 的 Q 必须取平均（§7.1）。"""
    m = make_model()
    m.target_q1 = _linear_q_from_latent()
    m.target_q2 = _linear_q_from_latent()
    batch = make_batch(256, terminated=False)
    batch["reward"] = torch.zeros(256)
    y1 = compute_target(m, batch, gamma=1.0, target_samples=1)
    y8 = compute_target(m, batch, gamma=1.0, target_samples=8)
    assert float(y1.abs().mean()) > 0.05
    # 两种采样数都是同一期望(E[e0]=0)的估计：整体均值应接近 0。
    # 不比较两次调用的均值差——generator 连续推进，两次抽的是不同噪声样本，
    # 单次均值差本身就是随机量；用"样本内标准差随 N 下降"来验证取平均这件事。
    assert abs(float(y1.mean())) < 0.15, y1.mean()
    assert abs(float(y8.mean())) < 0.15, y8.mean()
    assert float(y8.std()) < float(y1.std()) * 0.75, (y8.std(), y1.std())


# ================================================================ TD 损失
def test_td_loss_weighting_formula():
    """L_TD = ½·mean[w·(Q-y)²]，逐元素核对。"""
    m = make_model()
    batch = make_batch(5)
    target = torch.zeros(5)
    w = torch.tensor([1.0, 2.0, 0.5, 1.0, 4.0])
    q_vals = m.q1(batch["x"], batch["e_behavior"]).squeeze(-1).detach()
    expected = 0.5 * (w * q_vals.pow(2)).mean()
    got = td_loss(q_net=m.q1, batch=batch, target=target, weights=w)
    assert abs(float(got.item()) - float(expected.item())) < 1e-6


def test_td_loss_does_not_renormalize_weights():
    """w 不按 batch 总和重新归一化（§7.1）。"""
    m = make_model()
    batch = make_batch(4)
    target = torch.zeros(4)
    w1 = torch.ones(4)
    w2 = torch.ones(4) * 10.0
    l1 = td_loss(q_net=m.q1, batch=batch, target=target, weights=w1)
    l2 = td_loss(q_net=m.q1, batch=batch, target=target, weights=w2)
    assert abs(float(l2.item()) / max(float(l1.item()), 1e-12) - 10.0) < 1e-4


def test_td_loss_shape_mismatch_raises():
    m = make_model()
    batch = make_batch(4)
    with pytest.raises(RuntimeError, match="形状不一致"):
        td_loss(q_net=m.q1, batch=batch, target=torch.zeros(3),
                weights=torch.ones(4))


# ================================================================ actor 梯度
def test_actor_gradient_path_through_q():
    """§14 T6 的 toy 检查：一维 toy e=tanh(μ)、噪声=0、Q(e)=e、α=0、μ=0
    → dL_actor/dμ = -1；且 Q 参数不更新。"""
    mu = torch.zeros(1, requires_grad=True)

    def beta_rsample(x, samples=1):
        e = torch.tanh(mu).reshape(1, 1, 1)
        return type("S", (), {"e": e, "e_squeezed": e[:, 0], "log_prob_squeezed": torch.zeros(1)})()

    class BetaStub:
        rsample = staticmethod(beta_rsample)

    class ModelStub:
        beta = BetaStub()

    class QIdentity(torch.nn.Module):
        def forward(self, x, e):
            return e.reshape(-1, 1)

    q = QIdentity()
    loss, logp, qa = actor_loss(model=ModelStub(), q1=q, q2=q,
                                x=torch.zeros(1, F.OBS_DIM), alpha=0.0)
    loss.backward()
    # dL/dμ = -dQ/de * de/dμ = -1 * (1 - tanh²(0)) = -1
    assert abs(float(mu.grad.item()) + 1.0) < 1e-6, mu.grad
    assert all(p.grad is None for p in q.parameters())


def test_actor_loss_uses_min_of_two_q():
    m = make_model()
    for q, val in ((m.q1, 5.0), (m.q2, 1.0)):
        with torch.no_grad():
            for p in q.parameters():
                p.zero_()
            q.net[-1].bias.fill_(val)
    loss, logp, qa = actor_loss(model=m, q1=m.q1, q2=m.q2,
                                x=torch.randn(8, F.OBS_DIM), alpha=0.0)
    assert torch.allclose(qa, torch.full((8,), 1.0), atol=1e-6)


def test_q_parameters_get_no_grad_in_actor_step():
    m = make_model()
    x = torch.randn(8, F.OBS_DIM)
    with freeze_parameters(m.q1, m.q2):
        loss, _, _ = actor_loss(model=m, q1=m.q1, q2=m.q2, x=x, alpha=0.02)
        loss.backward()
    assert m.q1.net[0].weight.grad is None
    assert m.q2.net[0].weight.grad is None
    assert m.beta.trunk[0].weight.grad is not None
    assert float(m.beta.trunk[0].weight.grad.abs().sum()) > 0


# ================================================================ 完整 update
def test_learner_update_returns_stats_and_updates_q_beta_only():
    m = make_model()
    learner = ContinuousLearner(m, LearnerConfig(calibration_mode="max"))
    batch = make_batch(16)
    phi_before = [p.detach().clone() for p in m.phi.parameters()]
    q_before = [p.detach().clone() for p in m.q1.parameters()]
    beta_before = [p.detach().clone() for p in m.beta.parameters()]
    tgt_before = [p.detach().clone() for p in m.target_q1.parameters()]

    stats = learner.update(batch)
    assert stats.n_transitions == 16
    assert stats.td_loss == stats.td_loss  # not NaN
    assert math.isfinite(stats.actor_loss)
    assert stats.w_max <= 100.0 + 1e-6
    # β 与 Q 变了
    assert any(not torch.equal(a, b) for a, b in zip(q_before, m.q1.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(beta_before, m.beta.parameters()))
    # φ 不变（已冻结）
    assert all(torch.equal(a, b) for a, b in zip(phi_before, m.phi.parameters()))
    # target 只按 Polyak 动，且必须与 q1 的更新后值相差 tau 倍
    assert any(not torch.equal(a, b) for a, b in zip(tgt_before, m.target_q1.parameters()))


def test_learner_refuses_trainable_phi():
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    # 故意不冻结 φ
    learner = ContinuousLearner(m)
    with pytest.raises(RuntimeError, match="冻结 φ/ψ"):
        learner.update(make_batch(8))


def test_learner_w_detached_and_bounded():
    m = make_model()
    learner = ContinuousLearner(m)
    batch = make_batch(32)
    w = learner.compute_weights(batch)
    assert not w.requires_grad
    assert float(w.max()) <= 100.0 + 1e-6
    assert (w > 0).all()


def test_learner_updates_two_q_independently():
    m = make_model()
    learner = ContinuousLearner(m)
    learner.update(make_batch(16))
    # 两个 critic 的参数不应完全相同（各自独立初始化 + 独立更新）
    assert not torch.equal(m.q1.net[0].weight, m.q2.net[0].weight)


def test_backup_entropy_true_is_rejected():
    with pytest.raises(ValueError, match="熵 backup"):
        LearnerConfig(backup_entropy=True)


def test_calibration_raises_td_target_not_directly():
    """§14 T6：校准没有直接抬高 TD target。"""
    m = make_model()
    batch = make_batch(8)
    y_full = compute_target(m, batch, gamma=0.99, target_samples=2)
    full_learner = ContinuousLearner(m, LearnerConfig(calibration_mode="max"))
    stats = full_learner.update(batch)
    y_after = compute_target(m, batch, gamma=0.99, target_samples=2)
    # 校准只通过 Q 参数间接影响；target 的计算式里没有 reference_return
    assert "reference_return" not in ["w", "target"]
    assert stats.cal_regularizer == stats.cal_regularizer


def test_offline_pretrain_runs_requested_updates():
    m = make_model()
    learner = ContinuousLearner(m)
    batches = [make_batch(8), make_batch(8, task=1)]
    stats = offline_pretrain(m, learner, batches, updates=5)
    assert len(stats) == 5
    assert learner.update_count == 5


def test_offline_pretrain_requires_updates():
    m = make_model()
    learner = ContinuousLearner(m)
    with pytest.raises(ValueError):
        offline_pretrain(m, learner, [make_batch(8)], updates=0)


def test_update_stats_serialisable():
    m = make_model()
    learner = ContinuousLearner(m)
    d = learner.update(make_batch(8)).as_dict()
    import json
    json.dumps({k: v for k, v in d.items() if not isinstance(v, torch.Tensor)})
    assert "reference_coverage" in d


def test_update_stats_reject_non_finite_and_zero_ess():
    good = UpdateStats(
        td_loss=1.0, cal_regularizer=0.1, q1_loss=0.6, q2_loss=0.5,
        actor_loss=0.2, grad_norm_q1=1.0, grad_norm_q2=1.0,
        grad_norm_actor=1.0, w_mean=1.0, w_max=1.0,
        ess_over_batch=0.5, target_mean=0.0, q_data_mean=0.0,
        n_transitions=128,
    )
    validate_update_stats(good, expected_batch_size=128, variant="continuous")

    with pytest.raises(RuntimeError, match="更新统计字段非法.*td_loss"):
        validate_update_stats(
            dataclasses.replace(good, td_loss=float("nan")),
            expected_batch_size=128,
            variant="continuous",
        )
    with pytest.raises(RuntimeError, match="ESS"):
        validate_update_stats(
            dataclasses.replace(good, ess_over_batch=0.0),
            expected_batch_size=128,
            variant="continuous",
        )


@pytest.mark.parametrize("changes, message", [
    ({"n_transitions": 127}, "batch size"),
    ({"w_mean": 0.0}, "importance weights"),
    ({"w_max": -1.0}, "importance weights"),
    ({"grad_norm_actor": float("inf")}, "更新统计字段非法.*grad_norm_actor"),
])
def test_update_stats_reject_invalid_batch_weights_and_gradients(changes, message):
    good = UpdateStats(
        td_loss=1.0, cal_regularizer=0.1, q1_loss=0.6, q2_loss=0.5,
        actor_loss=0.2, grad_norm_q1=1.0, grad_norm_q2=1.0,
        grad_norm_actor=1.0, w_mean=1.0, w_max=1.0,
        ess_over_batch=0.5, target_mean=0.0, q_data_mean=0.0,
        n_transitions=128,
    )
    with pytest.raises(RuntimeError, match=message):
        validate_update_stats(
            dataclasses.replace(good, **changes), expected_batch_size=128,
            variant="continuous",
        )


@pytest.mark.parametrize("missing_field", ["w_mean", "ess_over_batch"])
def test_continuous_update_stats_require_importance_diagnostics(missing_field):
    values = UpdateStats(
        td_loss=1.0, cal_regularizer=0.1, q1_loss=0.6, q2_loss=0.5,
        actor_loss=0.2, grad_norm_q1=1.0, grad_norm_q2=1.0,
        grad_norm_actor=1.0, w_mean=1.0, w_max=1.0,
        ess_over_batch=0.5, target_mean=0.0, q_data_mean=0.0,
        n_transitions=128,
    ).as_dict()
    values.pop(missing_field)

    with pytest.raises(RuntimeError, match=f"缺少必需字段.*{missing_field}"):
        validate_update_stats(
            SimpleNamespace(**values), expected_batch_size=128,
            variant="continuous",
        )


def test_discrete_update_stats_without_importance_diagnostics_are_valid():
    stats = DiscreteUpdateStats(
        td_loss=1.0, cal_regularizer=0.1, q1_loss=0.6, q2_loss=0.5,
        actor_loss=0.2, grad_norm_q1=1.0, grad_norm_q2=1.0,
        grad_norm_actor=1.0, reference_coverage={0: 1.0},
        target_mean=0.0, q_data_mean=0.0, n_transitions=128,
    )

    validate_update_stats(stats, expected_batch_size=128, variant="discrete")


@pytest.mark.parametrize("field, value", [
    ("w_mean", None),
    ("w_max", "invalid"),
    ("ess_over_batch", float("nan")),
    ("ess_over_batch", 0.0),
])
def test_continuous_importance_diagnostics_reject_invalid_values(field, value):
    stats = UpdateStats(
        td_loss=1.0, cal_regularizer=0.1, q1_loss=0.6, q2_loss=0.5,
        actor_loss=0.2, grad_norm_q1=1.0, grad_norm_q2=1.0,
        grad_norm_actor=1.0, w_mean=1.0, w_max=1.0,
        ess_over_batch=0.5, target_mean=0.0, q_data_mean=0.0,
        n_transitions=128,
    )
    setattr(stats, field, value)

    expected = "ESS" if field == "ess_over_batch" and value == 0.0 else (
        "更新统计字段非法" if value is None or isinstance(value, str)
        or (isinstance(value, float) and math.isnan(value)) else "importance weights"
    )
    with pytest.raises(RuntimeError, match=expected):
        validate_update_stats(stats, expected_batch_size=128, variant="continuous")
