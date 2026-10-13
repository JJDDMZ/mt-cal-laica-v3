"""T4 验收：连续网络、稳定 tanh log_prob、K=2→4 扩展与 optimizer 状态继承（§6，§14 T4）。"""

from __future__ import annotations

import math

import pytest
import torch

from wemi_cal.envs import features as F
from wemi_cal.models.continuous import (
    Beta,
    ContinuousModel,
    Phi,
    QNet,
    Sample,
    count_parameters,
    freeze_parameters,
    gaussian_kl_to_standard,
    polyak_update,
    tanh_log_det_jacobian,
    tanh_squash_log_prob,
)


def make_model(seed=0) -> ContinuousModel:
    m = ContinuousModel(seed=seed)
    m.register_game("Flappybird")
    return m


# ---------------------------------------------------------------- 稳定 tanh log_prob
def test_tanh_log_prob_finite_at_extreme_u():
    """§14 T4：u=±20 时 log_prob 必须有限（不能是 -inf/NaN）。"""
    for u_val in (-20.0, 20.0, -50.0, 50.0, 0.0):
        u = torch.full((4, 2), u_val)
        mean = torch.zeros(4, 2)
        log_std = torch.zeros(4, 2)
        lp = tanh_squash_log_prob(u, mean, log_std)
        assert torch.isfinite(lp).all(), f"u={u_val} → {lp}"
    # 大 |u| 时雅可比项应当趋于 -2|u|（而不是 -inf）
    u = torch.tensor([[20.0, -20.0]])
    jac = tanh_log_det_jacobian(u)
    assert torch.isfinite(jac).all()
    assert (jac < -35).all(), jac  # ≈ 2*(log2 - 20)


def test_tanh_log_prob_matches_naive_formula_in_stable_range():
    """在 |u| 不大时，稳定形式应与朴素 log(1-tanh²u) 一致，保证没有写错公式。"""
    torch.manual_seed(0)
    u = torch.randn(64, 2) * 0.8
    stable = tanh_log_det_jacobian(u)
    naive = torch.log(1.0 - torch.tanh(u) ** 2)
    assert torch.allclose(stable, naive, atol=1e-5)


def test_naive_formula_would_overflow_but_stable_does_not():
    u = torch.tensor([[30.0, 30.0]])
    naive = torch.log(1.0 - torch.tanh(u) ** 2)
    assert not torch.isfinite(naive).all()          # 朴素写法失效
    assert torch.isfinite(tanh_log_det_jacobian(u)).all()


def test_kl_zero_at_standard_normal():
    """§14 T4：μψ=0、σψ=1 时 KL=0。"""
    mean = torch.zeros(5, 2)
    log_std = torch.zeros(5, 2)
    kl = gaussian_kl_to_standard(mean, log_std)
    assert torch.allclose(kl, torch.zeros(5), atol=1e-12)


def test_kl_formula_matches_analytic():
    mean = torch.tensor([[1.0, -2.0]])
    log_std = torch.tensor([[0.5, -0.5]])
    expected = 0.5 * ((1.0 + math.exp(1.0) - 1 - 1.0) + (4.0 + math.exp(-1.0) - 1 + 1.0))
    got = gaussian_kl_to_standard(mean, log_std).item()
    assert abs(got - expected) < 1e-6


# ---------------------------------------------------------------- 形状契约
def test_rsample_shapes_multi_sample():
    """§13：beta.rsample(x, samples=N) → e [B,N,2]、logp [B,N]。"""
    m = make_model()
    x = torch.randn(7, F.OBS_DIM)
    s = m.beta.rsample(x, samples=2)
    assert s.e.shape == (7, 2, 2)
    assert s.log_prob.shape == (7, 2)
    assert torch.isfinite(s.log_prob).all()
    # e 必须在 (-1,1)
    assert (s.e.abs() < 1.0).all()


def test_single_sample_squeeze_is_explicit():
    """N=1 必须显式 squeeze，禁止依赖广播生成 [B,B]。"""
    m = make_model()
    x = torch.randn(5, F.OBS_DIM)
    s = m.beta.rsample(x, samples=1)
    assert s.e.shape == (5, 1, 2)
    assert s.e_squeezed.shape == (5, 2)
    assert s.log_prob_squeezed.shape == (5,)
    with pytest.raises(ValueError, match="只有 N=1"):
        m.beta.rsample(x, samples=2).e_squeezed


def test_q_net_shapes_no_broadcast():
    """Q 输入 [B,N,latent] → [B,N,1]；min 后 [B,N,1]；对 N 取平均再 squeeze 得 [B]。

    注意：``Q`` 的最后一维是标量维（长度 1），所以 ``mean(-1)`` 只会在标量维上取平均
    （形状不变），对 N 个样本取平均必须用 ``mean(-2)`` 或 ``mean(1)``。
    """
    m = make_model()
    x = torch.randn(6, F.OBS_DIM)
    e = m.beta.rsample(x, samples=2).e
    q1 = m.q1(x, e)
    q2 = m.q2(x, e)
    assert q1.shape == (6, 2, 1)
    qmin = torch.minimum(q1, q2)
    assert qmin.shape == (6, 2, 1)
    assert qmin.mean(-1).shape == (6, 2)             # 只压掉标量维
    assert qmin.mean(1).squeeze(-1).shape == (6,)    # 对 N 平均 → [B]
    # 方案伪代码写的是 mean(-1)：因为标量维长度是 1，这一步数值上等价于恒等，
    # 必须显式对 N 取平均（mean(dim=1)）才能得到 [B]；learner 里按此实现。
    assert qmin.mean(1).squeeze(-1).shape == (6,)
    # mean(-1) 只压标量维：数值上等于"某个样本的 Q"而不是 N 个样本的平均，
    # 两者形状不同（[6,2] vs [6]），因此不能混用。
    assert qmin.mean(-1).shape == (6, 2)
    # 绝不能出现 [B,B] 广播
    assert qmin.shape[0] == 6 and qmin.shape[1] == 2
    # 单样本路径
    e1 = m.beta.rsample(x, samples=1).e_squeezed
    assert m.q1(x, e1).shape == (6, 1)
    assert torch.minimum(m.q1(x, e1), m.q2(x, e1)).squeeze(-1).shape == (6,)


def test_q_net_rejects_bad_shape():
    m = make_model()
    x = torch.randn(4, F.OBS_DIM)
    with pytest.raises(ValueError, match="必须是"):
        m.q1(x, torch.randn(4, 3, 3, 2))


def test_multisample_shapes_align_with_actor_path():
    """actor 路径的 e 是 [B,2]，logp 是 [B]。"""
    m = make_model()
    x = torch.randn(8, F.OBS_DIM)
    s = m.beta.rsample(x, samples=1)
    e = s.e_squeezed
    logp = s.log_prob_squeezed
    assert e.shape == (8, 2) and logp.shape == (8,)
    q = torch.minimum(m.q1(x, e), m.q2(x, e)).squeeze(-1)
    assert q.shape == (8,)
    loss = (0.02 * logp - q).mean()
    assert loss.dim() == 0


# ---------------------------------------------------------------- φ 掩码与概率
def test_phi_masks_illegal_tokens():
    """§13：只在该游戏合法动作内归一化；非法项 -inf、合法 log 概率有限。"""
    m = make_model()
    e = torch.randn(5, 2)
    logp = m.phi.log_probs(e, "Flappybird")
    assert logp.shape == (5, m.phi.total_tokens)
    legal = torch.tensor(F.legal_tokens("Flappybird"))
    assert torch.isfinite(logp[:, legal]).all()
    assert m.phi.num_tokens == 2
    assert set(m.phi.registered) == {0, 1}
    # 合法两项的指数和为 1
    probs = m.phi.probs(e, "Flappybird")[:, legal]
    assert torch.allclose(probs.sum(-1), torch.ones(5), atol=1e-5)


def test_illegal_tokens_are_minus_inf_after_registration():
    """注册 T 之后：F 的不合法槽位必须是真 -inf（§13），且有限变体可用。"""
    m = make_model()
    m.register_game("Timberman")
    e = torch.randn(6, 2)
    logp = m.phi.log_probs(e, "Flappybird")
    assert logp.shape == (6, 4)
    t_illegal = [2, 3]
    assert torch.isinf(logp[:, t_illegal]).all() and (logp[:, t_illegal] < 0).all()
    f_illegal = m.phi.finite_log_probs(e, "Flappybird")[:, t_illegal]
    assert torch.isfinite(f_illegal).all() and (f_illegal < -1e20).all()


def test_phi_illegal_tokens_never_sampled():
    """非法 token 的混合概率必须被压到 log=-inf，采样永不选中。"""
    m = make_model()
    m.register_game("Timberman")
    e = torch.randn(200, 2)
    logb = m.phi.behavior_log_probs(e, "Flappybird", uniform_mix=0.02)
    assert torch.isinf(logb[:, [2, 3]]).all()
    tok = m.phi.sample_token(e, "Flappybird", uniform_mix=0.02)
    assert set(tok.tolist()) <= {0, 1}


def test_behavior_mix_matches_formula():
    """b = 0.98*φ + 0.02/|A|，逐元素核对。"""
    m = make_model()
    e = torch.randn(3, 2)
    phi = m.phi.probs(e, "Flappybird")
    b = torch.exp(m.phi.behavior_log_probs(e, "Flappybird", uniform_mix=0.02))
    expected = 0.98 * phi + 0.02 / 2
    assert torch.allclose(b[:, :2], expected[:, :2], atol=1e-6)


def test_behavior_log_prob_of_records_mixed_distribution():
    """log_b_action 必须来自混合分布 b，而不是 φ 的概率（§6.2）。"""
    m = make_model()
    e = torch.randn(4, 2)
    tok = torch.tensor([0, 1, 0, 1])
    got = m.phi.behavior_log_prob_of(e, "Flappybird", tok, uniform_mix=0.02)
    logb = m.phi.behavior_log_probs(e, "Flappybird", uniform_mix=0.02)
    expected = logb[torch.arange(4), tok]
    assert torch.allclose(got, expected, atol=1e-6), (got, expected)
    # 与纯 φ 的 log 概率不同（探索混合把概率往 0.5 拉）
    pure = m.phi.log_probs(e, "Flappybird")[torch.arange(4), tok]
    assert not torch.allclose(got, pure, atol=1e-9)
    # 数值上等于 log(0.98*φ + 0.02/2)
    phi = m.phi.probs(e, "Flappybird")[torch.arange(4), tok]
    assert torch.allclose(got, torch.log(0.98 * phi + 0.02 / 2), atol=1e-6)


def test_behavior_log_prob_rejects_illegal_token_record():
    m = make_model()
    m.register_game("Timberman")
    e = torch.randn(2, 2)
    # 在 Flappybird 上请求 Timberman 的 token(2/3) → 非法
    with pytest.raises(ValueError, match="非法 token"):
        m.phi.behavior_log_prob_of(e, "Flappybird", torch.tensor([2, 3]), uniform_mix=0.02)
    # 未注册时索引越界也必须报错，而不是静默
    m2 = make_model()
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        m2.phi.behavior_log_prob_of(e, "Flappybird", torch.tensor([2, 3]), uniform_mix=0.02)


def test_greedy_ties_pick_smallest_token():
    """§12.1：并列时取最小 token。"""
    m = make_model()
    with torch.no_grad():
        m.phi.linear.weight.zero_()
        m.phi.linear.bias.zero_()
    e = torch.zeros(3, 2)
    tok = m.phi.greedy_token(e, "Flappybird")
    assert tok.tolist() == [0, 0, 0]


def test_deterministic_e_is_tanh_of_mean():
    m = make_model()
    x = torch.randn(4, F.OBS_DIM)
    mean, _ = m.beta.distribution_params(x)
    assert torch.allclose(m.beta.deterministic_e(x), torch.tanh(mean))


# ---------------------------------------------------------------- K=2 → K=4 扩展
def test_expansion_keeps_old_probabilities_identical():
    """§14 T4：扩展前后同一 e 的 F 概率完全一致。"""
    m = make_model()
    e = torch.randn(16, 2)
    legal_before = torch.tensor(F.legal_tokens("Flappybird"))
    before = m.phi.probs(e, "Flappybird")[:, legal_before]
    m.register_game("Timberman")
    after = m.phi.probs(e, "Flappybird")[:, legal_before]
    assert m.phi.num_tokens == 4
    assert torch.allclose(before, after, atol=1e-12), (before - after).abs().max()


def test_expansion_copies_weights_and_bias():
    """固定 token 空间：输出层形状不变，注册只给新行赋值，旧行逐位不变。"""
    m = make_model()
    reg_before = list(m.phi.registered)
    w_before = m.phi.linear.weight.detach().clone()
    b_before = m.phi.linear.bias.detach().clone()
    assert m.phi.linear.weight.shape == (4, 2)      # 一开始就是完整 token 空间
    m.register_game("Timberman")
    assert m.phi.linear.weight.shape == (4, 2)
    for tok in reg_before:
        assert torch.equal(m.phi.linear.weight[tok], w_before[tok])
        assert torch.equal(m.phi.linear.bias[tok], b_before[tok])
    # 新注册的行被真正初始化（不再是零）
    assert not torch.allclose(m.phi.linear.weight[2], torch.zeros(2))


def test_expansion_new_rows_random_old_rows_untouched_by_init():
    """新行随机初始化，且不应改变旧行。"""
    m = make_model()
    m.register_game("Timberman")
    new_w = m.phi.linear.weight[2:]
    assert not torch.allclose(new_w, torch.zeros_like(new_w))   # 随机初始化
    assert torch.equal(m.phi.linear.bias[2:], torch.zeros(2))


def test_expansion_requires_global_token_order():
    m = make_model()
    with pytest.raises(ValueError, match="按全局顺序追加"):
        m.phi.register_actions([(3, "Timberman", 1)])
    with pytest.raises(ValueError, match="按全局顺序追加"):
        m.phi.register_actions([(1, "Flappybird", 1)])


def test_expansion_rejects_wrong_local_action():
    m = make_model()
    with pytest.raises(ValueError, match="本地动作"):
        m.phi.register_actions([(2, "Timberman", 1)])


def test_first_token_can_be_timberman_st():
    """§14 T4：ST-T 首组全局 token 可以是 T 的 2/3，不依赖 token 等于输出行号。"""
    m = ContinuousModel(seed=0)
    m.register_game_as_first("Timberman")
    assert m.phi.tokens == (2, 3)
    # 行号 ≠ token 号：输出行 0 对应 token 2
    assert m.phi.greedy_token(torch.zeros(1, 2), "Timberman").item() in (2, 3)
    # 未注册的 token 行不参与该游戏的概率归一化
    legal = m.phi.registered_only(m.phi.probs(torch.zeros(1, 2), "Timberman"), "Timberman")
    assert torch.allclose(legal.sum(-1), torch.ones(1), atol=1e-6)
    e = torch.randn(4, 2)
    legal = m.phi.registered_only(m.phi.log_probs(e, "Timberman"), "Timberman")
    assert torch.isfinite(legal).all()
    full = m.phi.log_probs(e, "Timberman")
    assert torch.isinf(full[:, [0, 1]]).all()      # 未注册的 F 列不参与归一化
    tok = m.phi.greedy_token(e, "Timberman")
    assert set(tok.tolist()) <= {2, 3}


def test_duplicate_registration_rejected():
    m = make_model()
    with pytest.raises(ValueError, match="已经注册过"):
        m.register_game("Flappybird")


def test_expansion_does_not_change_beta_psi_q_shapes():
    m = make_model()
    h_before = {n: p.shape for n, p in m.named_parameters() if not n.startswith("phi")}
    m.register_game("Timberman")
    h_after = {n: p.shape for n, p in m.named_parameters() if not n.startswith("phi")}
    assert h_before == h_after
    assert m.beta.trunk[0].in_features == F.OBS_DIM
    assert m.q1.net[0].weight.shape[1] == F.OBS_DIM + 2


def test_optimizer_state_row_inheritance():
    """§14 T4：旧行 Adam 矩继承、新行矩为 0（固定 token 空间下按行对齐）。"""
    from wemi_cal.repr_train import expand_decoder_optimizer

    m = make_model()
    opt = torch.optim.Adam(m.phi.parameters(), lr=1e-3)
    x = torch.randn(8, F.OBS_DIM)
    lp = m.phi.log_probs(m.beta.rsample(x, 1).e_squeezed, "Flappybird")
    loss = -lp[:, 0].mean()
    loss.backward()
    opt.step()
    opt.zero_grad()

    exp_before = opt.state[m.phi.linear.weight]["exp_avg"].detach().clone()
    assert exp_before.shape == (4, 2)
    assert exp_before[:2].abs().sum() > 0

    old_param = m.phi.linear.weight
    old_bias = m.phi.linear.bias
    m.register_game("Timberman")
    # 固定 token 空间下参数对象本身不变，因此 optimizer 状态天然保留；
    # 仍需显式调用以保证新注册行的矩被置零。
    expand_decoder_optimizer(opt, m.phi, [old_param, old_bias])

    new_state = opt.state[m.phi.linear.weight]
    assert new_state["exp_avg"].shape == (4, 2)
    assert torch.equal(new_state["exp_avg"][:2], exp_before[:2])      # 旧行继承
    assert torch.equal(new_state["exp_avg"][2:], torch.zeros(2, 2))   # 新行置零
    assert torch.equal(new_state["exp_avg_sq"][2:], torch.zeros(2, 2))
    # step 计数保留，不重置整个优化器
    assert new_state["step"].item() == opt.state[old_bias].get("step", new_state["step"]).item()


def test_optimizer_expansion_without_prior_step():
    """没有旧矩时也要能扩展（新行全零）。"""
    from wemi_cal.repr_train import expand_decoder_optimizer

    m = make_model()
    opt = torch.optim.Adam(m.phi.parameters(), lr=1e-3)
    old = [m.phi.linear.weight, m.phi.linear.bias]
    m.register_game("Timberman")
    expand_decoder_optimizer(opt, m.phi, old)
    st = opt.state[m.phi.linear.weight]
    assert st["exp_avg"].shape == (4, 2)
    # 从未 step 过：没有任何历史矩，全部为零（新注册行也必须是零）
    assert torch.equal(st["exp_avg"], torch.zeros(4, 2))


# ---------------------------------------------------------------- 梯度契约
def test_actor_gradient_flows_through_q_to_e():
    """§14 T4：actor 能经 Q→e 反传；Q 参数不更新。"""
    m = make_model()
    x = torch.randn(6, F.OBS_DIM)
    s = m.beta.rsample(x, samples=1)
    e = s.e_squeezed
    logp = s.log_prob_squeezed
    with freeze_parameters(m.q1, m.q2):          # §7.5：critic 参数在该步冻结
        q = torch.minimum(m.q1(x, e), m.q2(x, e)).squeeze(-1)
        loss = (0.02 * logp - q).mean()
    loss.backward()
    g = m.beta.head.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    # critic 参数被冻结 → 不积累梯度；但 Q→e 的路径仍然有效（见上一个测试）
    assert m.q1.net[0].weight.grad is None
    assert m.q2.net[0].weight.grad is None
    assert m.phi.linear.weight.grad is None


def test_freeze_parameters_blocks_grad_but_keeps_q_grad_to_e():
    """freeze_parameters != no_grad：Q 参数被冻结，但 Q 对 e 的导数保留。"""
    m = make_model()
    x = torch.randn(5, F.OBS_DIM)
    e = m.beta.rsample(x, 1).e_squeezed.clone().detach().requires_grad_(True)
    with freeze_parameters(m.q1, m.q2):
        q = m.q1(x, e).sum()
        q.backward()
    assert e.grad is not None and e.grad.abs().sum() > 0     # Q→e 的梯度还在
    assert m.q1.net[0].weight.grad is None                   # Q 参数没攒梯度
    assert all(p.requires_grad for p in m.q1.parameters())    # 退出后恢复


def test_two_q_are_independent():
    m = make_model()
    assert m.q1 is not m.q2
    w1 = m.q1.net[0].weight.detach().clone()
    with torch.no_grad():
        m.q2.net[0].weight.add_(1.0)
    assert torch.equal(m.q1.net[0].weight, w1)


def test_targets_only_change_by_polyak():
    m = make_model()
    t_before = m.target_q1.net[0].weight.detach().clone()
    with torch.no_grad():
        m.q1.net[0].weight.add_(5.0)
    # 未调用 polyak 时 target 不变
    assert torch.equal(m.target_q1.net[0].weight, t_before)
    polyak_update(m.q1, m.target_q1, tau=0.005)
    expected = t_before * 0.995 + (t_before + 5.0) * 0.005
    assert torch.allclose(m.target_q1.net[0].weight, expected, atol=1e-6)


def test_hard_update_targets_copies_weights():
    m = make_model()
    with torch.no_grad():
        m.q1.net[0].weight.add_(3.0)
    m.hard_update_targets()
    assert torch.equal(m.target_q1.net[0].weight, m.q1.net[0].weight)


def test_psi_rsample_shapes_and_kl_zero_region():
    m = make_model()
    x = torch.randn(9, F.OBS_DIM)
    nx = torch.randn(9, F.OBS_DIM)
    s = m.psi.rsample(x, nx, samples=1)
    assert s.e.shape == (9, 1, 2)
    mean, log_std = m.psi.distribution_params(x, nx)
    kl = gaussian_kl_to_standard(mean, log_std)
    assert kl.shape == (9,) and (kl >= -1e-6).all()


def test_psi_does_not_read_action_or_reward():
    """ψ 的输入只有 [x,x']，签名里不存在 j/reward。"""
    m = make_model()
    x = torch.randn(3, F.OBS_DIM)
    nx = torch.randn(3, F.OBS_DIM)
    mu1, _ = m.psi.distribution_params(x, nx)
    mu2, _ = m.psi.distribution_params(x, nx)
    assert torch.equal(mu1, mu2)
    assert m.psi.trunk[0].in_features == 2 * F.OBS_DIM


def test_repr_loss_backward_updates_phi_and_psi():
    """L_repr = -log φ(j|e_ψ,g) + λKL；应对 φ/ψ 产生梯度，对 β/Q 无。"""
    m = make_model()
    x = torch.randn(12, F.OBS_DIM)
    nx = torch.randn(12, F.OBS_DIM)
    token = torch.tensor([0, 1] * 6)
    s = m.psi.rsample(x, nx, samples=1)
    e_psi = s.e_squeezed
    logp = m.phi.log_probs(e_psi, "Flappybird")[torch.arange(12), token]
    mean, log_std = m.psi.distribution_params(x, nx)
    kl = gaussian_kl_to_standard(mean, log_std)
    loss = (-logp + 1e-3 * kl).mean()
    loss.backward()
    assert m.phi.linear.weight.grad is not None and m.phi.linear.weight.grad.abs().sum() > 0
    assert m.psi.head.weight.grad is not None and m.psi.head.weight.grad.abs().sum() > 0
    assert m.beta.head.weight.grad is None
    assert m.q1.net[0].weight.grad is None


def test_coverage_grid_reports_all_actions():
    m = make_model()
    rep = m.phi.coverage_grid("Flappybird", res=11)
    assert rep["grid"].shape == (11, 11)
    assert set(rep["counts"]) == {0, 1}
    assert isinstance(rep["all_actions_reachable"], bool)


def test_parameter_hash_changes_with_params():
    m = make_model()
    h1 = m.parameter_hash()
    with torch.no_grad():
        m.beta.head.bias.add_(1e-3)
    assert m.parameter_hash() != h1
    assert m.subset_hash(["beta."]) != m.subset_hash(["q1."])


def test_sampling_is_seedable_and_deterministic_mode_is_reproducible():
    """同一 generator 种子可复现；explore=False 时 e=tanh(μ) 完全确定。"""
    # 同 seed 构造的模型参数必须完全一致（否则"可重放"在模型层面就不成立）
    a, b = Beta(seed=7), Beta(seed=7)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    x = torch.randn(4, F.OBS_DIM)
    a.gen.manual_seed(123)
    b.gen.manual_seed(123)
    assert torch.equal(a.rsample(x, 3).e, b.rsample(x, 3).e)
    # 同一模型上，把 generator 复位后必须重现同一批噪声
    a.gen.manual_seed(123)
    first = a.rsample(x, 3).e
    a.gen.manual_seed(123)
    assert torch.equal(first, a.rsample(x, 3).e)
    # 不同种子 → 不同噪声
    b.gen.manual_seed(124)
    assert not torch.equal(a.rsample(x, 3).e, b.rsample(x, 3).e)
    # explore=False 与 μ 路径一致，多次调用结果相同
    a.gen.manual_seed(1)
    d1 = a.rsample(x, 2, explore=False).e
    a.gen.manual_seed(999)
    d2 = a.rsample(x, 2, explore=False).e
    assert torch.equal(d1, d2)
    assert torch.allclose(d1[:, 0], torch.tanh(a.distribution_params(x)[0]))


def test_log_std_clamped():
    m = make_model()
    with torch.no_grad():
        m.beta.head.weight.mul_(100.0)
    x = torch.randn(8, F.OBS_DIM) * 5
    _, log_std = m.beta.distribution_params(x)
    assert log_std.max() <= 2.0 + 1e-6
    assert log_std.min() >= -5.0 - 1e-6


def test_parameter_count_is_small():
    """§11.3：CPU 两层 64 宽 MLP、二维 latent → 参数量应很小。"""
    m = make_model()
    n = count_parameters(m)
    assert n < 60_000, n


def test_log_prob_of_given_e_matches_rsample_construction():
    """log_prob(x,e) 与用同一 u 手工算的密度一致。"""
    m = make_model()
    x = torch.randn(3, F.OBS_DIM)
    mean, log_std = m.beta.distribution_params(x)
    e = torch.tanh(mean + 0.3)
    u = torch.atanh(e.clamp(-1 + 1e-7, 1 - 1e-7))
    manual = tanh_squash_log_prob(u, mean, log_std)
    got = m.beta.log_prob(x, e)
    assert torch.allclose(manual, got, atol=1e-5)
