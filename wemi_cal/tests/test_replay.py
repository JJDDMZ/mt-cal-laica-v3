"""T5 验收：事实字段、episode/MC、容量淘汰、采样比例与概率修正（§7.2/§8.2/§8.4/§8.5）。"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from wemi_cal import correction as COR
from wemi_cal.envs import features as F
from wemi_cal.models.continuous import ContinuousModel
from wemi_cal.replay import Episode, ReplayBuffer, ReplayError, Transition, make_transition


def obs(v: float = 0.0) -> np.ndarray:
    a = np.zeros(F.OBS_DIM, dtype=np.float32)
    a[0] = v
    return a


def make_buf(**kw) -> ReplayBuffer:
    return ReplayBuffer(gamma=0.99, batch_size=128, **kw)


def approx_phi(e_value: float, probs=(0.2, 0.8), total_tokens=4) -> ContinuousModel:
    """构造一个 φ 恰好给出指定两动作概率的模型（用解析反推 logits）。"""
    m = ContinuousModel(seed=0, latent_dim=2)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    # 令 logits[0] - logits[1] = log(p0/p1)，通过 bias 的两行差实现；权重置零使 e 不影响
    with torch.no_grad():
        m.phi.linear.weight.zero_()
        m.phi.linear.bias.zero_()
        m.phi.linear.bias[0] = math.log(probs[0])
        m.phi.linear.bias[1] = math.log(probs[1])
    return m


# ================================================================ §8.5 枚举单元测试
def test_enumerate_8_5_expected_values():
    """§8.5：b_old=[0.9,0.1]、φ=[0.2,0.8]、奖励=[1,3]
    → 不修正 1.2；加权后 2.6；按新映射直接算 2.6。"""
    r = COR.enumerate_correction_check()
    assert abs(r["naive_expectation"] - 1.2) < 1e-12
    assert abs(r["corrected_expectation"] - 2.6) < 1e-12
    assert abs(r["direct_new_mapping_expectation"] - 2.6) < 1e-12
    assert abs(r["weights"][0] - 0.2 / 0.9) < 1e-12
    assert abs(r["weights"][1] - 0.8 / 0.1) < 1e-12


def test_mapping_correction_equals_exact_ratio():
    """用真实 φ 复现同一组比值：应为 [0.2/0.9, 0.8/0.1]。"""
    m = approx_phi(0.0, probs=(0.2, 0.8))
    e = torch.zeros(2, 2)
    token = torch.tensor([0, 1])
    log_b = torch.log(torch.tensor([0.9, 0.1]))
    w = COR.mapping_correction(
        phi=m.phi, x=torch.zeros(2, F.OBS_DIM), e_behavior=e,
        token=token, log_b_action=log_b, game="Flappybird",
    )
    assert torch.allclose(w, torch.tensor([0.2 / 0.9, 0.8 / 0.1]), atol=1e-6), w


def test_correction_reproduces_target_mixture():
    """加权后的动作混合必须等于 φ 的混合（§8.2 的等式，用枚举验证）。"""
    b_old = np.array([0.9, 0.1])
    rew = np.array([1.0, 3.0])
    phi = np.array([0.2, 0.8])
    # 模拟采样：枚举每个动作，用 w 加权并乘以 b_old 的测度
    total = 0.0
    for j in range(2):
        w_j = phi[j] / b_old[j]
        total += b_old[j] * w_j * rew[j]
    assert abs(total - float((phi * rew).sum())) < 1e-12


def test_uniform_warmup_ratio_is_phi_over_half():
    """uniform 热身 → φ：b=0.5，w=2φ，上界为 2。"""
    m = approx_phi(0.0, probs=(0.2, 0.8))
    e = torch.zeros(2, 2)
    log_b = torch.log(torch.tensor([0.5, 0.5]))
    w = COR.mapping_correction(
        phi=m.phi, x=torch.zeros(2, F.OBS_DIM), e_behavior=e,
        token=torch.tensor([0, 1]), log_b_action=log_b, game="Flappybird",
    )
    assert torch.allclose(w, torch.tensor([0.4, 1.6]), atol=1e-6)
    assert float(w.max()) <= COR.uniform_behavior_weight_bound() + 1e-9


def test_exploration_mix_ratio_bound():
    """探索混合 b=0.98φ+0.01 → φ 的比值上界不超过 100。"""
    m = approx_phi(0.0, probs=(0.2, 0.8))
    e = torch.zeros(2, 2)
    phi_p = torch.tensor([0.2, 0.8])
    b = 0.98 * phi_p + 0.02 / 2
    w = COR.mapping_correction(
        phi=m.phi, x=torch.zeros(2, F.OBS_DIM), e_behavior=e,
        token=torch.tensor([0, 1]), log_b_action=torch.log(b), game="Flappybird",
    )
    assert float(w.max()) <= COR.MAX_WEIGHT + 1e-6


def test_correction_weight_is_detached():
    """§14 T5：w 必须 detach，不能向 φ 反传。"""
    m = approx_phi(0.0)
    e = torch.zeros(3, 2)
    w = COR.mapping_correction(
        phi=m.phi, x=torch.zeros(3, F.OBS_DIM), e_behavior=e,
        token=torch.tensor([0, 1, 0]),
        log_b_action=torch.log(torch.tensor([0.9, 0.1, 0.5])), game="Flappybird",
    )
    assert not w.requires_grad
    assert w.grad_fn is None


def test_correction_rejects_bad_behavior_probability():
    """错误的行为概率（>1 或 NaN）必须拒绝。"""
    m = approx_phi(0.0)
    e = torch.zeros(2, 2)
    with pytest.raises(COR.CorrectionError, match="含 NaN"):
        COR.mapping_correction(
            phi=m.phi, x=torch.zeros(2, F.OBS_DIM), e_behavior=e,
            token=torch.tensor([0, 1]),
            log_b_action=torch.tensor([float("nan"), 0.0]), game="Flappybird",
        )


def test_correction_rejects_illegal_token():
    m = approx_phi(0.0)
    e = torch.zeros(2, 2)
    with pytest.raises(COR.CorrectionError):
        COR.mapping_correction(
            phi=m.phi, x=torch.zeros(2, F.OBS_DIM), e_behavior=e,
            token=torch.tensor([0, 3]),           # 3 属于 Timberman
            log_b_action=torch.tensor([0.0, 0.0]), game="Flappybird",
        )


def test_correction_aborts_when_bound_exceeded():
    """超过上界要说清楚是记录错误，而不是静默裁剪（§8.2）。"""
    m = approx_phi(0.0)
    e = torch.zeros(2, 2)
    with pytest.raises(COR.CorrectionError, match="上界被突破"):
        COR.mapping_correction(
            phi=m.phi, x=torch.zeros(2, F.OBS_DIM), e_behavior=e,
            token=torch.tensor([0, 1]),
            log_b_action=torch.log(torch.tensor([1e-6, 1e-6])), game="Flappybird",
        )


def test_correction_shape_validation():
    m = approx_phi(0.0)
    with pytest.raises(COR.CorrectionError, match="latent_dim=2"):
        COR.mapping_correction(
            phi=m.phi, x=torch.zeros(2, F.OBS_DIM), e_behavior=torch.zeros(2, 3),
            token=torch.tensor([0, 1]), log_b_action=torch.zeros(2), game="Flappybird",
        )


def test_correction_report_ess():
    w = torch.tensor([1.0, 1.0, 1.0, 1.0])
    rep = COR.correction_report(w, "Flappybird")
    assert abs(rep.ess - 4.0) < 1e-9
    assert abs(rep.ess_over_batch - 1.0) < 1e-9
    assert rep.n_above_max == 0
    # 权重极不均衡 → ESS 下降。§8.5 的排障触发阈值是 ESS/B < 0.1，
    # 这里用更极端的权重让阈值真正被触发。
    w2 = torch.tensor([100.0, 0.01, 0.01, 0.01])
    rep2 = COR.correction_report(w2, "Flappybird")
    # 解析值：ESS = (Σw)²/Σw² = (100.03)²/(10000+3e-4) ≈ 1.0006 → ESS/B ≈ 0.25
    assert abs(rep2.ess - 1.0006) < 1e-3, rep2.ess
    assert abs(rep2.ess_over_batch - 0.25015) < 1e-3, rep2.ess_over_batch
    # 跨过 §8.5 的 ESS/B < 0.1 排障阈值：需要足够大的 batch 才会被单条样本主导。
    # B=4 时 ESS 的下界约为 1，ESS/B ≥ 0.25，所以阈值只能在更大 batch 上触发。
    B = 64
    w3 = torch.cat([torch.tensor([100.0]), torch.full((B - 1,), 0.01)])
    rep3 = COR.correction_report(w3, "Flappybird")
    assert rep3.ess_over_batch < 0.1, rep3.ess_over_batch
    assert rep3.ess > 0.9


# ================================================================ MC（§14 T5）
def test_mc_backfill_matches_plan_example():
    """§14 T5 的 MC 递推：奖励 [1,1]、γ=0.99 → 完整 MC = [1.99, 1]。

    注意方案里 §14 T5 写的示例奖励是 [1,2]，但 §5.5 明确训练奖励必须落在 [-1,1]，
    两者不能同时成立。这里按 §5.5 的硬约束取 [-1,1] 内的奖励，只验证 G 的递推本身；
    reward=2.0 会被 replay 的字段校验直接拒绝（见 test_transition_rejects_out_of_range_reward）。
    """
    buf = make_buf()
    for i, r in enumerate([1.0, 1.0]):
        tr = make_transition(
            buffer=buf, game="Flappybird", episode_id=0, request_id=i,
            x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
            global_token=0, reward_train=r, log_b_action=math.log(0.5),
            behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
        )
        buf.add_confirmed(tr, game="Flappybird")
    ep = buf.finish_episode(game="Flappybird", episode_id=0, terminated=True,
                            truncated=False, termination_reason="collision")
    gs = [t.reference_return for t in ep.transitions]
    assert abs(gs[0] - 1.99) < 1e-9, gs
    assert abs(gs[1] - 1.0) < 1e-9, gs
    assert all(t.ref_valid for t in ep.transitions)


def test_truncated_episode_has_no_mc_and_stays_invalid():
    """同样轨迹若外部截断 → 全部 ref_valid=False（§14 T5）。"""
    buf = make_buf()
    for i, r in enumerate([1.0, 1.0]):
        tr = make_transition(
            buffer=buf, game="Flappybird", episode_id=0, request_id=i,
            x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
            global_token=0, reward_train=r, log_b_action=math.log(0.5),
            behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
        )
        buf.add_confirmed(tr, game="Flappybird")
    ep = buf.finish_episode(game="Flappybird", episode_id=0, terminated=False,
                            truncated=True, termination_reason="none")
    assert ep.mc_filled is False
    assert all(t.ref_valid is False for t in ep.transitions)
    assert all(t.reference_return == 0.0 for t in ep.transitions)


def test_reward_out_of_range_is_rejected_so_mc_example_must_stay_bounded():
    """§5.5 与 §14 T5 示例冲突的显式记录：reward=2.0 必须被拒绝。"""
    buf = make_buf()
    with pytest.raises(ReplayError, match=r"\[-1,1\]"):
        buf.add_confirmed(make_transition(
            buffer=buf, game="Flappybird", episode_id=0, request_id=0,
            x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
            global_token=0, reward_train=2.0, log_b_action=math.log(0.5),
            behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
        ), game="Flappybird")


def test_mc_uses_training_reward_not_raw_score():
    """MC 必须用 bounded 训练奖励；horizon 终止也是有效 MC（§7.2）。"""
    buf = make_buf()
    tr = make_transition(
        buffer=buf, game="Timberman", episode_id=0, request_id=0,
        x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
        global_token=2, reward_train=0.5, log_b_action=math.log(0.5),
        behavior_kernel="uniform", behavior_mapping_version=1, score_raw=99.0, clock_mode="lockstep-v1",
    )
    buf.add_confirmed(tr, game="Timberman")
    ep = buf.finish_episode(game="Timberman", episode_id=0, terminated=True,
                            truncated=False, termination_reason="horizon")
    assert ep.mc_filled is True
    assert abs(ep.transitions[0].reference_return - 0.5) < 1e-12


def test_pending_then_finish_backfills_by_sample_id():
    """完整回合结束前数据可立即用于 TD（ref_valid=False），结束后补齐（§8.4）。"""
    buf = make_buf()
    tr = make_transition(
        buffer=buf, game="Flappybird", episode_id=7, request_id=0,
        x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
        global_token=0, reward_train=1.0, log_b_action=math.log(0.5),
        behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
    )
    buf.add_confirmed(tr, game="Flappybird")
    assert tr.ref_valid is False
    buf.finish_episode(game="Flappybird", episode_id=7, terminated=True,
                       truncated=False, termination_reason="ground")
    assert tr.ref_valid is True
    assert abs(tr.reference_return - 1.0) < 1e-12


# ================================================================ 事实字段校验
def test_transition_rejects_unknown_behavior_kernel():
    buf = make_buf()
    tr = make_transition(
        buffer=buf, game="Flappybird", episode_id=0, request_id=0,
        x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
        global_token=0, reward_train=0.0, log_b_action=0.0,
        behavior_kernel="mystery", behavior_mapping_version=1, clock_mode="lockstep-v1",
    )
    with pytest.raises(ReplayError, match="behavior_kernel"):
        buf.add_confirmed(tr, game="Flappybird")


def test_transition_rejects_positive_log_b():
    buf = make_buf()
    tr = make_transition(
        buffer=buf, game="Flappybird", episode_id=0, request_id=0,
        x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
        global_token=0, reward_train=0.0, log_b_action=0.5,
        behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
    )
    with pytest.raises(ReplayError, match="log_b_action 必须"):
        buf.add_confirmed(tr, game="Flappybird")


def test_transition_rejects_out_of_range_reward():
    buf = make_buf()
    tr = make_transition(
        buffer=buf, game="Flappybird", episode_id=0, request_id=0,
        x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
        global_token=0, reward_train=5.0, log_b_action=0.0,
        behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
    )
    with pytest.raises(ReplayError, match=r"\[-1,1\]"):
        buf.add_confirmed(tr, game="Flappybird")


def test_transition_rejects_token_local_mismatch():
    buf = make_buf()
    tr = make_transition(
        buffer=buf, game="Flappybird", episode_id=0, request_id=0,
        x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
        global_token=0, reward_train=0.0, log_b_action=0.0,
        behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
    )
    tr.local_action = 1              # 与 token=0 矛盾
    with pytest.raises(ReplayError, match="本地动作"):
        buf.add_confirmed(tr, game="Flappybird")


def test_transition_rejects_both_terminated_and_truncated():
    buf = make_buf()
    tr = make_transition(
        buffer=buf, game="Flappybird", episode_id=0, request_id=0,
        x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
        global_token=0, reward_train=0.0, log_b_action=0.0,
        behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
        terminated=True, truncated=True, termination_reason="collision",
    )
    with pytest.raises(ReplayError, match="不能同时为真"):
        buf.add_confirmed(tr, game="Flappybird")


def test_transition_rejects_nan_observation():
    buf = make_buf()
    bad = obs()
    bad[3] = float("nan")
    tr = make_transition(
        buffer=buf, game="Flappybird", episode_id=0, request_id=0,
        x=bad, next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
        global_token=0, reward_train=0.0, log_b_action=0.0,
        behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
    )
    with pytest.raises(ReplayError, match="NaN"):
        buf.add_confirmed(tr, game="Flappybird")


def test_episode_cannot_be_appended_after_finish():
    buf = make_buf()
    tr = make_transition(
        buffer=buf, game="Flappybird", episode_id=0, request_id=0,
        x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
        global_token=0, reward_train=0.0, log_b_action=0.0,
        behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
    )
    buf.add_confirmed(tr, game="Flappybird")
    buf.finish_episode(game="Flappybird", episode_id=0, terminated=True,
                       truncated=False, termination_reason="collision")
    tr2 = make_transition(
        buffer=buf, game="Flappybird", episode_id=0, request_id=1,
        x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
        global_token=0, reward_train=0.0, log_b_action=0.0,
        behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
    )
    with pytest.raises(ReplayError, match="已结束"):
        buf.add_confirmed(tr2, game="Flappybird")


def test_env_hash_mismatch_rejected():
    """不同 env_hash 拒绝合并；同一游戏不同 mapping_version 允许共存（§8.4）。"""
    buf = make_buf()
    for i, h in enumerate(["hash-a", "hash-a"]):
        buf.add_confirmed(make_transition(
            buffer=buf, game="Flappybird", episode_id=i, request_id=0,
            x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
            global_token=0, reward_train=0.0, log_b_action=0.0,
            behavior_kernel="uniform", behavior_mapping_version=1, env_hash=h, clock_mode="lockstep-v1",
        ), game="Flappybird")
    with pytest.raises(ReplayError, match="env_hash 不一致"):
        buf.add_confirmed(make_transition(
            buffer=buf, game="Flappybird", episode_id=9, request_id=0,
            x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
            global_token=0, reward_train=0.0, log_b_action=0.0,
            behavior_kernel="uniform", behavior_mapping_version=1, env_hash="hash-b", clock_mode="lockstep-v1",
        ), game="Flappybird")
    # 另一个游戏有自己的 env_hash：这是合法的，不能因此拒绝
    buf.add_confirmed(make_transition(
        buffer=buf, game="Timberman", episode_id=0, request_id=0,
        x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
        global_token=2, reward_train=0.0, log_b_action=0.0,
        behavior_kernel="uniform", behavior_mapping_version=1, env_hash="hash-timber", clock_mode="lockstep-v1",
    ), game="Timberman")
    assert buf.compat["env_hash:0"] == "hash-a"
    assert buf.compat["env_hash:1"] == "hash-timber"


def test_mapping_version_can_differ_within_same_game():
    buf = make_buf()
    for mv in (1, 2):
        buf.add_confirmed(make_transition(
            buffer=buf, game="Flappybird", episode_id=mv, request_id=0,
            x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
            global_token=0, reward_train=0.0, log_b_action=0.0,
            behavior_kernel="uniform", behavior_mapping_version=mv, env_hash="h", clock_mode="lockstep-v1",
        ), game="Flappybird")
    assert buf.diagnostics()["pending_episodes"] == 2


# ================================================================ 采样与容量
def _fill(buf: ReplayBuffer, game: str, episode_id: int, n: int, *, complete=True,
          reward=0.5, mv=1) -> None:
    for i in range(n):
        buf.add_confirmed(make_transition(
            buffer=buf, game=game, episode_id=episode_id, request_id=i,
            x=obs(i), next_x=obs(i + 1), e_behavior=np.zeros(2, dtype=np.float32),
            global_token=F.legal_tokens(game)[0], reward_train=reward,
            log_b_action=math.log(0.5), behavior_kernel="uniform",
            behavior_mapping_version=mv, env_hash="h", clock_mode="lockstep-v1",
        ), game=game)
    buf.finish_episode(game=game, episode_id=episode_id, terminated=complete,
                       truncated=not complete, termination_reason="collision" if complete else "none")


def test_single_task_stage_uses_only_active_task():
    buf = make_buf()
    _fill(buf, "Flappybird", 0, 300)
    plan = buf.sample_plan(0)
    assert all(task == 0 for task, _, _ in plan)
    assert sum(c for _, _, c in plan) == 128


def test_two_task_batch_is_half_half():
    buf = make_buf()
    buf.freeze_historical(0)          # F 暂无内容，先固化空历史
    _fill(buf, "Flappybird", 0, 200)
    buf.freeze_historical(0)          # F 进入历史（strict S1→S2）
    _fill(buf, "Timberman", 0, 200)   # T 在线
    plan = buf.sample_plan(1)
    total = sum(c for _, _, c in plan)
    assert total == 128
    other = sum(c for t, _, c in plan if t == 0)
    active = sum(c for t, _, c in plan if t == 1)
    assert other == 64 and active == 64


def test_assert_task_counts_rejects_non_64_64_joint_batch():
    buf = make_buf()
    _fill(buf, "Flappybird", 0, 200)
    buf.freeze_historical(0)
    _fill(buf, "Timberman", 0, 200)
    transitions = buf.sample(1)

    assert buf.task_counts(transitions) == {0: 64, 1: 64}
    buf.assert_task_counts(transitions, {0: 64, 1: 64})
    with pytest.raises(ReplayError, match="任务构成"):
        buf.assert_task_counts(transitions[:-1], {0: 64, 1: 64})


def test_active_task_splits_history_and_online():
    buf = make_buf()
    _fill(buf, "Timberman", 0, 100)
    buf.commit_online_to_history(1)
    _fill(buf, "Timberman", 1, 100)
    buf.freeze_historical(0)
    _fill(buf, "Flappybird", 0, 200)
    buf.freeze_historical(0)
    plan = buf.sample_plan(1)
    h = sum(c for t, b, c in plan if t == 1 and b == "historical")
    o = sum(c for t, b, c in plan if t == 1 and b == "online")
    assert h == 32 and o == 32


def test_sample_returns_requested_count_and_no_duplicate_state_mutation():
    buf = make_buf()
    _fill(buf, "Flappybird", 0, 500)
    s = buf.sample(0)
    assert len(s) == 128
    # 采样不修改行为概率字段
    assert all(t.log_b_action <= 0 for t in s)
    assert all(t.ref_valid for t in s)


def test_capacity_eviction_by_complete_episode():
    buf = make_buf(per_task_capacity=10)
    for ep in range(5):
        _fill(buf, "Flappybird", ep, 4)
    diag = buf.diagnostics()
    assert diag["evictions"] > 0
    assert buf._task_transitions(0) <= 10
    # 淘汰后剩下的都是完整回合（不会留下假 MC 的截断尾巴）
    remaining = buf.historical[0] + buf.online[0]
    assert all(e.completed for e in remaining)
    assert all(t.ref_valid for e in remaining for t in e.transitions)


def test_truncated_episode_keeps_invalid_after_eviction_pressure():
    buf = make_buf(per_task_capacity=6)
    _fill(buf, "Flappybird", 0, 4, complete=True)
    _fill(buf, "Flappybird", 1, 4, complete=False)
    remaining = buf.historical[0] + buf.online[0]
    for e in remaining:
        if not e.mc_filled:
            assert all(t.ref_valid is False for t in e.transitions)


def test_total_storage_respects_global_cap():
    buf = make_buf(per_task_capacity=200, max_stored_transitions=400)
    for ep in range(10):
        _fill(buf, "Flappybird", ep, 100)
        _fill(buf, "Timberman", ep, 100)
    assert buf._task_transitions(0) <= 200
    assert buf._task_transitions(1) <= 200


def test_freeze_historical_refuses_with_pending_episode():
    buf = make_buf()
    buf.add_confirmed(make_transition(
        buffer=buf, game="Flappybird", episode_id=0, request_id=0,
        x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
        global_token=0, reward_train=0.0, log_b_action=0.0,
        behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
    ), game="Flappybird")
    # pending 尚未结束 → 不允许固化（否则会丢掉该回合的 MC）
    with pytest.raises(ReplayError, match="未完成回合"):
        buf.freeze_historical(0)
    buf.finish_episode(game="Flappybird", episode_id=0, terminated=True,
                       truncated=False, termination_reason="collision")
    buf.freeze_historical(0)
    assert buf.online[0] == []


def test_historical_content_frozen_after_switch():
    """§14 T5：strict 第二阶段 F historical 内容不变，T 持续增长。"""
    buf = make_buf()
    _fill(buf, "Flappybird", 0, 50)
    buf.freeze_historical(0)
    f_snapshot = [t.sample_id for e in buf.historical[0] for t in e.transitions]
    for ep in range(1, 5):
        _fill(buf, "Timberman", ep, 50)
    f_after = [t.sample_id for e in buf.historical[0] for t in e.transitions]
    assert f_snapshot == f_after
    assert buf._task_transitions(1) > 0


def test_pending_episodes_counted_in_total():
    buf = make_buf()
    buf.add_confirmed(make_transition(
        buffer=buf, game="Flappybird", episode_id=0, request_id=0,
        x=obs(), next_x=obs(), e_behavior=np.zeros(2, dtype=np.float32),
        global_token=0, reward_train=0.0, log_b_action=0.0,
        behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
    ), game="Flappybird")
    assert buf.pending_transitions() == 1
    assert buf.total_transitions() == 1


def test_stack_keeps_behavior_fields_verbatim():
    buf = make_buf()
    _fill(buf, "Timberman", 0, 8, reward=0.25, mv=3)
    buf.commit_online_to_history(1)
    trs = buf.historical[1][0].transitions
    batch = buf.stack(trs, game_of_task={0: "Flappybird", 1: "Timberman"})
    assert batch["size"] == 8
    assert torch.allclose(batch["e_behavior"], torch.zeros(8, 2))
    assert batch["mapping_version"].tolist() == [3] * 8
    assert batch["task_id"].tolist() == [1] * 8
    assert torch.allclose(batch["reward"], torch.full((8,), 0.25))


def test_stack_does_not_rewrite_log_b():
    buf = make_buf()
    _fill(buf, "Flappybird", 0, 4)
    buf.commit_online_to_history(0)
    trs = buf.historical[0][0].transitions
    before = [t.log_b_action for t in trs]
    buf.stack(trs, game_of_task={0: "Flappybird", 1: "Timberman"})
    assert [t.log_b_action for t in trs] == before
