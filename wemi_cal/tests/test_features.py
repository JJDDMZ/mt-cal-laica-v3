"""T3 验收：特征、动作掩码、奖励与终止语义（技术方案 §5）。

关键区分：feature_mask vs action_mask；horizon vs 死亡 vs 外部截断。
"""

from __future__ import annotations

import numpy as np
import pytest

from wemi_cal.envs import features as F


class Resp:
    """最小 Response 替身：只暴露 reward/终止语义需要的字段。"""

    def __init__(self, obs, score=0.0, terminated=False, reason="none"):
        self.obs = obs
        self.score = float(score)
        self.terminated = bool(terminated)
        self.termination_reason = reason


def flappy_obs(by=300.0, gt=200.0, gb=400.0, px=200.0):
    return {
        "bird_y": by / F.WORLD_HEIGHT,
        "gap_top": gt / F.WORLD_HEIGHT,
        "gap_bottom": gb / F.WORLD_HEIGHT,
        "pipe_x": px / F.WORLD_WIDTH,
    }


def timber_obs(sides=None, npc=1.0, blood=6000.0):
    s = list(sides) if sides is not None else [0.0] * 11
    return {"features": s + [npc, blood / F.BLOOD_MAX]}


# ---------------------------------------------------------------- 29 维结构
def test_obs_dim_is_29():
    assert F.OBS_DIM == 29
    x = F.encode_observation("Flappybird", flappy_obs(), steps_taken=1, horizon=1000)
    assert x.shape == (29,)
    assert x.dtype == np.float32


def test_flappy_layout_features_mask_task_budget():
    x = F.encode_observation("Flappybird", flappy_obs(by=300, gt=200, gb=400, px=200),
                             steps_taken=250, horizon=1000)
    np.testing.assert_allclose(x[0:4], [0.5, 200 / 600, 400 / 600, 0.5], rtol=0, atol=1e-6)
    np.testing.assert_allclose(x[4:13], np.zeros(9), atol=0)
    # feature_mask：F 前 4 项为 1，其余 0
    np.testing.assert_allclose(x[13:26], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], atol=0)
    np.testing.assert_allclose(x[26:28], [1, 0], atol=0)
    assert abs(x[28] - (1000 - 250) / 1000) < 1e-6


def test_timberman_layout_all_mask_ones():
    x = F.encode_observation("Timberman", timber_obs(), steps_taken=1, horizon=1000)
    np.testing.assert_allclose(x[13:26], np.ones(13), atol=0)
    np.testing.assert_allclose(x[26:28], [0, 1], atol=0)
    assert abs(x[28] - (1000 - 1) / 1000) < 1e-6


def test_budget_goes_one_to_zero():
    """§5.3：reset=1，horizon 终止=0，H=1000 给 1000 个可用动作。"""
    first = F.encode_observation("Flappybird", flappy_obs(), steps_taken=0, horizon=1000)
    after_one = F.encode_observation("Flappybird", flappy_obs(), steps_taken=1, horizon=1000)
    last = F.encode_observation("Flappybird", flappy_obs(), steps_taken=1000, horizon=1000)
    assert first[28] == 1.0
    assert abs(after_one[28] - 0.999) < 1e-6
    assert last[28] == 0.0


def test_feature_mask_distinguishes_real_zero_from_padding():
    """真 0 与 pad 0 必须能区分：mask 不同。"""
    real_zero = F.encode_observation("Timberman", timber_obs(sides=[0.0] * 11),
                                     steps_taken=1, horizon=1000)
    padded = F.encode_observation("Flappybird", flappy_obs(by=0.0 * F.WORLD_HEIGHT),
                                  steps_taken=1, horizon=1000)
    assert real_zero[0] == 0.0 and real_zero[13] == 1.0     # 真 0，mask=1
    assert padded[4] == 0.0 and padded[17] == 0.0           # pad 0，mask=0


def test_steps_taken_bounds():
    # reset 观测是第 0 步（剩余预算=1）；不允许为负或超过 horizon
    x0 = F.encode_observation("Flappybird", flappy_obs(), steps_taken=0, horizon=1000)
    assert x0[F.BUDGET_INDEX] == 1.0
    with pytest.raises(F.FeatureError):
        F.encode_observation("Flappybird", flappy_obs(), steps_taken=-1, horizon=1000)
    with pytest.raises(F.FeatureError):
        F.encode_observation("Flappybird", flappy_obs(), steps_taken=1001, horizon=1000)


def test_no_running_normalization_state():
    """编码是纯函数：同输入同输出，且不因历史改变（§5.3）。"""
    a = F.encode_observation("Flappybird", flappy_obs(), steps_taken=5, horizon=1000)
    for _ in range(100):
        F.encode_observation("Timberman", timber_obs(), steps_taken=7, horizon=1000)
    b = F.encode_observation("Flappybird", flappy_obs(), steps_taken=5, horizon=1000)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------- 命名字段校验
def test_flappy_rejects_null_and_nan():
    with pytest.raises(F.FeatureError, match="null"):
        F.encode_observation("Flappybird", {**flappy_obs(), "bird_y": None},
                             steps_taken=1, horizon=1000)
    with pytest.raises(F.FeatureError, match="NaN"):
        F.encode_observation("Flappybird", {**flappy_obs(), "pipe_x": float("nan")},
                             steps_taken=1, horizon=1000)


def test_flappy_rejects_missing_field():
    bad = flappy_obs()
    del bad["pipe_x"]
    with pytest.raises(F.FeatureError, match="缺少字段"):
        F.encode_observation("Flappybird", bad, steps_taken=1, horizon=1000)


def test_timberman_rejects_string_feature_list():
    """不要先用 np.array 把混合列表强转成字符串（§5.2）。"""
    with pytest.raises(F.FeatureError, match="不能是字符串"):
        F.encode_observation("Timberman", {"features": "not-a-list"},
                             steps_taken=1, horizon=1000)


def test_timberman_rejects_unknown_side_enum():
    bad = timber_obs()
    bad["features"][3] = 2.0
    with pytest.raises(F.FeatureError, match="树方位"):
        F.encode_timberman(bad)


def test_timberman_rejects_wrong_length():
    with pytest.raises(F.FeatureError, match="长度必须是 13"):
        F.encode_timberman({"features": [0.0] * 12})


def test_timberman_center_vs_side_encoding():
    """center(0) 与 ±1 必须可区分，且 npc 只有 ±1。"""
    o = F.encode_timberman(timber_obs(sides=[0.0, 1.0, -1.0] + [0.0] * 8, npc=-1.0))
    assert o[0] == 0.0 and o[1] == 1.0 and o[2] == -1.0
    assert o[11] == -1.0
    with pytest.raises(F.FeatureError, match="npc_side"):
        F.encode_timberman(timber_obs(npc=0.0))


# ---------------------------------------------------------------- 动作注册表
def test_action_registry_four_tokens():
    assert len(F.ACTION_REGISTRY) == 4
    assert F.ACTION_REGISTRY[0] == ("Flappybird", 0)
    assert F.ACTION_REGISTRY[1] == ("Flappybird", 1)
    assert F.ACTION_REGISTRY[2] == ("Timberman", 0)
    assert F.ACTION_REGISTRY[3] == ("Timberman", 1)


def test_action_masks_are_disjoint():
    mf = F.action_mask("Flappybird")
    mt = F.action_mask("Timberman")
    np.testing.assert_array_equal(mf, [1, 1, 0, 0])
    np.testing.assert_array_equal(mt, [0, 0, 1, 1])
    np.testing.assert_array_equal(mf + mt, np.ones(4))
    # 两个掩码是不同对象，改一个不影响另一个
    mf[0] = 0
    assert F.action_mask("Flappybird")[0] == 1


def test_token_to_local_decoding():
    assert F.local_action("Flappybird", 0) == 0
    assert F.local_action("Flappybird", 1) == 1
    assert F.local_action("Timberman", 2) == 0
    assert F.local_action("Timberman", 3) == 1


@pytest.mark.parametrize("game,token", [
    ("Flappybird", 2), ("Flappybird", 3), ("Timberman", 0), ("Timberman", 1),
    ("Flappybird", -1), ("Timberman", 4),
])
def test_illegal_token_errors_not_clamped(game, token):
    with pytest.raises(F.FeatureError, match="不 clamp"):
        F.local_action(game, token)


def test_token_roundtrip():
    for game in F.GAMES:
        for tok in F.legal_tokens(game):
            assert F.token_for_local(game, F.local_action(game, tok)) == tok


def test_task_ids():
    assert F.task_id("Flappybird") == 0
    assert F.task_id("Timberman") == 1
    with pytest.raises(F.FeatureError):
        F.task_id("Game2048")


# ---------------------------------------------------------------- F 奖励
@pytest.mark.parametrize("by,gt,gb,expected", [
    (300, 200, 400, 1.0),      # 正中：min(100,100)=100 → 1.0
    (200, 200, 400, 0.0),      # 边界相等 → 0
    (400, 200, 400, 0.0),      # 边界相等（另一侧）→ 0
    (250, 200, 400, 0.5),      # 距上 50
    (350, 200, 400, 0.5),      # 距下 50
    (100, 200, 400, -1.0),     # 越上界 -100 → -1.0
    (550, 200, 400, -1.0),     # 越下界
])
def test_flappy_reward_formula(by, gt, gb, expected):
    assert abs(F.flappy_reward(died=False, bird_y=by, gap_top=gt, gap_bottom=gb) - expected) < 1e-12


def test_flappy_reward_clipped():
    """间隙很大时也不超过 1。"""
    assert F.flappy_reward(died=False, bird_y=500, gap_top=0, gap_bottom=1000) == 1.0


def test_flappy_death_reward_is_minus_one():
    assert F.flappy_reward(died=True, bird_y=300, gap_top=200, gap_bottom=400) == -1.0


# ---------------------------------------------------------------- T 奖励
def test_timberman_reward_increment():
    assert F.timberman_reward(died=False, score_t=3, score_next=4) == 1.0
    assert F.timberman_reward(died=False, score_t=3, score_next=3) == 0.0
    assert F.timberman_reward(died=True, score_t=3, score_next=3) == -1.0   # 死亡无增分
    assert F.timberman_reward(died=True, score_t=3, score_next=4) == 0.0    # 死亡但增分 → 1-1=0


def test_timberman_reward_clipped():
    assert F.timberman_reward(died=False, score_t=0, score_next=5) == 1.0
    assert F.timberman_reward(died=True, score_t=10, score_next=0) == -1.0  # 分数回退也被 clip


def test_timberman_cumulative_score_is_not_summed_as_reward():
    """原始累计分数不能逐步求和当奖励（§5.5）：3→4 与 100→101 都是 1。"""
    assert F.timberman_reward(died=False, score_t=3, score_next=4) == 1.0
    assert F.timberman_reward(died=False, score_t=100, score_next=101) == 1.0


# ---------------------------------------------------------------- 统一 train_reward
def test_train_reward_dispatch_flappy():
    r = F.train_reward("Flappybird", Resp(flappy_obs(by=250, gt=200, gb=400)),
                       score_before=0.0, died=False)
    assert abs(r - 0.5) < 1e-9


def test_train_reward_dispatch_timberman():
    r = F.train_reward("Timberman", Resp(timber_obs(), score=7.0), score_before=6.0, died=False)
    assert r == 1.0


# ---------------------------------------------------------------- 终止语义
def test_horizon_is_not_death():
    """§4.5：到达 H 是 horizon，不是死亡，不加死亡罚分。"""
    term = F.classify_termination(
        response=Resp(flappy_obs(), terminated=False), steps_taken=1000, horizon=1000
    )
    assert term.terminated and term.reason == "horizon"
    assert not term.died
    r = F.train_reward("Flappybird", Resp(flappy_obs(by=300, gt=200, gb=400)),
                       score_before=0.0, died=term.died)
    assert r > 0  # 正常奖励，不是 -1


def test_game_death_reasons():
    for reason in ("collision", "ground", "blood"):
        term = F.classify_termination(
            response=Resp(flappy_obs(), terminated=True, reason=reason),
            steps_taken=17, horizon=1000,
        )
        assert term.terminated and not term.truncated
        assert term.died is True
        assert term.reason == reason


def test_unknown_termination_reason_rejected():
    with pytest.raises(F.FeatureError, match="未知终止原因"):
        F.classify_termination(
            response=Resp(flappy_obs(), terminated=True, reason="exploded"),
            steps_taken=1, horizon=1000,
        )


def test_external_truncation_keeps_bootstrap_and_no_mc():
    """操作性中止：truncated=True、保留 bootstrap、不产生 MC 标签（§4.5/§7.2）。"""
    term = F.classify_termination(
        response=Resp(flappy_obs(), terminated=False), steps_taken=5, horizon=1000,
        externally_truncated=True,
    )
    assert term.truncated and not term.terminated
    assert F.bootstrap_allowed(term) is True
    assert F.reference_label_valid(term) is False


def test_real_terminal_has_no_bootstrap_but_has_mc():
    term = F.classify_termination(
        response=Resp(flappy_obs(), terminated=True, reason="collision"),
        steps_taken=5, horizon=1000,
    )
    assert F.bootstrap_allowed(term) is False
    assert F.reference_label_valid(term) is True


def test_horizon_terminal_has_mc_label():
    term = F.classify_termination(
        response=Resp(flappy_obs(), terminated=False), steps_taken=1000, horizon=1000
    )
    assert F.reference_label_valid(term) is True
    assert F.bootstrap_allowed(term) is False


def test_steps_exceeding_horizon_rejected():
    with pytest.raises(F.FeatureError, match="超过 horizon"):
        F.classify_termination(response=Resp(flappy_obs()), steps_taken=1001, horizon=1000)


# ---------------------------------------------------------------- T3 通过证据：target 数值
def test_bootstrap_target_arithmetic_from_plan():
    """§14 T3 通过证据：r=1, γ=0.99, next_value=10 时
    terminal target = 1；外部截断 target = 1 + 0.99*10 = 10.9。
    """
    gamma = 0.99
    r = 1.0
    next_v = 10.0

    term_terminal = F.classify_termination(
        response=Resp(flappy_obs(), terminated=True, reason="collision"),
        steps_taken=3, horizon=1000,
    )
    y_term = r + gamma * (0.0 if term_terminal.terminated else 1.0) * next_v
    assert abs(y_term - 1.0) < 1e-12

    term_trunc = F.classify_termination(
        response=Resp(flappy_obs(), terminated=False), steps_taken=3, horizon=1000,
        externally_truncated=True,
    )
    y_trunc = r + gamma * (0.0 if term_trunc.terminated else 1.0) * next_v
    assert abs(y_trunc - 10.9) < 1e-9


# ---------------------------------------------------------------- 旧包装对照
def test_legacy_pad_wrapper_is_36_dim():
    x = F.legacy_pad_observation([float(i) for i in range(34)], "Timberman")
    assert x.shape == (36,)
    assert x[34] == 0.0 and x[35] == 1.0
