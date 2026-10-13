"""T3 集成验收：session → game_env 的 29 维状态、奖励与终止（假游戏，无模拟器）。"""

from __future__ import annotations

import numpy as np
import pytest

from wemi_cal.envs import bridge, features as F
from wemi_cal.envs.game_env import WemiGameEnv
from wemi_cal.envs.session import Session

from .fake_game import FakeGame, FakeGameConfig


def flappy_obs_factory(ep, t):
    return {
        "bird_y": 300.0 / F.WORLD_HEIGHT,
        "gap_top": 200.0 / F.WORLD_HEIGHT,
        "gap_bottom": 400.0 / F.WORLD_HEIGHT,
        "pipe_x": max(0.0, 200.0 - 10.0 * t) / F.WORLD_WIDTH,
    }


def timber_obs_factory(ep, t):
    return {
        "features": [1.0, 0.0, -1.0] + [0.0] * 8 + [1.0, 1.0],
    }


def make_env(game="Flappybird", *, episode_len=10, obs_factory=None, horizon=1000,
             score_factory=None, fail_mode=None, fail_after=0):
    conn = bridge.Connection(host="127.0.0.1", port=0)
    conn.create_server()
    cfg = FakeGameConfig(
        game_id=game,
        episode_len=episode_len,
        obs_factory=obs_factory or (flappy_obs_factory if game == "Flappybird" else timber_obs_factory),
        fail_mode=fail_mode,
        fail_after=fail_after,
        score_factory=score_factory or (lambda ep, t: float(t)),
    )
    game_srv = FakeGame("127.0.0.1", conn.bound_port, cfg).start()
    sess = Session(connection=conn, game_id=game, step_timeout_s=3.0, connect_timeout_s=10.0)
    sess.connect()
    env = WemiGameEnv(game, sess, horizon=horizon)
    return env, game_srv, conn


def test_flappy_env_obs_shape_and_budget():
    env, game, conn = make_env("Flappybird")
    try:
        x, info = env.reset(seed=1)
        assert x.shape == (29,) and x.dtype == np.float32
        assert x[28] == 1.0                       # §5.3：reset 剩余时域=1
        assert info["episode_id"] == 0
        assert np.allclose(x[26:28], [1, 0])      # F 的 task one-hot
        x2, r, term, trunc, st = env.step(0)
        assert abs(x2[28] - 0.999) < 1e-6
        assert term is False and trunc is False
        assert st.steps_taken == 1
    finally:
        env.close()
        game.stop()
        conn.close()


def test_timberman_env_obs_shape():
    env, game, conn = make_env("Timberman")
    try:
        x, _ = env.reset(seed=1)
        assert np.allclose(x[13:26], np.ones(13))
        assert np.allclose(x[26:28], [0, 1])
    finally:
        env.close()
        game.stop()
        conn.close()


def test_reward_is_bounded_and_logged_separately():
    """日志同时有 score_raw 与 reward_train；两者语义不同（§5.5）。"""
    env, game, conn = make_env("Timberman")
    try:
        env.reset(seed=1)
        _, r1, _, _, st = env.step(0)
        assert r1 == 1.0                     # score 0→1 增 1
        assert st.score_raw == 1.0
        assert st.reward_train == 1.0
        assert -1.0 <= r1 <= 1.0
    finally:
        env.close()
        game.stop()
        conn.close()


def test_flappy_train_reward_matches_formula_in_env():
    env, game, conn = make_env("Flappybird")
    try:
        env.reset(seed=1)
        _, r, _, _, _ = env.step(0)
        # 假游戏固定 obs：by=300, gap_top=200, gap_bottom=400 → min(100,100)=100 → 1.0
        assert abs(r - 1.0) < 1e-9
    finally:
        env.close()
        game.stop()
        conn.close()


def test_death_reward_minus_one_and_terminated():
    env, game, conn = make_env("Flappybird", episode_len=3)
    try:
        env.reset(seed=1)
        env.step(0)
        env.step(0)
        _, r, term, trunc, st = env.step(0)
        assert term is True and trunc is False
        assert st.termination_reason == "collision"
        assert st.died is True
        assert r == -1.0
    finally:
        env.close()
        game.stop()
        conn.close()


def test_horizon_termination_is_not_death():
    """horizon 到达：terminated=True、reason=horizon、奖励仍是正常值。"""
    env, game, conn = make_env("Flappybird", episode_len=10_000, horizon=3)
    try:
        env.reset(seed=1)
        _, r1, t1, _, _ = env.step(0)
        _, r2, t2, _, _ = env.step(0)
        assert (t1, t2) == (False, False)
        x, r3, t3, tr3, st = env.step(0)   # 第 3 个动作 → 预算到 0
        assert t3 is True and tr3 is False
        assert st.termination_reason == "horizon"
        assert st.died is False
        assert r3 > 0                      # 不是死亡罚分
        assert x[F.BUDGET_INDEX] == 0.0    # 剩余预算到 0
    finally:
        env.close()
        game.stop()
        conn.close()


def test_step_after_end_requires_reset():
    env, game, conn = make_env("Flappybird", episode_len=2)
    try:
        env.reset(seed=1)
        env.step(0)
        env.step(0)      # 死亡
        with pytest.raises(RuntimeError, match="先 reset"):
            env.step(0)
    finally:
        env.close()
        game.stop()
        conn.close()


def test_external_truncation_keeps_bootstrap_semantics():
    env, game, conn = make_env("Flappybird", episode_len=10_000)
    try:
        env.reset(seed=1)
        _, _, term, trunc, st = env.step(0, externally_truncated=True)
        assert term is False and trunc is True
        assert F.bootstrap_allowed(F.Termination(term, trunc, st.termination_reason)) is True
        assert F.reference_label_valid(F.Termination(term, trunc, st.termination_reason)) is False
    finally:
        env.close()
        game.stop()
        conn.close()


def test_reset_clears_episode_return_and_score():
    env, game, conn = make_env("Flappybird", episode_len=4)
    try:
        env.reset(seed=1)
        for _ in range(3):
            env.step(0)
        assert env.episode_return_train != 0.0
        env.reset(seed=2)
        assert env.episode_return_train == 0.0
        assert env.steps_taken == 0
        assert env.ended is False
    finally:
        env.close()
        game.stop()
        conn.close()


def test_action_mask_matches_env_game():
    env_f, gf, cf = make_env("Flappybird")
    env_t, gt, ct = make_env("Timberman")
    try:
        np.testing.assert_array_equal(env_f.action_mask, [1, 1, 0, 0])
        np.testing.assert_array_equal(env_t.action_mask, [0, 0, 1, 1])
        assert env_f.legal_tokens == (0, 1)
        assert env_t.legal_tokens == (2, 3)
    finally:
        env_f.close(); gf.stop(); cf.close()
        env_t.close(); gt.stop(); ct.close()


def test_env_rejects_unknown_game():
    conn = bridge.Connection(host="127.0.0.1", port=0)
    conn.create_server()
    try:
        sess = Session(connection=conn, game_id="Flappybird")
        with pytest.raises(F.FeatureError):
            WemiGameEnv("Game2048", sess)
    finally:
        conn.close()


def test_score_raw_tracked_independently_of_reward():
    """score_raw 是原始累计分；episode_return_train 是训练回报，二者不同。"""
    env, game, conn = make_env("Timberman", episode_len=20)
    try:
        env.reset(seed=1)
        for _ in range(5):
            env.step(0)
        assert env.score_raw == 5.0                    # reset 后 t=0，5 步后 t=5
        assert env.episode_return_train == 5.0         # 每步 +1
    finally:
        env.close()
        game.stop()
        conn.close()
