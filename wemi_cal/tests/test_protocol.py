"""T1 验收：协议配对、超时、错配拒绝（技术方案 §4.2）。

不依赖微信开发者工具：全部用假客户端 + 假游戏。
"""

from __future__ import annotations

import pytest

from wemi_cal.config import ConfigError, load_config
from wemi_cal.envs import bridge, session as session_mod
from wemi_cal.envs.session import EnvRejectedError, EnvTimeout, ProtocolError, Session

from .fake_game import FakeGame, FakeGameConfig

CONFIG_PATH = "wemi_cal/configs/mt_cal.yaml"


def _pair(cfg: FakeGameConfig | None = None, *, step_timeout_s: float = 5.0,
          connect_timeout_s: float = 10.0):
    """起一个端口 0（系统分配）的服务端 + 假游戏客户端，返回连接好的 Session。"""
    conn = bridge.Connection(host="127.0.0.1", port=0)
    conn.create_server()
    port = conn.bound_port
    game = FakeGame("127.0.0.1", port, cfg or FakeGameConfig()).start()
    sess = Session(
        connection=conn,
        game_id=(cfg.game_id if cfg else "Flappybird"),
        step_timeout_s=step_timeout_s,
        connect_timeout_s=connect_timeout_s,
    )
    sess.connect()
    return sess, game, conn


# ---------------------------------------------------------------- 正常路径
def test_hello_reset_step_ordering():
    """正常 reset/step：请求次序正确，episode/request 单调递增。"""
    sess, game, conn = _pair()
    try:
        r0 = sess.reset(seed=12345)
        assert r0.raw["episode_id"] == 0
        assert r0.raw["request_id"] == 1  # 0 被 hello 用掉
        assert r0.raw["termination_reason"] == "none"
        assert r0.raw["terminated"] is False

        r1 = sess.step(0)
        assert r1.sim_ticks == 3
        r2 = sess.step(1)
        assert r2.sim_ticks == 6
        assert r2.score == 2.0
        assert r2.terminated is False

        ops = [m["op"] for m in game.requests]
        assert ops == ["hello", "reset", "step", "step"]
        # 每个请求三元组唯一且递增
        triples = [(m["session_id"], m["episode_id"], m["request_id"]) for m in game.requests]
        assert len(set(triples)) == len(triples)
    finally:
        sess.close()
        game.stop()
        conn.close()


def test_reset_returns_early_without_executing_action():
    """reset 不得在同一条消息后再触发一次 step（§4.2）。"""
    sess, game, conn = _pair()
    try:
        sess.reset(seed=1)
        assert [m["op"] for m in game.requests] == ["hello", "reset"]
        # 观测是 reset 后的初始观测
        assert sess.step_count == 0
    finally:
        sess.close()
        game.stop()
        conn.close()


def test_reset_carries_seed_and_no_action():
    sess, game, conn = _pair()
    try:
        sess.reset(seed=987)
        reset_msg = [m for m in game.requests if m["op"] == "reset"][0]
        assert reset_msg["seed"] == 987
        assert "action" not in reset_msg
    finally:
        sess.close()
        game.stop()
        conn.close()


def test_new_episode_increments_episode_id():
    sess, game, conn = _pair()
    try:
        sess.reset(seed=1)
        for _ in range(3):
            sess.step(0)
        sess.reset(seed=2)
        assert game._episodes_seen == [0, 1]
    finally:
        sess.close()
        game.stop()
        conn.close()


def test_step_before_reset_rejected():
    sess, game, conn = _pair()
    try:
        with pytest.raises(ProtocolError, match="必须先 reset"):
            sess.step(0)
    finally:
        sess.close()
        game.stop()
        conn.close()


# ---------------------------------------------------------------- 超时
def test_step_timeout_raises_and_fabricates_nothing():
    """静默不回复 → EnvTimeout；不返回 reward=0/done=true 的假 transition。

    请求编号实测序列：hello=0, reset=1, step#1=2, step#2=3, step#3=4。
    注入从 rid>=3 开始静默，因此前两次 step 正常、第三次超时。
    """
    cfg = FakeGameConfig(fail_mode="silent", fail_after=3)
    sess, game, conn = _pair(cfg, step_timeout_s=0.6)
    try:
        sess.reset(seed=1)
        assert sess.step(0).sim_ticks == 3
        with pytest.raises(EnvTimeout):
            sess.step(0)
        assert sess.timeout_total == 1
        # 超时后步数不增加
        assert sess.step_count == 1
    finally:
        game.stop()
        conn.close()


def test_waiting_client_timeout():
    """没有客户端连接时，等待必须超时而不是死转。"""
    conn = bridge.Connection(host="127.0.0.1", port=0)
    conn.create_server()
    try:
        with pytest.raises(EnvTimeout):
            conn.waiting_client(timeout_s=0.3)
    finally:
        conn.close()


def test_recv_message_timeout_not_none():
    conn = bridge.Connection(host="127.0.0.1", port=0)
    conn.create_server()
    try:
        with pytest.raises(EnvTimeout):
            conn.recv_message(timeout_s=0.2)
    finally:
        conn.close()


# ---------------------------------------------------------------- 错配与脏数据
def test_wrong_game_rejected():
    """连错游戏必须拒绝（§4.1）。"""
    conn = bridge.Connection(host="127.0.0.1", port=0)
    conn.create_server()
    game = FakeGame("127.0.0.1", conn.bound_port, FakeGameConfig(game_id="Timberman")).start()
    sess = Session(connection=conn, game_id="Flappybird", connect_timeout_s=10.0)
    try:
        with pytest.raises(EnvRejectedError, match="game_id 错配"):
            sess.connect()
    finally:
        game.stop()
        conn.close()


def test_wrong_protocol_rejected_at_handshake():
    conn = bridge.Connection(host="127.0.0.1", port=0)
    conn.create_server()
    game = FakeGame("127.0.0.1", conn.bound_port, FakeGameConfig(protocol="wemi-cal-v0")).start()
    sess = Session(connection=conn, game_id="Flappybird", connect_timeout_s=10.0)
    try:
        with pytest.raises(ProtocolError, match="协议版本错配"):
            sess.connect()
    finally:
        game.stop()
        conn.close()


def test_env_hash_mismatch_rejected():
    conn = bridge.Connection(host="127.0.0.1", port=0)
    conn.create_server()
    game = FakeGame("127.0.0.1", conn.bound_port, FakeGameConfig()).start()
    sess = Session(
        connection=conn, game_id="Flappybird", connect_timeout_s=10.0,
        expected_env_hash="expected-hash",
    )
    try:
        with pytest.raises(EnvRejectedError, match="env_hash 错配"):
            sess.connect()
    finally:
        game.stop()
        conn.close()


def test_clock_mode_mismatch_rejected():
    cfg = FakeGameConfig(info={"clock_mode": "realtime", "env_hash": "h",
                              "obs_schema": "compact29-v1", "reward_schema": "bounded-v1"})
    conn = bridge.Connection(host="127.0.0.1", port=0)
    conn.create_server()
    game = FakeGame("127.0.0.1", conn.bound_port, cfg).start()
    sess = Session(connection=conn, game_id="Flappybird", connect_timeout_s=10.0)
    try:
        with pytest.raises(EnvRejectedError, match="clock_mode 错配"):
            sess.connect()
    finally:
        game.stop()
        conn.close()


@pytest.mark.parametrize("mode,exc,match", [
    ("bad_json", ProtocolError, "合法 JSON"),
    ("missing_field", ProtocolError, "缺少字段"),
    ("nan", ProtocolError, "NaN"),
    ("bad_reason", ProtocolError, "termination_reason"),
])
def test_dirty_responses_rejected(mode, exc, match):
    """NaN/null/缺字段/未知枚举都必须报错，不能转成 0 继续训练（§4.2 规则 5）。"""
    cfg = FakeGameConfig(fail_mode=mode, fail_after=0)
    sess, game, conn = _pair(cfg)
    try:
        with pytest.raises(exc, match=match):
            sess.reset(seed=1)
    finally:
        game.stop()
        conn.close()


def test_stale_response_is_dropped_not_mistaken_for_current():
    """旧 session 的响应被丢弃，当前请求仍拿到正确结果。"""
    cfg = FakeGameConfig(fail_mode="stale_first", fail_after=1)
    sess, game, conn = _pair(cfg)
    try:
        sess.reset(seed=1)
        resp = sess.step(0)
        assert resp.sim_ticks == 3
        # reset(rid=1) 与 step(rid=2) 各被注入一条旧 session 响应
        assert resp.stale_dropped == 1
        assert sess.stale_total == 2
    finally:
        game.stop()
        conn.close()


def test_second_client_is_ignored_on_same_port():
    """同一端口只允许一个训练连接（§4.1）。"""
    sess, game, conn = _pair()
    extra = None
    try:
        from .fake_client import WSClient
        extra = WSClient("127.0.0.1", conn.bound_port, timeout=2.0)
        extra.connect()
        sess.reset(seed=1)
        # 第二个连接不会顶掉第一个：active 仍是原客户端
        assert conn._active is not None
        assert conn._active["id"] not in conn._ignored_client_ids
        # 原会话仍然可用
        assert sess.step(0).sim_ticks == 3
    finally:
        if extra is not None:
            extra.close()
        sess.close()
        game.stop()
        conn.close()


# ---------------------------------------------------------------- 重连
def test_reconnect_uses_new_session_and_clears_queue():
    sess, game, conn = _pair()
    try:
        old_session = sess.session_id
        sess.reset(seed=1)
        sess.step(0)
        sess.reconnect()
        assert sess.session_id != old_session
        assert sess.episode_id == -1
        assert sess.request_id == 1  # 新 session 的 hello 已用掉 request_id=0
        # 新 session 上 reset 正常，且 episode 从 0 重新开始
        r = sess.reset(seed=5)
        assert r.raw["session_id"] == sess.session_id
    finally:
        game.stop()
        conn.close()


# ---------------------------------------------------------------- 连接层细节
def test_connection_drain_discards_backlog():
    conn = bridge.Connection(host="127.0.0.1", port=0)
    conn.create_server()
    try:
        conn._messages.put("a")
        conn._messages.put("b")
        assert conn.drain() == 2
        assert conn.drain() == 0
    finally:
        conn.close()


def test_send_without_client_raises():
    conn = bridge.Connection(host="127.0.0.1", port=0)
    conn.create_server()
    try:
        with pytest.raises(bridge.NoClientError):
            conn.send_message("{}")
    finally:
        conn.close()


# ---------------------------------------------------------------- 配置相关
def test_config_step_timeout_matches_plan():
    """配置里的连接/步超时供 session 使用，不能被随手改。"""
    cfg = load_config(CONFIG_PATH)
    assert cfg.env["connect_timeout_s"] == 60
    assert cfg.env["step_timeout_s"] == 5


def test_config_rejects_unknown_protocol(tmp_path):
    import yaml

    raw = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    raw["env"]["protocol"] = "wemi-cal-v0"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="protocol"):
        load_config(p)


def test_config_rejects_shared_port(tmp_path):
    import yaml

    raw = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    raw["env"]["ports"] = {"Flappybird": 17911, "Timberman": 17911}
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="端口必须不同"):
        load_config(p)


def test_session_module_has_no_termination_side_channel():
    """TERMINATION_REASONS 必须是有限枚举且含 horizon。"""
    assert set(session_mod.TERMINATION_REASONS) == {
        "none", "collision", "ground", "blood", "horizon"
    }
