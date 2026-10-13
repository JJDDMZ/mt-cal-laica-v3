"""real_env 的两道护栏（这两个 bug 都在真机实验里造成过误判）：

1. **观测必须来自扩展后的客户端**。原版 F 客户端只发 3 项（无 pipe_x / score）。
   如果静默接受 3 项：``pipe_x`` 会被伪造成 0，且 ``score_raw`` 恒为 0 →
   评测会输出"0 分/0 通过"，看起来像"模型学废了"，实际是**打开错了项目目录**。
   必须响亮失败。

2. **客户端掉线必须快速失败**。原版小游戏客户端是**纯反应式**的（只在收到服务端
   消息后才回一帧），且**没有重连**。任何一次"编译"都会杀掉 socket，
   原来的实现会静默干等 30s 才超时，让人以为是"没连上"。
"""

from __future__ import annotations

import json
import time
import types
from pathlib import Path

import pytest

from wemi_cal import real_env as RE

WEMI_ROOT = Path(__file__).resolve().parents[2] / "WemiEnv"


@pytest.fixture()
def env(monkeypatch):
    """构造 RealTimeGameEnv，但不真的起服务端（不 import 原版游戏模块）。"""
    monkeypatch.setattr(RE, "_load_legacy_module", lambda game, root: types.SimpleNamespace())
    e = RE.RealTimeGameEnv("Flappybird", wemi_env_root=WEMI_ROOT)
    return e


def _attach_fake_conn(env, *, messages, clients=None, drop_after=None):
    """给 env 装一个假 conn：``messages`` 依次返回，之后返回 None。"""

    class FakeServer:
        def __init__(self):
            self.clients = list(clients if clients is not None else [object()])

    class FakeConn:
        def __init__(self):
            self.server = FakeServer()
            self._n = 0
            self.sent = []

        def recv_message(self):
            if drop_after is not None and self._n >= drop_after:
                self.server.clients = []          # 客户端走了
                return None
            if self._n >= len(messages):
                return None
            m = messages[self._n]
            self._n += 1
            return m

        def send_message(self, msg):
            self.sent.append(msg)

    conn = FakeConn()
    env._env = types.SimpleNamespace(conn=conn)
    return conn


# ---------------------------------------------------------------- 护栏 1
def test_flappy_rejects_legacy_three_field_obs(env):
    """3 维观测 = 未扩展的原版客户端 → 必须报错，且提示打开了哪个目录。"""
    with pytest.raises(RE.LegacyProtocolError) as ei:
        env._to_named([100.0, 200.0, 150.0])
    msg = str(ei.value)
    assert "wechat_games_rt" in msg, "错误信息必须告诉操作者该开哪个目录"
    assert "5" in msg


def test_flappy_accepts_extended_obs_and_reads_score(env):
    named = env._to_named([100.0, 200.0, 150.0, 320.0, 7.0])
    assert env._native_score == 7.0
    assert named["pipe_x"] > 0.0


def test_flappy_extended_obs_pipe_x_is_real_not_fabricated(env):
    """扩展客户端的 pipe_x 必须来自观测，不能是 0 占位。"""
    named = env._to_named([100.0, 200.0, 150.0, 320.0, 0.0])
    assert named["pipe_x"] == pytest.approx(320.0 / RE.F.WORLD_WIDTH)


# ---------------------------------------------------------------- 护栏 2
def test_recv_returns_valid_reply(env):
    _attach_fake_conn(env, messages=[json.dumps(
        {"obs": [1.0, 2.0, 3.0, 4.0, 5.0], "reward": 0.5, "done": False})])
    rep = env._recv(1.0)
    assert rep.obs == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert rep.reward == 0.5 and rep.done is False


def test_recv_fails_fast_when_client_disappears(env):
    """客户端掉线时应立刻抛 ClientGoneError，而不是干等到 action_timeout。"""
    _attach_fake_conn(env, messages=[], clients=[], drop_after=0)
    t0 = time.monotonic()
    with pytest.raises(RE.ClientGoneError) as ei:
        env._recv(30.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, f"掉线后等了 {elapsed:.1f}s，太久了"
    assert "编译" in str(ei.value)


def test_client_gone_error_is_a_legacy_protocol_error(env):
    assert issubclass(RE.ClientGoneError, RE.LegacyProtocolError)


# ---------------------------------------------------------------- 启动横幅
def test_startup_hint_names_the_right_project_dir(env):
    hint = env.operator_hint()
    assert "wechat_games_rt/Flappybird" in hint
    assert "编译" in hint
