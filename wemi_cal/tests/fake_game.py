"""假游戏客户端：按 wemi-cal-v1 协议应答，用于离线验证 Python 侧。

它是环境侧的替身，不是策略：只按预设脚本回包，不含任何学习逻辑。
可注入故障（重复/过期/错配/缺字段/NaN/超时）来验证 bridge/session 的拒绝路径。
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .fake_client import WSClient, response


@dataclass
class FakeGameConfig:
    game_id: str = "Flappybird"
    episode_len: int = 10
    obs_factory: Callable[[int, int], Dict[str, Any]] = lambda ep, t: {"t": float(t), "v": 0.5}
    score_factory: Callable[[int, int], float] = lambda ep, t: float(t)
    reward_step: float = 0.1
    info: Dict[str, Any] = field(default_factory=lambda: {
        "clock_mode": "lockstep-v1",
        "env_hash": "fake-env-hash",
        "obs_schema": "compact29-v1",
        "reward_schema": "bounded-v1",
    })
    # 故障注入（默认全部关闭）
    fail_mode: Optional[str] = None
    fail_after: int = 0
    protocol: str = "wemi-cal-v1"
    reply_delay_s: float = 0.0
    reset_delay_s: float = 0.0


class FakeGame:
    """在后台线程里跑一个假游戏服务端会话。"""

    def __init__(self, host: str, port: int, cfg: Optional[FakeGameConfig] = None) -> None:
        self.host = host
        self.port = int(port)
        self.cfg = cfg or FakeGameConfig()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.client: Optional[WSClient] = None
        self.error: Optional[BaseException] = None
        # 记录收到的请求，供断言使用
        self.requests: List[Dict[str, Any]] = []
        self._episode_t = 0
        self._episode_id = -1
        self._episodes_seen: List[int] = []

    # ---------- 生命周期 ----------
    def start(self) -> "FakeGame":
        self._thread = threading.Thread(target=self._run, name=f"fake-{self.cfg.game_id}", daemon=True)
        self._thread.start()
        return self

    def join(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def stop(self) -> None:
        self._stop.set()
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.join(timeout=2.0)

    # ---------- 主循环 ----------
    def _run(self) -> None:
        try:
            self.client = WSClient(self.host, self.port, timeout=10.0)
            deadline = time.monotonic() + 10.0
            while True:
                try:
                    self.client.connect()
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.05)
            assert self.client.sock is not None
            # 短读超时：socket.timeout 让循环可以检查 stop 标志而不会卡满长超时
            self.client.sock.settimeout(0.2)
            n = 0
            while not self._stop.is_set():
                try:
                    msg = self.client.recv_json()
                except socket.timeout:
                    continue
                n += 1
                self.requests.append(msg)
                self._handle(msg, n)
        except ConnectionError:
            pass  # 对端正常关闭
        except BaseException as exc:  # pragma: no cover - 仅测试诊断用
            self.error = exc

    def _handle(self, msg: Dict[str, Any], msg_index: int) -> None:
        assert self.client is not None
        op = msg.get("op")
        if op == "close":
            self._stop.set()
            return
        if op == "reset":
            if self.cfg.reset_delay_s:
                time.sleep(self.cfg.reset_delay_s)
            self._episode_t = 0
            self._episode_id = int(msg["episode_id"])
            self._episodes_seen.append(self._episode_id)
            self._maybe_fail(msg)
            self._reply(msg, obs=self.cfg.obs_factory(self._episode_id, 0), score=0.0)
            return
        if op == "step":
            self._episode_t += 1
            terminated = self._episode_t >= self.cfg.episode_len
            self._maybe_fail(msg)
            self._reply(
                msg,
                obs=self.cfg.obs_factory(self._episode_id, self._episode_t),
                score=self.cfg.score_factory(self._episode_id, self._episode_t),
                terminated=terminated,
                termination_reason="collision" if terminated else "none",
                sim_ticks=self._episode_t * 3,
            )
            return
        if op == "hello":
            self._reply(msg, obs=self.cfg.obs_factory(-1, 0), score=0.0, is_hello=True)
            return
        raise AssertionError(f"假客户端收到未知 op: {op!r}")

    def _maybe_fail(self, msg: Dict[str, Any]) -> None:
        """故障注入：在回复位置制造协议问题。"""
        mode = self.cfg.fail_mode
        if mode is None:
            return
        assert self.client is not None
        rid = int(msg["request_id"])
        if rid < self.cfg.fail_after:
            return
        if mode == "silent":
            time.sleep(30.0)  # 永不回复 → 触发超时
        elif mode == "bad_json":
            self.client.send("{not json")
        elif mode == "missing_field":
            bad = response(
                game_id=self.cfg.game_id, session_id=msg["session_id"],
                episode_id=msg["episode_id"], request_id=rid, obs={"t": 0.0},
            )
            del bad["score"]
            self.client.send_json(bad)
        elif mode == "nan":
            self.client.send_json(response(
                game_id=self.cfg.game_id, session_id=msg["session_id"],
                episode_id=msg["episode_id"], request_id=rid, obs={"t": float("nan")},
            ))
        elif mode == "wrong_game":
            self.client.send_json(response(
                game_id="Timberman", session_id=msg["session_id"],
                episode_id=msg["episode_id"], request_id=rid, obs={"t": 0.0},
            ))
        elif mode == "wrong_protocol":
            self.client.send_json(response(
                game_id=self.cfg.game_id, session_id=msg["session_id"],
                episode_id=msg["episode_id"], request_id=rid, obs={"t": 0.0},
                protocol="wemi-cal-v0",
            ))
        elif mode == "bad_reason":
            self.client.send_json(response(
                game_id=self.cfg.game_id, session_id=msg["session_id"],
                episode_id=msg["episode_id"], request_id=rid, obs={"t": 0.0},
                terminated=True, termination_reason="exploded",
            ))
        elif mode == "stale_first":
            # 先发一条旧 session 的响应，再发正确响应
            self.client.send_json(response(
                game_id=self.cfg.game_id, session_id="s-old-session",
                episode_id=msg["episode_id"], request_id=rid, obs={"t": -1.0},
            ))
        else:
            raise AssertionError(f"未知 fail_mode: {mode!r}")

    def _reply(
        self,
        msg: Dict[str, Any],
        *,
        obs: Dict[str, Any],
        score: float,
        terminated: bool = False,
        termination_reason: str = "none",
        sim_ticks: int = 0,
        is_hello: bool = False,
    ) -> None:
        assert self.client is not None
        if self.cfg.reply_delay_s:
            time.sleep(self.cfg.reply_delay_s)
        self.client.send_json(response(
            protocol=self.cfg.protocol,
            game_id=self.cfg.game_id,
            session_id=msg["session_id"],
            episode_id=msg["episode_id"],
            request_id=int(msg["request_id"]),
            obs=obs,
            score=score,
            terminated=terminated,
            termination_reason=termination_reason,
            sim_ticks=sim_ticks,
            info=dict(self.cfg.info),
        ))
