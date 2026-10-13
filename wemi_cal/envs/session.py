"""会话层：请求/响应配对、超时、错配拒绝（技术方案 §4.2，任务 T1）。

契约要点：
- ``op ∈ {hello, reset, step, close}``；reset 额外带 seed，且完成后立即返回，
  绝不在同一条消息后再触发一次 step。
- 每个环境最多一个未完成请求；响应必须匹配 ``(session_id, episode_id, request_id)``。
- 超时抛 EnvTimeout；不伪造 reward=0 / done=true，也不把该步放进 replay。
- 协议/game_id 错配、非法 JSON、缺字段都明确失败并留下证据。
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .bridge import Connection, EnvTimeout

PROTOCOL = "wemi-cal-v1"
TERMINATION_REASONS = ("none", "collision", "ground", "blood", "horizon")
OPS = ("hello", "reset", "step", "close")

# 响应必须携带的字段（§4.2）
REQUIRED_RESPONSE_FIELDS = (
    "protocol", "game_id", "session_id", "episode_id", "request_id",
    "obs", "score", "terminated", "termination_reason", "sim_ticks", "info",
)


class ProtocolError(RuntimeError):
    """协议错配、字段缺失、非法枚举。"""


class StaleResponseError(ProtocolError):
    """收到了不属于当前待处理请求的响应。"""


class EnvRejectedError(ProtocolError):
    """连接时 game_id / 协议版本 / env_hash 不匹配，必须拒绝（§4.1）。"""


@dataclass
class Response:
    """一条已确认的响应，保留原始 JSON 与配对信息。"""

    raw: Dict[str, Any]
    received_at: float
    latency_s: float = 0.0
    stale_dropped: int = 0

    @property
    def obs(self) -> Dict[str, Any]:
        return self.raw["obs"]

    @property
    def score(self) -> float:
        return float(self.raw["score"])

    @property
    def terminated(self) -> bool:
        return bool(self.raw["terminated"])

    @property
    def termination_reason(self) -> str:
        return str(self.raw["termination_reason"])

    @property
    def sim_ticks(self) -> int:
        return int(self.raw["sim_ticks"])

    @property
    def info(self) -> Dict[str, Any]:
        return dict(self.raw.get("info") or {})

    @property
    def game_over(self) -> bool:
        """游戏自身结束（碰撞/落地/血尽），不含 horizon。"""
        return self.terminated and self.termination_reason != "horizon"


def _validate_response(msg: Mapping[str, Any]) -> None:
    missing = [k for k in REQUIRED_RESPONSE_FIELDS if k not in msg]
    if missing:
        raise ProtocolError(f"响应缺少字段: {missing}")
    if msg["termination_reason"] not in TERMINATION_REASONS:
        raise ProtocolError(
            f"未知 termination_reason: {msg['termination_reason']!r}，允许 {TERMINATION_REASONS}"
        )
    if not isinstance(msg["terminated"], bool):
        raise ProtocolError(f"terminated 必须是 bool，得到 {type(msg['terminated']).__name__}")
    if not isinstance(msg["obs"], dict):
        raise ProtocolError(f"obs 必须是命名对象，得到 {type(msg['obs']).__name__}")
    try:
        score = float(msg["score"])
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"score 不是数值: {msg['score']!r}") from exc
    if score != score:  # NaN
        raise ProtocolError("score 是 NaN")
    for key, val in msg["obs"].items():
        if isinstance(val, float) and val != val:
            raise ProtocolError(f"obs.{key} 是 NaN")
        if val is None:
            raise ProtocolError(f"obs.{key} 是 null")


@dataclass
class Session:
    """一个环境（一个游戏 + 一个端口）上的训练会话。

    ``env_hash`` 由环境侧（JS）给出；会话只负责比对，不自己编造。
    """

    connection: Connection
    game_id: str
    session_id: str = field(default_factory=lambda: f"s-{uuid.uuid4().hex[:12]}")
    episode_id: int = -1
    request_id: int = 0
    expected_env_hash: Optional[str] = None
    connect_timeout_s: float = 60.0
    step_timeout_s: float = 5.0
    obs_schema: Optional[str] = None
    reward_schema: Optional[str] = None
    clock_mode: Optional[str] = None
    env_hash: Optional[str] = None

    # ---- 统计与证据 ----
    stale_total: int = 0
    timeout_total: int = 0
    step_count: int = 0
    reset_count: int = 0
    bad_messages: List[str] = field(default_factory=list)

    # ---------- 连接 ----------
    def connect(self) -> Dict[str, Any]:
        """等待客户端并完成 hello 握手；错配直接拒绝（§4.1）。"""
        self.connection.waiting_client(timeout_s=self.connect_timeout_s)
        hello = self._request("hello", timeout_s=self.connect_timeout_s)
        info = hello.info
        got_protocol = hello.raw["protocol"]
        if got_protocol != PROTOCOL:
            raise EnvRejectedError(f"协议版本错配: 期望 {PROTOCOL}，得到 {got_protocol!r}")
        if hello.raw["game_id"] != self.game_id:
            raise EnvRejectedError(
                f"连错游戏: 期望 {self.game_id}，得到 {hello.raw['game_id']!r}"
            )
        self.obs_schema = info.get("obs_schema")
        self.reward_schema = info.get("reward_schema")
        self.clock_mode = info.get("clock_mode")
        self.env_hash = info.get("env_hash")
        if self.clock_mode is not None and self.clock_mode != "lockstep-v1":
            raise EnvRejectedError(f"clock_mode 错配: 期望 lockstep-v1，得到 {self.clock_mode!r}")
        if self.expected_env_hash is not None and self.env_hash != self.expected_env_hash:
            raise EnvRejectedError(
                f"env_hash 错配: 期望 {self.expected_env_hash!r}，得到 {self.env_hash!r}"
            )
        return info

    # ---------- 请求 ----------
    def _request(self, op: str, *, timeout_s: Optional[float] = None, **extra: Any) -> Response:
        if op not in OPS:
            raise ProtocolError(f"未知 op: {op!r}")
        timeout = self.step_timeout_s if timeout_s is None else timeout_s
        req_id = self.request_id
        payload = {
            "protocol": PROTOCOL,
            "game_id": self.game_id,
            "session_id": self.session_id,
            "episode_id": self.episode_id,
            "request_id": req_id,
            "op": op,
        }
        if op == "reset":
            payload["seed"] = int(extra["seed"])
        elif op == "step":
            payload["action"] = int(extra["action"])
        elif op == "close":
            pass
        elif op == "hello":
            # hello 发生在 episode 开始前
            payload["episode_id"] = -1

        started = time.monotonic()
        self.connection.send_message(json.dumps(payload))

        stale = 0
        deadline = started + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.timeout_total += 1
                raise EnvTimeout(
                    f"等待 {op} 响应超时（{timeout}s, game={self.game_id}, "
                    f"episode={self.episode_id}, request={req_id}）"
                )
            try:
                raw_text = self.connection.recv_message(timeout_s=remaining)
            except EnvTimeout:
                # silent 注入会真的什么都不回；交给循环顶部统一报超时
                continue
            try:
                msg = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                self.bad_messages.append(raw_text[:512])
                raise ProtocolError(f"响应不是合法 JSON（已保存样本）: {exc}") from exc
            if not isinstance(msg, dict):
                self.bad_messages.append(raw_text[:512])
                raise ProtocolError("响应 JSON 根节点不是对象")
            _validate_response(msg)

            if msg["protocol"] != PROTOCOL:
                raise ProtocolError(f"响应协议版本错配: {msg['protocol']!r}")
            if msg["game_id"] != self.game_id:
                raise EnvRejectedError(f"响应 game_id 错配: {msg['game_id']!r}")

            # 配对三元组（§4.2 规则 1）
            if (msg["session_id"], msg["episode_id"], msg["request_id"]) == (
                self.session_id, payload["episode_id"], req_id
            ):
                self.request_id += 1
                return Response(
                    raw=msg,
                    received_at=time.monotonic(),
                    latency_s=time.monotonic() - started,
                    stale_dropped=stale,
                )
            # 旧会话/旧回合/旧请求：丢弃继续等，不静默当成当前结果
            if msg["session_id"] != self.session_id:
                stale += 1
                self.stale_total += 1
                continue
            if int(msg["request_id"]) < req_id or int(msg["episode_id"]) < payload["episode_id"]:
                stale += 1
                self.stale_total += 1
                continue
            raise StaleResponseError(
                f"响应配对失败: 期望 {(self.session_id, payload['episode_id'], req_id)}，"
                f"得到 {(msg['session_id'], msg['episode_id'], msg['request_id'])}"
            )

    # ---------- 环境操作 ----------
    def reset(self, seed: int, *, timeout_s: Optional[float] = None) -> Response:
        """开始新回合。返回 reset 后的第一条观测；不额外执行动作。"""
        self.episode_id += 1
        self.connection.drain()  # 清空旧队列，避免上一回合残留
        resp = self._request("reset", timeout_s=timeout_s, seed=int(seed))
        if resp.raw["termination_reason"] != "none":
            raise ProtocolError(f"reset 响应不应带终止原因: {resp.raw['termination_reason']!r}")
        if resp.raw["terminated"]:
            raise ProtocolError("reset 响应不应 terminated=true")
        self.reset_count += 1
        self.step_count = 0
        return resp

    def step(self, action: int) -> Response:
        """执行一次本地动作。"""
        if self.episode_id < 0:
            raise ProtocolError("step 之前必须先 reset")
        resp = self._request("step", action=int(action))
        self.step_count += 1
        return resp

    def close(self) -> None:
        """尽力通知客户端并关闭连接；失败不抛给上层。"""
        try:
            if self.connection.has_client:
                self._request("close", timeout_s=1.0)
        except Exception:
            pass
        self.connection.close()

    # ---------- 重连 ----------
    def reconnect(self) -> None:
        """重连生成新 session_id、清空旧队列（§4.2 规则 4）。

        调用方必须丢弃没有确认 next_obs 的最后一步，并在 reset 后重新开始；
        不盲目重发 step，避免执行两次。
        """
        self.session_id = f"s-{uuid.uuid4().hex[:12]}"
        self.episode_id = -1
        self.request_id = 0
        self.connection.drain()
        self.connect()


def hello_info_template() -> Dict[str, Any]:
    """假客户端构造 info 的辅助（测试与离线校验共用）。"""
    return {
        "clock_mode": "lockstep-v1",
        "env_hash": None,
        "obs_schema": "compact29-v1",
        "reward_schema": "bounded-v1",
        "python": platform.python_version(),
    }
