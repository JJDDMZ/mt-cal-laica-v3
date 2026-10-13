"""最小 stdlib WebSocket 客户端（测试用，不引入新依赖）。

只实现 RFC6455 里协议所需的部分：握手、客户端掩码文本帧、服务端帧解析
（含分片、ping/pong、close）。刻意不做重连与并发，只求可预测。
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
from typing import Any, Dict, Optional

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WSTimeout(TimeoutError):
    pass


DEFAULT_READ_TIMEOUT = 0.2  # 短读超时便于 stop() 快速中断轮询


class WSClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 17911, *, timeout: float = 5.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None
        self._buf = b""

    # ---------- 连接 ----------
    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET / HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        header = b""
        while b"\r\n\r\n" not in header:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("握手期间连接关闭")
            header += chunk
        head, _, rest = header.partition(b"\r\n\r\n")
        self._buf = rest
        if b"101" not in head.split(b"\r\n")[0]:
            raise ConnectionError(f"握手失败: {head[:200]!r}")

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self._send_frame(b"", opcode=0x8)
        except OSError:
            pass
        try:
            self.sock.close()
        finally:
            self.sock = None

    # ---------- 帧 ----------
    def _recv_exact(self, n: int) -> bytes:
        assert self.sock is not None
        while len(self._buf) < n:
            chunk = self.sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                raise ConnectionError("连接被关闭")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _send_frame(self, payload: bytes, opcode: int = 0x1) -> None:
        assert self.sock is not None
        header = bytearray([0x80 | opcode])
        n = len(payload)
        if n < 126:
            header.append(0x80 | n)
        elif n < (1 << 16):
            header.append(0x80 | 126)
            header += struct.pack("!H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack("!Q", n)
        mask = os.urandom(4)
        header += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def send(self, text: str) -> None:
        self._send_frame(text.encode("utf-8"), opcode=0x1)

    def send_json(self, obj: Any) -> None:
        self.send(json.dumps(obj))

    def recv(self) -> str:
        """读一条完整文本消息。"""
        fragments = b""
        while True:
            b0, b1 = self._recv_exact(2)
            fin = bool(b0 & 0x80)
            opcode = b0 & 0x0F
            masked = bool(b1 & 0x80)
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if masked else b""
            data = self._recv_exact(length) if length else b""
            if masked:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))

            if opcode == 0x8:
                raise ConnectionError("对端发送 close")
            if opcode == 0x9:  # ping → pong
                self._send_frame(data, opcode=0xA)
                continue
            if opcode == 0xA:
                continue
            fragments += data
            if fin:
                return fragments.decode("utf-8")

    def recv_json(self) -> Dict[str, Any]:
        return json.loads(self.recv())


def response(
    *,
    game_id: str,
    session_id: str,
    episode_id: int,
    request_id: int,
    obs: Dict[str, Any],
    score: float = 0.0,
    terminated: bool = False,
    termination_reason: str = "none",
    sim_ticks: int = 0,
    info: Optional[Dict[str, Any]] = None,
    protocol: str = "wemi-cal-v1",
) -> Dict[str, Any]:
    """构造一条合法响应的辅助函数。"""
    return {
        "protocol": protocol,
        "game_id": game_id,
        "session_id": session_id,
        "episode_id": episode_id,
        "request_id": request_id,
        "obs": obs,
        "score": score,
        "terminated": terminated,
        "termination_reason": termination_reason,
        "sim_ticks": sim_ticks,
        "info": info if info is not None else {},
    }
