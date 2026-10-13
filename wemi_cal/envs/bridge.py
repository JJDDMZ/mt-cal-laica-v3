"""WebSocket 连接层（技术方案 §4.1–§4.2，任务 T1）。

只负责：起一个绑定 127.0.0.1 的 WebSocket 服务端、等待一个客户端、收发文本消息。
不做请求配对、不做 schema 解释、不伪造数据——那是 session.py 的事。

必须修复原生 WemiEnv 的两个问题（§4.2）：
1. ``latest_msg`` 会被覆盖 → 改为线程安全队列。
2. ``while len(clients)==0: pass`` 纯自旋烧核 → 改为带超时的等待。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Optional

from websocket_server import WebsocketServer

logger = logging.getLogger(__name__)


class EnvTimeout(TimeoutError):
    """等待客户端消息/连接超时。绝不转成 reward=0 或 done=true。"""


class NoClientError(RuntimeError):
    """没有客户端连接时尝试发送。"""


class ConnectionClosed(RuntimeError):
    """客户端断开。"""


class Connection:
    """单端口 WebSocket 服务端 + 单客户端消息队列。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 17911,
        *,
        loglevel: int = logging.WARNING,
        poll_interval_s: float = 0.0005,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.poll_interval_s = float(poll_interval_s)
        self._server: Optional[WebsocketServer] = None
        self._messages: "queue.Queue[str]" = queue.Queue()
        # §4.1：同一端口只接受一个训练连接；后到的连接不进入 _active，也不收消息。
        self._active: Optional[dict[str, Any]] = None
        self._ignored_client_ids: set[Any] = set()
        self._lock = threading.Lock()
        self._client_connected = threading.Event()
        self._client_left = threading.Event()
        self._closed = False

    # ---------- 生命周期 ----------
    @property
    def bound_port(self) -> int:
        """实际绑定端口（port=0 时由系统分配，便于测试）。"""
        if self._server is None:
            return self.port
        try:
            return int(self._server.server_address[1])
        except Exception:  # pragma: no cover
            return self.port

    def create_server(self, loglevel: int = logging.WARNING) -> None:
        if self._server is not None:
            raise RuntimeError("server 已经创建")

        def _new_client(client: dict[str, Any], server: WebsocketServer) -> None:
            with self._lock:
                accept = self._active is None
                if accept:
                    self._active = client
                else:
                    self._ignored_client_ids.add(client["id"])
            if accept:
                self._client_left.clear()
                self._client_connected.set()
                logger.info("客户端已连接: %s", client.get("address"))
            else:
                # §4.1：连错游戏/重复连接必须被识别；这里只接受第一个连接。
                logger.warning("第二个客户端连接，已忽略: %s", client.get("address"))

        def _left_client(client: dict[str, Any], server: WebsocketServer) -> None:
            # 关闭过程中 websocket_server 可能传入已置空的 client；不因此抛到日志里
            if not client or "id" not in client:
                return
            with self._lock:
                self._ignored_client_ids.discard(client["id"])
                was_active = self._active is not None and client["id"] == self._active["id"]
                if was_active:
                    self._active = None
            if was_active:
                self._client_connected.clear()
                self._client_left.set()
                logger.info("客户端已断开")

        def _received(client: dict[str, Any], server: WebsocketServer, message: str) -> None:
            # 队列不丢消息；不覆盖（对照原实现的 latest_msg）
            self._messages.put(message)

        self._server = WebsocketServer(self.host, self.port, loglevel=loglevel)
        self._server.set_fn_new_client(_new_client)
        self._server.set_fn_client_left(_left_client)
        self._server.set_fn_message_received(_received)
        self._server.run_forever(threaded=True)
        # run_forever(threaded=True) 立即返回，等 bind 生效
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if getattr(self._server, "server", None) is not None or self.bound_port:
                break
            time.sleep(0.01)

    def waiting_client(self, timeout_s: float = 60.0) -> None:
        """等待第一个客户端连接；超时抛 EnvTimeout（§4.2 规则 2）。"""
        if self._server is None:
            raise RuntimeError("先调用 create_server()")
        if not self._client_connected.wait(timeout=timeout_s):
            raise EnvTimeout(f"等待客户端连接超时（{timeout_s}s，端口 {self.bound_port}）")

    @property
    def has_client(self) -> bool:
        return self._client_connected.is_set()

    @property
    def client_left_since_connect(self) -> bool:
        return self._client_left.is_set()

    # ---------- 收发 ----------
    def send_message(self, message: str) -> None:
        if self._server is None:
            raise RuntimeError("先调用 create_server()")
        with self._lock:
            client = self._active
        if client is None:
            raise NoClientError("没有客户端连接，无法发送")
        self._server.send_message(client, message)

    def recv_message(self, timeout_s: float = 5.0) -> str:
        """取下一条消息；超时抛 EnvTimeout（不返回 None 让上层死循环）。"""
        try:
            return self._messages.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise EnvTimeout(f"等待客户端消息超时（{timeout_s}s）") from exc

    def drain(self) -> int:
        """丢弃尚未消费的消息，返回丢弃条数（重连时清空旧队列，§4.2 规则 4）。"""
        n = 0
        while True:
            try:
                self._messages.get_nowait()
                n += 1
            except queue.Empty:
                return n

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            try:
                self._server.shutdown_gracefully()
            except Exception:  # pragma: no cover - 关闭失败不影响上层
                try:
                    self._server.shutdown()
                except Exception:
                    pass
            self._server = None
        with self._lock:
            self._active = None
            self._ignored_client_ids.clear()
        self._client_connected.clear()
