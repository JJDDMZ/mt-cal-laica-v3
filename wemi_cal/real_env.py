"""原版（实时）协议的适配器：让 wemi_cal 的训练流程能直接吃 WemiEnv 老接口。

用户已确认：**按原来训练成功的方案跑，不做 lockstep 改动**。因此这里：

- 直接复用原版 Python 游戏类（``WemiEnv/WemiEnv/Flappybird.py`` 等），
  它们用 ``Connection`` + 消息 ``{action, restart}``，客户端回 ``{obs, reward, done}``；
- 服务端监听 **7911**（原版客户端硬编码的端口）→ **游戏 JS 一个字节都不用改**；
- 客户端自带 ``setInterval(40ms)`` 物理与 100ms 延迟回包 → 保持原速（不加速）。

只做两件必要的翻译：
1. 把原版**扁平观测**补成主 schema 的命名字段（F 3 维、T 34 维 → 29 维统一输入）；
2. 把原版的 ``reward/done`` 翻译成 bounded-v1 训练奖励与终止语义。

注意：原版 F 观测缺管道横坐标，T 观测带 3×11+1 结构——补名时会如实置掩码，
不伪造缺失维度。
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .envs import features as F
from .envs.game_env import EnvStepInfo

# 原版客户端硬编码的端口
LEGACY_PORT = 7911

# 原版 obs 的维度（真实值，不是文件里声明的错误值）
LEGACY_OBS_DIM = {"Flappybird": 5, "Timberman": 34}   # F 已扩展为 5 项：+ px, score


class LegacyProtocolError(RuntimeError):
    """原版协议回包不合法（缺字段/NaN/维度不对）。"""


class ClientGoneError(LegacyProtocolError):
    """小游戏客户端掉线（多半是又按了一次「编译」）。

    原版客户端是**纯反应式**的：只在收到服务端消息后才回一帧，而且**没有重连**。
    所以任何一次重新编译 / 关闭项目 / 切换项目都会杀掉 socket，当前回合的状态
    （鸟的位置、管道、计时）全部丢失，无法续跑 —— 只能作废这一回合。
    旧实现会在这里静默干等满 action_timeout（30s），让人误判成「没连上」。
    """


# 每个游戏**必须**打开的项目目录（这是唯一会让实验作废的操作性错误）
REQUIRED_PROJECT_DIR: Dict[str, str] = {
    "Flappybird": "wechat_games_rt/Flappybird",
    "Timberman": "wechat_games/Timberman",
}


def operator_hint(game: str) -> str:
    """给操作者的启动提示。写清楚"别按第二次编译"，因为那会杀死当前回合。"""
    d = REQUIRED_PROJECT_DIR.get(game, f"wechat_games_rt/{game}")
    return (
        f"操作提示（{game}，违反任意一条实验就作废）：\n"
        f"  1) 开发者工具里打开的**必须是** {d}\n"
        f"     —— 不要开 wechat_games/{game}（那是未扩展的原版，"
        f"{'不发 score/pipe_x' if game == 'Flappybird' else '不发 blood'}）；\n"
        f"  2) 只按**一次**「编译」，之后整个训练/评测期间不要再按、不要关项目、不要切项目；\n"
        f"     （客户端纯反应式且无重连：一次重编译 = 当前回合作废）\n"
        f"  3) 先起 Python（已在监听 {LEGACY_PORT}），再点编译。"
    )


def _load_legacy_module(game: str, wemi_env_root: Path) -> Any:
    """加载原版游戏模块（它内部 import interact，所以要把 WemiEnv/WemiEnv 加进 sys.path）。"""
    # 仓库根是 WemiEnv/，游戏模块在它下面的 WemiEnv/ 子目录里
    pkg = wemi_env_root / "WemiEnv"
    if not pkg.is_dir():
        raise FileNotFoundError(f"找不到原版 WemiEnv 源码: {pkg}")
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    return importlib.import_module(game)


@dataclass
class LegacyReply:
    obs: List[float]
    reward: float
    done: bool


class RealTimeGameEnv:
    """原版实时环境：负责起服务端、等连接、发动作、收观测。

    它**不**假设 lockstep：物理由客户端自己的 setInterval 驱动，与原版一致。
    """

    def __init__(self, game: str, *, wemi_env_root: Path, port: int = LEGACY_PORT,
                 horizon: int = 1000, connect_timeout_s: float = 900.0,
                 action_timeout_s: float = 30.0,
                 disconnect_grace_s: float = 2.0) -> None:
        if game not in F.GAMES:
            raise ValueError(f"未知游戏: {game!r}")
        self.game = game
        self.horizon = int(horizon)
        self.port = int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self.action_timeout_s = float(action_timeout_s)
        # 客户端掉线后先给这么多秒（可能是瞬时抖动）；超过就直接判定"掉线"。
        # 不给太长的原因：原版客户端没有重连，重新出现的客户端一定是新实例，
        # 回合状态已经丢了，继续等只会浪费几百秒。
        self.disconnect_grace_s = float(disconnect_grace_s)

        self._module = _load_legacy_module(game, Path(wemi_env_root))
        self._env = None
        self._episode_id = -1
        self._steps = 0
        self._score_raw = 0.0
        self._episode_return = 0.0
        self._last_error: Optional[str] = None
        # 原版一次只能有一个未完成请求；这里记录未确认的最后一步以便丢弃
        self._pending = False

    def operator_hint(self) -> str:
        return operator_hint(self.game)

    # ---------------------------------------------------------------- 连接
    def start_server(self) -> None:
        """按原版方式创建服务端并等待客户端。原版类在 __init__ 里就会 create_server。"""
        # 类名不统一：Flappybird 是 FlappybirdEnv，Timberman 直接叫 Timberman。
        # 依次尝试 {game}Env / {game} / 唯一以 Env 结尾的类，不猜错也不静默兜底。
        cls = getattr(self._module, f"{self.game}Env", None) or getattr(self._module, self.game, None)
        if cls is None:
            cands = [v for k, v in vars(self._module).items()
                     if isinstance(v, type) and (k.endswith("Env") or k == self.game)]
            if len(cands) != 1:
                raise LegacyProtocolError(
                    f"{self.game} 模块里找不到环境类（候选: {[c.__name__ for c in cands]}）")
            cls = cands[0]
        self._env_cls_name = cls.__name__
        print(f"[legacy] {self.game} 环境类={cls.__name__}，启动原版服务端，端口 {self.port}，等待客户端…", flush=True)
        print(self.operator_hint(), flush=True)
        t0 = time.monotonic()
        # 原版 Connection.create_server 的默认端口是 7891，需要注入我们自己的端口
        self._patch_bind_port()
        self._env = cls()
        waited = time.monotonic() - t0
        print(f"[legacy] 客户端已连接（{waited:.1f}s）", flush=True)

    def _patch_bind_port(self) -> None:
        """把原版 Connection.create_server 的默认端口改成我们要用的端口。

        原版六个游戏都把 7891 硬编码在默认参数里，只能从这里注入。
        """
        conn_cls = getattr(self._module, "Connection", None)
        if conn_cls is None:
            raise LegacyProtocolError(f"{self.game} 模块里没有 Connection 类")
        if getattr(conn_cls, "_wemi_cal_port_patched", False):
            conn_cls._wemi_cal_port = self.port
            return
        original = conn_cls.create_server
        target_port = self.port

        def create_server(self, host="0.0.0.0", port=None, loglevel=logging.INFO):
            use = target_port if port in (None, 7891) else port
            return original(self, host, use, loglevel)

        conn_cls.create_server = create_server
        conn_cls._wemi_cal_port_patched = True
        conn_cls._wemi_cal_port = target_port

    @property
    def connected(self) -> bool:
        if self._env is None:
            return False
        conn = getattr(self._env, "conn", None)
        if conn is None or conn.server is None:
            return False
        return len(conn.server.clients) > 0

    def reconnect(self) -> None:
        """掉线后重建（原版没有重连语义，只能重新建服务端并等客户端）。"""
        print("[legacy] 尝试重连：重建服务端并等待客户端…", flush=True)
        self.close()
        self._episode_id = -1
        self._steps = 0
        self._pending = False
        self.start_server()

    def close(self) -> None:
        conn = getattr(self._env, "conn", None) if self._env is not None else None
        if conn is not None and conn.server is not None:
            try:
                conn.server.shutdown_gracefully()
            except Exception:
                try:
                    conn.server.shutdown()
                except Exception:
                    pass
        self._env = None

    # ---------------------------------------------------------------- 收发
    def _send(self, message: Dict[str, Any]) -> None:
        conn = self._env.conn
        if not conn.server.clients:
            raise LegacyProtocolError("没有客户端连接")
        conn.send_message(json.dumps(message))
        self._pending = True

    def _recv(self, timeout_s: float) -> LegacyReply:
        """原版 recv_message 是非阻塞的；这里加真超时 + 掉线快速失败。

        掉线检测很关键：原版客户端没有重连，一次「编译」就会让当前回合永远等不到
        回包。旧实现会干等到 action_timeout（30s）并报一个含糊的 TimeoutError，
        看上去像"没连上"；这里改成独立的 ClientGoneError 并说清原因。
        """
        conn = self._env.conn
        deadline = time.monotonic() + timeout_s
        gone_since: Optional[float] = None
        while True:
            data = conn.recv_message()
            if data is not None:
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise LegacyProtocolError(f"回包不是合法 JSON: {exc}") from exc
                for key in ("obs", "reward", "done"):
                    if key not in msg:
                        raise LegacyProtocolError(f"回包缺少字段 {key!r}: {msg}")
                obs = msg["obs"]
                if not isinstance(obs, list):
                    raise LegacyProtocolError(f"obs 不是列表: {type(obs).__name__}")
                # 原版 T 的观测是**混合类型**：[树方位, null, null, ...]，
                # 树方位是 true/false/"center"。不能整体转 float，
                # 否则 np.array 会把整个列表退化成字符串数组（正是 wemi_mt 踩过的坑）。
                # 这里只校验"能转成浮点的项"，null 与字符串留给按游戏解码处理。
                for i, v in enumerate(obs):
                    if v is None or isinstance(v, str):
                        continue
                    try:
                        f = float(v)
                    except (TypeError, ValueError) as exc:
                        raise LegacyProtocolError(f"obs[{i}] 不是数值/null/字符串: {v!r}") from exc
                    if not np.isfinite(f):
                        raise LegacyProtocolError(f"obs[{i}] 是 NaN/Inf")
                reward = float(msg["reward"])
                if not np.isfinite(reward):
                    raise LegacyProtocolError("reward 含 NaN/Inf")
                self._pending = False
                return LegacyReply(obs=list(obs), reward=reward, done=bool(msg["done"]))
            if not self.connected:
                now = time.monotonic()
                if gone_since is None:
                    gone_since = now
                elif now - gone_since >= self.disconnect_grace_s:
                    raise ClientGoneError(
                        f"{self.game} 客户端在第 {self._episode_id} 回合第 "
                        f"{self._steps} 步掉线（已等 {self.disconnect_grace_s:.1f}s 仍未回来）。"
                        f"最可能的原因：又按了一次「编译」/关闭了项目/切换了项目"
                        f"——客户端纯反应式且无重连，重新编译会杀掉 socket 并丢失回合状态。"
                        f"请重新打开 {REQUIRED_PROJECT_DIR.get(self.game)}，"
                        f"按一次「编译」后不要再动，然后重跑。")
            else:
                gone_since = None
            if time.monotonic() > deadline:
                raise TimeoutError(f"等待原版回包超时（{timeout_s:.1f}s）")
            time.sleep(0.0005)   # 原版是纯自旋；这里加节流，避免烧满一个核

    def reset(self, seed: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """原版 reset：action=3 + restart=True。"""
        if self._env is None:
            raise LegacyProtocolError("服务端尚未启动")
        # 原版的 reset 自己循环收包，这里复用它的实现保证语义一致
        self._send({"action": 3, "restart": True})
        reply = self._recv(self.action_timeout_s)
        self._episode_id += 1
        self._steps = 0
        self._score_raw = 0.0
        self._episode_return = 0.0
        self._last_error = None
        self._last_obs = reply.obs
        self._score_peak = 0.0
        x = self._encode(reply.obs)
        return x, {"episode_id": self._episode_id, "score_raw": 0.0,
                   "raw_obs": reply.obs}

    def step(self, local_action: int) -> Tuple[np.ndarray, float, bool, bool, EnvStepInfo]:
        """原版 step：action=0/1。奖励与终止在 Python 单点计算（§4.2）。"""
        if self._env is None:
            raise LegacyProtocolError("服务端尚未启动")
        if local_action not in (0, 1):
            raise ValueError(f"{self.game} 本地动作必须是 0/1，得到 {local_action}")
        t0 = time.monotonic()
        self._send({"action": int(local_action), "restart": False})
        reply = self._recv(self.action_timeout_s)
        latency = time.monotonic() - t0

        self._steps += 1
        prev_score = self._score_raw
        # 先解码观测：F 的真实分数就在 obs 里，必须在使用前解析（否则永远读到上一帧）
        x = self._encode(reply.obs)
        # 原版 F 的 reward 是间隙距离 shaping；T 的 reward 是累计分数
        if self.game == "Timberman":
            self._score_raw = float(reply.reward)     # 原版 T 回传累计分
        elif getattr(self, "_native_score", None) is not None:
            self._score_raw = float(self._native_score)   # F 扩展后的真实分数
        else:
            self._score_raw = prev_score              # F 原版不发分数（兼容旧客户端）
        # 峰值分数：T 的累计分在 reward 里逐帧给出，回合结束时最后一帧未必是最大值，
        # 因此按峰值统计（这是评测 T 成绩的正确口径）。
        self._score_peak = max(getattr(self, "_score_peak", 0.0), self._score_raw)
        died = bool(reply.done)

        # 终止语义：死亡（原版 done）或到达任务定义的 H
        if died:
            reason = "ground" if self.game == "Flappybird" else "collision"
            terminated, truncated = True, False
        elif self._steps >= self.horizon:
            reason = "horizon"
            terminated, truncated = True, False
        else:
            reason, terminated, truncated = "none", False, False

        reward = self._train_reward(reply, died=died, prev_score=prev_score)
        self._episode_return += reward
        info = EnvStepInfo(
            game=self.game, steps_taken=self._steps,
            remaining_budget=(self.horizon - self._steps) / self.horizon,
            score_raw=self._score_peak, reward_train=reward,
            terminated=terminated, truncated=truncated,
            termination_reason=reason, died=died,
            sim_ticks=self._steps,          # 原版无 tick 概念；用步数代替并如实标注
            latency_s=latency, episode_return_train=self._episode_return,
            episode_id=self._episode_id,
        )
        return x, reward, terminated, truncated, info

    # ---------------------------------------------------------------- 翻译
    def _train_reward(self, reply: LegacyReply, *, died: bool, prev_score: float) -> float:
        """把原版 reward 翻成 bounded-v1 训练奖励（§5.5）。

        - F：原版回传的就是间隙距离 shaping（可能为负），clip 到 [-1,1]；死亡 -1；
        - T：原版回传累计分，用增量；死亡额外 -1。
        """
        if self.game == "Flappybird":
            if died:
                return -1.0
            # 原版 shaping 以像素为单位（量级几十），按 60 缩放到 [-1,1] 附近
            return float(np.clip(reply.reward / 60.0, -1.0, 1.0))
        if died:
            return -1.0
        return float(np.clip(self._score_raw - prev_score, -1.0, 1.0))

    def _encode(self, raw_obs: List[float]) -> np.ndarray:
        """把原版扁平观测补成 29 维主 schema。

        F（3 维）: [bird_y, gap_bottom, gap_top] → 名 4 个字段，但 pipe_x **未知**，
                   因此把它置 0 并把 mask 的第 4 位保持为 1（位置真实存在，只是原版没给值）。
        T（34 维）: [11×(posiDr, x, y) + npc.posi] → 取 posiDr 得 11 维 + npc + blood(未知)
        """
        named = self._to_named(raw_obs)
        return F.encode_observation(
            self.game, named, steps_taken=self._steps, horizon=self.horizon)

    def _to_named(self, raw: List[float]) -> Dict[str, Any]:
        if self.game == "Flappybird":
            # 必须是扩展后的 5 项：+ pipe_x + 真实 score。
            # 只接受 3 项（原版）会**静默伪造**：pipe_x 被填 0（掩码却仍是 1），
            # score_raw 恒为 0 → 评测输出"0 分/0 通过"，看起来像"模型学废了"。
            # 这个坑真实发生过：操作者打开的是未扩展的 wechat_games/Flappybird。
            if len(raw) != LEGACY_OBS_DIM["Flappybird"]:
                raise LegacyProtocolError(
                    f"F 观测应为 {LEGACY_OBS_DIM['Flappybird']} 项（含 pipe_x 与真实 score），"
                    f"实得 {len(raw)} 项。这几乎总是因为开发者工具里打开的是**未扩展的"
                    f"原版目录 wechat_games/Flappybird**。请改开 "
                    f"{REQUIRED_PROJECT_DIR['Flappybird']} 后重试"
                    f"（只按一次「编译」，之后不要动）。")
            bird_y, gap_bottom, gap_top = float(raw[0]), float(raw[1]), float(raw[2])
            world_h, world_w = F.WORLD_HEIGHT, F.WORLD_WIDTH
            named = {
                "bird_y": bird_y / world_h,
                "gap_top": gap_top / world_h,
                "gap_bottom": gap_bottom / world_h,
                "pipe_x": float(raw[3]) / world_w,
            }
            # raw[4] 是游戏真实累计分数（过一组管道 +1）
            self._native_score = float(raw[4])
            return named
        # Timberman：34 = 11 棵树 × (posiDr, x, y) + npc.posi，
        # 其中 x/y 在真实数据里是 null（客户端不发）。posiDr 为 true/false/"center"。
        if len(raw) != LEGACY_OBS_DIM["Timberman"]:
            raise LegacyProtocolError(f"T 观测应为 34 维，得到 {len(raw)} 维")
        sides: List[float] = []
        for i in range(11):
            v = raw[i * 3]
            # §5.2 的编码：false → -1，'center' → 0，true → +1
            if v is True or v == 'true' or v == 1:
                sides.append(F.SIDE_RIGHT)
            elif v is False or v == 'false' or v == 0:
                sides.append(F.SIDE_LEFT)
            elif v == 'center':
                sides.append(F.SIDE_CENTER)
            else:
                raise LegacyProtocolError(
                    f"第 {i} 棵树的方位无法识别: {v!r}（拒绝猜测）")
        npc_raw = raw[33]
        if npc_raw is True or npc_raw == 'true' or npc_raw == 1:
            npc_code = F.SIDE_RIGHT
        elif npc_raw is False or npc_raw == 'false' or npc_raw == 0:
            npc_code = F.SIDE_LEFT
        else:
            raise LegacyProtocolError(f"npc 方位无法识别: {npc_raw!r}")
        # blood 原版不发（画布专用）→ 置满值，不伪造变化
        return {"features": sides + [npc_code, 1.0]}
