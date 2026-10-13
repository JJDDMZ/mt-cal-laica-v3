"""JS↔Python 桥接验证的服务端 + 驱动侧（任务 T2 的"无模拟器真机替代品"）。

角色分工与真机完全一致：

    [Python 驱动]  --请求-->  [wemi-cal-v1 桥接]  --请求-->  [JS 游戏运行时]
                   <--响应--                  <--响应--

- **驱动侧**用 ``Session``（真正的协议实现）发 hello/reset/step，并校验每条响应的
  五元组、字段类型、枚举、lockstep 语义；
- **JS 侧**用 ``wechat_games_cal`` 的运行时执行真实物理并回包；
- 本模块既可以当驱动跑（默认），也支持只当服务端跑（``--server-only``）。

这样在打开微信开发者工具之前，就能确认"Python 协议层 ↔ JS 游戏层"整条链路正确。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wemi_cal.envs import bridge, features as F
from wemi_cal.envs.session import Session

FAIL: List[str] = []


def ok(name: str, cond: bool, detail: str = "") -> bool:
    if cond:
        print(f"ok    {name}")
    else:
        FAIL.append(f"{name}  [{detail}]")
        print(f"FAIL  {name}  [{detail}]")
    return bool(cond)


# ---------------------------------------------------------------- 驱动
def drive(
    game: str,
    *,
    port: int,
    env_hash: str,
    play_steps: int = 12,
    replay_actions: Tuple[int, ...] = (1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0),
    replay_seed: int = 777,
) -> int:
    host = "127.0.0.1"
    conn = bridge.Connection(host=host, port=port)
    conn.create_server()
    bound = conn.bound_port
    print(f"PORT={bound}", flush=True)

    sess = Session(
        connection=conn,
        game_id=game,
        step_timeout_s=20.0,
        connect_timeout_s=60.0,
        expected_env_hash=env_hash,
    )
    try:
        info = sess.connect()
        ok(f"{game} 握手成功（协议/game_id/env_hash/clock_mode 全部匹配）", True)
        ok(f"{game} info 带 obs_schema/reward_schema",
           info.get("obs_schema") == "compact29-v1" and info.get("reward_schema") == "bounded-v1",
           json.dumps(info, ensure_ascii=False))

        # ---- reset ----
        r0 = sess.reset(seed=12345)
        ok(f"{game} reset 不带终止原因", r0.termination_reason == "none" and not r0.terminated)
        ok(f"{game} reset 后 sim_ticks=0", r0.sim_ticks == 0, str(r0.sim_ticks))
        _check_obs_schema(game, r0.obs)

        # ---- 逐步执行：记录 tick 推进 ----
        max_ticks = 3 if game == "Flappybird" else 6
        prev_ticks = r0.sim_ticks
        overlarge = 0
        for i in range(play_steps):
            r = sess.step(i % 2)
            dt = r.sim_ticks - prev_ticks
            if dt < 0 or dt > max_ticks:
                overlarge += 1
            prev_ticks = r.sim_ticks
            if r.terminated:
                break
        ok(f"{game} 每次 step 推进不超过 {max_ticks} 个 tick", overlarge == 0,
           f"越界次数={overlarge}")

        # ---- 打到死亡，核对终止原因与"死后 tick 不再增长" ----
        died = None
        for _ in range(2000):
            r = sess.step(0 if game == "Flappybird" else 1)
            if r.terminated:
                died = r
                break
        ok(f"{game} 最终能 terminated", died is not None)
        if died is not None:
            allowed = ({"ground", "collision"} if game == "Flappybird"
                       else {"collision", "blood"})
            ok(f"{game} 终止原因属于允许集合", died.termination_reason in allowed,
               died.termination_reason)
            after = sess.step(0)
            ok(f"{game} 死后 sim_ticks 不再增长",
               after.sim_ticks == died.sim_ticks,
               f"{died.sim_ticks} -> {after.sim_ticks}")

        # ---- 可重放：同 seed + 同动作序列 ----
        def replay(seed: int) -> str:
            rr = sess.reset(seed=seed)
            traj: List[Any] = [rr.obs]
            for a in replay_actions:
                st = sess.step(a)
                traj.append(st.obs)
                if st.terminated:
                    break
            return json.dumps(traj, sort_keys=True, default=str)

        t1 = replay(replay_seed)
        t2 = replay(replay_seed)
        ok(f"{game} 同 seed + 同动作序列完全可重放", t1 == t2)
        t3 = replay(replay_seed + 1)
        ok(f"{game} 不同 seed 轨迹不同", t1 != t3)

        # ---- 非法动作必须被拒绝，而不是 clamp 成一个合法动作 ----
        # 在一个干净回合里验证：非法动作既不改变游戏状态，也必须回一条明确错误
        # （而不是静默超时——静默超时在真机上会把根因藏起来）。
        sess.reset(seed=4242)
        before = sess.step(0)                      # 先往前走一步拿到基线
        illegal = 2 if game == "Flappybird" else 7
        bad = sess._request("step", action=illegal, timeout_s=5.0)
        ok(f"{game} 非法动作返回明确错误（不 clamp）", bool(bad.raw.get("error")),
           json.dumps({k: v for k, v in bad.raw.items() if "error" in k},
                      ensure_ascii=False)[:200])
        ok(f"{game} 错误响应仍保留配对字段",
           bad.raw.get("request_id") is not None and bad.raw.get("game_id") == game)
        ok(f"{game} 非法动作被标记为非法请求而非逻辑崩溃",
           bad.raw.get("error_kind") == "illegal_request",
           str(bad.raw.get("error_kind")))
        after = sess.step(0)                       # 非法动作没有执行 → tick 应从 before 正常继续
        step_span = after.sim_ticks - before.sim_ticks
        max_ticks = 3 if game == "Flappybird" else 6
        ok(f"{game} 非法动作后逻辑没有被执行两次（tick 正常推进）",
           0 <= step_span <= max_ticks,
           f"before={before.sim_ticks} illegal={bad.sim_ticks} after={after.sim_ticks}")

        try:
            sess.close()
        except Exception:
            pass
    finally:
        conn.close()

    print(f"\n{game}: {len(FAIL)} failures")
    return 1 if FAIL else 0


def _check_obs_schema(game: str, obs: Dict[str, Any]) -> None:
    if game == "Flappybird":
        keys = ["bird_y", "gap_top", "gap_bottom", "pipe_x"]
        ok("F 观测是 4 个命名字段",
           all(isinstance(obs.get(k), (int, float)) for k in keys),
           json.dumps(obs, ensure_ascii=False))
        ok("F reset 后 bird_y 在 (0,1)", 0 < float(obs["bird_y"]) < 1, str(obs["bird_y"]))
        ok("F reset 后 pipe_x=0（沿用源码 px 初值）", float(obs["pipe_x"]) == 0.0,
           str(obs["pipe_x"]))
    else:
        feats = obs.get("features")
        ok("T 观测是 13 项 features", isinstance(feats, list) and len(feats) == 13,
           f"len={len(feats) if isinstance(feats, list) else None}")
        if isinstance(feats, list) and len(feats) == 13:
            ok("T 树方位在 {-1,0,1}",
               all(v in (-1, 0, 1) for v in feats[:11]), json.dumps(feats[:11]))
            ok("T npc 方位是 ±1", feats[11] in (-1, 1), str(feats[11]))
            ok("T blood/6000 在 [0,1]", 0 <= feats[12] <= 1, str(feats[12]))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="JS↔Python 桥接驱动")
    ap.add_argument("--game", required=True, choices=("Flappybird", "Timberman"))
    ap.add_argument("--env-hash", default=None)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--server-only", action="store_true")
    args = ap.parse_args(argv)
    env_hash = args.env_hash or f"{args.game.lower()}-lockstep-v1"
    if args.server_only:
        # 只当服务端：交给 JS 侧自己发请求（保留给需要单向验证的场景）
        from .js_bridge_server import serve
        return serve(args.game, env_hash, port=args.port)
    return drive(args.game, port=args.port, env_hash=env_hash)


if __name__ == "__main__":
    raise SystemExit(main())
