"""正式训练 CLI：一个阶段一个游戏（技术方案 §4.1/§6/§9）。

    python -m wemi_cal.run_real --game Flappybird --steps 50000 --run-root runs_real/c-strict
    python -m wemi_cal.run_real --game Timberman  --steps 50000 --run-root runs_real/c-strict \\
        --resume runs_real/c-strict/stage_final.pt --no-initial-offline

为什么一次只跑一个游戏：微信模拟器一次只应开一个项目。两个游戏各自占端口、
各自编译，同时开会互相争抢机器且难以判断状态。因此双任务按阶段串行，
阶段切换需要人工换项目（方案 §9.1 也要求"按当前阶段只启动对应游戏的训练连接"）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from .config import ConfigError, load_config, validate_config
from .real_train import (
    DEFAULT_PORTS, ENV_HASH, PROJECTS, RealTrainer, make_real_env, open_project,
)


def build_cfg(args: argparse.Namespace):
    import yaml
    from pathlib import Path as P

    raw = yaml.safe_load(P(args.config).read_text(encoding="utf-8"))
    raw["variant"] = "continuous"
    raw["mode"] = "full"
    raw["seed"] = args.seed
    raw["schedule"] = args.schedule
    if args.steps:
        # 允许用 test_only 缩小预算跑 pilot；正式跑用原预算
        raw["train"]["stage_schedules"]["strict"] = [
            {"game": "Flappybird", "steps": args.steps // 2},
            {"game": "Timberman", "steps": args.steps // 2},
        ]
        raw["train"]["stage_schedules"]["rehearsal"] = [
            {"game": "Flappybird", "steps": args.steps // 2},
            {"game": "Timberman", "steps": args.steps // 2},
            {"game": "Flappybird", "steps": 0}, {"game": "Timberman", "steps": 0},
            {"game": "Flappybird", "steps": 0}, {"game": "Timberman", "steps": 0},
        ]
    test_only = bool(args.steps) or bool(args.first_data) or bool(args.repr_updates)
    if args.first_data:
        raw["train"]["random_steps_first_visit"] = args.first_data
    if args.repr_updates:
        raw["train"]["repr_updates_first_visit"] = args.repr_updates
    if args.offline_updates is not None:
        raw["train"]["initial_offline_updates"] = max(1, args.offline_updates)
    return validate_config(raw, test_only=test_only)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MT-Cal-LAICA 正式训练（单阶段）")
    ap.add_argument("--config", default="wemi_cal/configs/mt_cal.yaml")
    ap.add_argument("--game", required=True, choices=("Flappybird", "Timberman"))
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None,
                    help="本阶段总步数；默认用配置里的阶段预算")
    ap.add_argument("--schedule", default="strict", choices=("strict", "rehearsal"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--resume", default=None, help="接上一阶段的 checkpoint")
    ap.add_argument("--no-initial-offline", action="store_true",
                    help="非首个任务：不做初始离线预训练（§6.5）")
    ap.add_argument("--first-data", type=int, default=None, help="首次随机采集步数")
    ap.add_argument("--repr-updates", type=int, default=None)
    ap.add_argument("--offline-updates", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--save-every", type=int, default=5000,
                    help="每多少在线步落一次周期 checkpoint（0=关闭）")
    ap.add_argument("--reconnect-timeout", type=float, default=300.0,
                    help="掉线后等待模拟器重连的秒数")
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--no-devtools-open", action="store_true",
                    help="不自动打开项目（项目已开着时用）")
    ap.add_argument("--workspace", default=".")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    cfg = build_cfg(args)
    port = args.port or DEFAULT_PORTS[args.game]
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    stage_steps = args.steps or cfg.steps_for_game(args.game)
    print(f"=== 阶段开始: game={args.game} port={port} steps={stage_steps} "
          f"test_only={cfg.test_only} ===", flush=True)

    # 1) 打开项目（可选）
    if not args.no_devtools_open:
        open_project(args.game, workspace)

    # 2) 起服务端 + 等连接
    bundle = make_real_env(cfg, args.game, port=port,
                           connect_timeout_s=float(cfg.env["connect_timeout_s"]))
    trainer = RealTrainer(cfg, args.game, bundle, run_root=run_root, seed=args.seed)

    # 3) 接上一阶段
    if args.resume:
        restored = __import__("wemi_cal.checkpoint", fromlist=["x"]).load_checkpoint(
            args.resume, cfg=cfg, model=trainer.model, restore_replay=True)
        trainer.replay = restored.get("replay", trainer.replay)
        ts = restored.get("trainer_state", {})
        trainer.mapping_version = int(ts.get("mapping_version", 0))
        print(f"[resume] 已恢复 mapping_version={trainer.mapping_version}", flush=True)

    first_visit = args.resume is None
    random_steps = int(cfg.train["random_steps_first_visit"])
    repr_updates = int(cfg.train["repr_updates_first_visit"])
    offline_updates = int(cfg.train["initial_offline_updates"])

    t0 = time.monotonic()
    try:
        if first_visit:
            print(f"[stage] 首次接入：随机采集 {random_steps} 步", flush=True)
            trainer.collect_first_data(random_steps)
            print(f"[stage] 表示适配 {repr_updates} updates", flush=True)
            trainer.adapt(repr_updates)
            if not args.no_initial_offline:
                print(f"[stage] 初始离线 {offline_updates} updates", flush=True)
                trainer.initial_offline(offline_updates)
            online_steps = stage_steps - random_steps
        else:
            # 返回已知游戏：不重新适配、不重新初始化（§9.2 R3–R6）
            print("[stage] 非首次接入：直接在线，不重做适配", flush=True)
            online_steps = stage_steps

        if online_steps <= 0:
            raise ConfigError(f"在线步数必须为正，得到 {online_steps}")

        # 评测用独立进程内环境不可行（真实游戏只有一路连接），因此在线阶段不插评测，
        # 改为按固定交互步数保存候选 checkpoint，由 evaluate 单独跑（§9.4）。
        trainer.online(
            online_steps,
            update_every=int(cfg.train["update_every_online_steps"]),
            eval_env=None, eval_every=args.eval_every, eval_episodes=args.eval_episodes,
            save_every=args.save_every, reconnect_timeout_s=args.reconnect_timeout)

        ckpt = trainer.save(run_root / f"stage_{args.game}.pt", stage=args.game)
        summary = {
            "game": args.game, "steps": stage_steps, "port": port,
            "wall_clock_s": time.monotonic() - t0,
            "stats": {k: v for k, v in trainer.stats.items() if k != "latencies"},
            "latency": trainer._latency_summary(),
            "checkpoint": str(ckpt),
            "replay": trainer.replay.diagnostics(),
            "mapping_version": trainer.mapping_version,
        }
        (run_root / f"stage_{args.game}_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\n=== 阶段完成: {args.game} ===", flush=True)
        print(json.dumps({"latency": summary["latency"],
                          "online_steps": summary["stats"]["online_steps"],
                          "learner_updates": summary["stats"]["learner_updates"],
                          "episodes": len(summary["stats"]["episodes"])},
                         indent=2, ensure_ascii=False), flush=True)
        scores = [e["score_raw"] for e in trainer.stats["episodes"]]
        if scores:
            print(f"原始分数: n={len(scores)} mean={sum(scores)/len(scores):.2f} "
                  f"max={max(scores):.0f}", flush=True)
        return 0
    except BaseException as exc:
        # 训练中途失败也必须留下现场：不然几万步白跑
        try:
            rescue = trainer.save(run_root / f"stage_{args.game}_INTERRUPTED.pt",
                                  stage=f"{args.game}-interrupted")
            print(f"[rescue] 已保存中断现场: {rescue}", flush=True)
        except Exception as save_exc:
            print(f"[rescue] 保存失败: {save_exc}", flush=True)
        raise
    finally:
        try:
            bundle.session.close()
        except Exception:
            pass
        try:
            bundle.conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
