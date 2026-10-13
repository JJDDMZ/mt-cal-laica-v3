#!/usr/bin/env python3
"""Build and run the published v3 real-time experiment stages.

Commands are built with the current interpreter and run from the publication
repository root. Real execution expects a human to operate the WeChat client.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


STAGES = (
    "baseline-f",
    "eval-baseline-f",
    "baseline-t",
    "eval-baseline-t",
    "strict-f",
    "gate-f",
    "strict-t",
    "dual",
)
GAMES = {"baseline-f": "Flappybird", "baseline-t": "Timberman",
         "eval-baseline-f": "Flappybird", "eval-baseline-t": "Timberman",
         "strict-f": "Flappybird", "gate-f": "Flappybird", "strict-t": "Timberman"}
STEPS = 50_000
EPISODES = 50
PORT = 7911
CONNECT_TIMEOUT = 900


class ExperimentError(RuntimeError):
    """A requested stage cannot safely be built or started."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def baseline_root(repo: Path, game: str, seed: int) -> Path:
    return repo / "runs_v3" / "reproductions" / "baselines" / game / f"seed{seed}"


def strict_root(repo: Path, seed: int) -> Path:
    return repo / "runs_v3" / "reproductions" / "main" / "strict" / f"seed{seed}"


def gate_passed(path: Path) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"无法读取 F 中间门槛报告 {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ExperimentError(f"F 中间门槛报告格式错误: {path}")
    episodes, passed = report.get("episodes"), report.get("pass_count")
    score_mean = report.get("score_mean")
    if (isinstance(episodes, bool) or not isinstance(episodes, int) or episodes != EPISODES
            or report.get("game") != "Flappybird"
            or report.get("policy_mode") != "greedy"
            or isinstance(passed, bool) or not isinstance(passed, int) or not 0 <= passed <= EPISODES):
        raise ExperimentError(
            f"F 中间门槛报告必须是 Flappybird greedy、50 局且 pass_count 在 0..50: {path}")
    if score_mean is not None and (isinstance(score_mean, bool)
                                   or not isinstance(score_mean, (int, float))
                                   or not math.isfinite(score_mean)):
        raise ExperimentError(f"F 中间门槛报告 score_mean 必须为有限数值: {path}")
    # evaluate.py defines pass_count as the number of episode scores >= 10.
    return passed >= 40


def _run_command(repo: Path, game: str, seed: int, run_root: Path, *, evaluate: Path | None = None,
                 resume: Path | None = None) -> list[str]:
    command = [sys.executable, "-u", "-m", "wemi_cal.run_rt", "--game", game,
               "--steps", str(STEPS), "--seed", str(seed),
               "--schedule", "strict", "--run-root", str(run_root), "--port", str(PORT),
               "--connect-timeout", str(CONNECT_TIMEOUT), "--wemi-env-root", str((repo / "WemiEnv").resolve())]
    if evaluate is not None:
        command.extend(("--evaluate", str(evaluate), "--episodes", str(EPISODES)))
    if resume is not None:
        command.extend(("--resume", str(resume)))
    return command


def _dual_command(repo: Path, seed: int, run_root: Path, published: bool) -> list[str]:
    checkpoint = run_root / "stage_Timberman.pt"
    command = [sys.executable, "-u", "-m", "wemi_cal.evaluate_rt_dual",
               "--checkpoint", str(checkpoint), "--steps", str(STEPS), "--episodes", str(EPISODES),
               "--seed", str(seed), "--port", str(PORT), "--run-root", str(run_root / "eval"),
               "--wemi-env-root", str((repo / "WemiEnv").resolve())]
    if published:
        results = repo / "results" / "seed0"
        reports = (results / "baseline-Flappybird.json", results / "baseline-Timberman.json")
    else:
        reports = tuple(baseline_root(repo, game, seed) / "eval" / f"eval_{game}_greedy.json"
                        for game in ("Flappybird", "Timberman"))
    command.extend(("--st-flappy-report", str(reports[0]), "--st-timber-report", str(reports[1])))
    return command


def build_command(stage: str, seed: int, *, repo: Path | None = None,
                  use_published_baselines: bool = False) -> list[str]:
    """Return the child command for a stage; performs no runtime checks."""
    repo = (repo or repository_root()).resolve()
    if stage not in STAGES:
        raise ExperimentError(f"未知阶段: {stage}")
    if seed < 0:
        raise ExperimentError("seed 必须为非负整数")
    if use_published_baselines and stage != "dual":
        raise ExperimentError("--use-published-baselines 仅适用于 dual 阶段")
    if use_published_baselines and seed != 0:
        raise ExperimentError("已发布基线只对应 seed0；其他 seed 必须先重做单任务基线")

    if stage.startswith("baseline-"):
        game = GAMES[stage]
        return _run_command(repo, game, seed, baseline_root(repo, game, seed))
    if stage.startswith("eval-baseline-"):
        game = GAMES[stage]
        root = baseline_root(repo, game, seed)
        return _run_command(repo, game, seed, root / "eval", evaluate=root / f"stage_{game}.pt")
    root = strict_root(repo, seed)
    if stage in ("strict-f", "gate-f"):
        if stage == "strict-f":
            return _run_command(repo, "Flappybird", seed, root)
        return _run_command(repo, "Flappybird", seed, root / "eval_F_preT",
                            evaluate=root / "stage_Flappybird.pt")
    if stage == "strict-t":
        return _run_command(repo, "Timberman", seed, root,
                            resume=root / "stage_Flappybird.pt")
    return _dual_command(repo, seed, root, use_published_baselines)


def _expected_outputs(stage: str, seed: int, repo: Path) -> tuple[Path, ...]:
    if stage.startswith(("baseline-", "eval-baseline-")):
        game = GAMES[stage]
        root = baseline_root(repo, game, seed)
        if stage.startswith("baseline-"):
            return root / f"stage_{game}.pt", root / f"stage_{game}_summary.json"
        return (root / "eval" / f"eval_{game}_greedy.json",)
    root = strict_root(repo, seed)
    if stage == "strict-f":
        return root / "stage_Flappybird.pt", root / "stage_Flappybird_summary.json"
    if stage == "gate-f":
        return (root / "eval_F_preT" / "eval_Flappybird_greedy.json",)
    if stage == "strict-t":
        return root / "stage_Timberman.pt", root / "stage_Timberman_summary.json"
    return (root / "eval" / "dual_game_greedy.json",)


def _preflight(stage: str, seed: int, repo: Path, *, use_published_baselines: bool = False) -> None:
    config = repo / "wemi_cal" / "configs" / "mt_cal.yaml"
    env_root = (repo / "WemiEnv").resolve()
    required = [config]
    required.extend(env_root / "WemiEnv" / filename for filename in
                    ("WemiEnv.py", "Flappybird.py", "Timberman.py", "interact.py"))
    games = ("Flappybird", "Timberman") if stage == "dual" else (GAMES.get(stage, "Flappybird"),)
    required.extend(repo / "wechat_games_rt" / game / "project.config.json" for game in games)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ExperimentError("真实执行前置文件缺失:\n" + "\n".join(map(str, missing)))

    strict = strict_root(repo, seed)
    if stage.startswith("eval-baseline-"):
        game = GAMES[stage]
        checkpoint = baseline_root(repo, game, seed) / f"stage_{game}.pt"
        if not checkpoint.is_file():
            raise ExperimentError(f"单任务评测缺少 checkpoint: {checkpoint}")
    if stage == "strict-t":
        checkpoint = strict / "stage_Flappybird.pt"
        gate_report = strict / "eval_F_preT" / "eval_Flappybird_greedy.json"
        if not checkpoint.is_file():
            raise ExperimentError(f"strict-t 缺少 F checkpoint: {checkpoint}")
        if not gate_passed(gate_report):
            raise ExperimentError(f"F gate 未达到至少 40/50 局得分不低于 10，阻止启动 T: {gate_report}")
    elif stage == "gate-f" and not (strict / "stage_Flappybird.pt").is_file():
        raise ExperimentError(f"gate-f 缺少 F checkpoint: {strict / 'stage_Flappybird.pt'}")
    elif stage == "dual" and not (strict / "stage_Timberman.pt").is_file():
        raise ExperimentError(f"dual 缺少最终 T checkpoint: {strict / 'stage_Timberman.pt'}")
    if stage == "dual":
        if use_published_baselines:
            reports = (repo / "results/seed0/baseline-Flappybird.json",
                       repo / "results/seed0/baseline-Timberman.json")
        else:
            reports = tuple(baseline_root(repo, game, seed) / "eval" /
                            f"eval_{game}_greedy.json" for game in ("Flappybird", "Timberman"))
        missing_reports = [path for path in reports if not path.is_file()]
        if missing_reports:
            raise ExperimentError(f"dual 缺少单任务基线报告: {missing_reports[0]}")

    if stage in ("baseline-f", "baseline-t"):
        protected = (baseline_root(repo, GAMES[stage], seed),)
    elif stage == "strict-f":
        protected = (strict,)
    elif stage.startswith("eval-baseline-"):
        game = GAMES[stage]
        protected = (baseline_root(repo, game, seed) / "eval",)
    elif stage == "gate-f":
        protected = (strict / "eval_F_preT",)
    elif stage == "dual":
        protected = (strict / "eval",)
    else:
        protected = ()
    existing = [path for path in protected if path.exists() or path.is_symlink()]
    if existing:
        raise ExperimentError(f"阶段输出目录已存在，拒绝覆盖: {existing[0]}")
    if stage == "strict-t":
        outputs = sorted(strict.glob("stage_Timberman*"))
        if outputs:
            raise ExperimentError(f"strict-t 阶段产物已存在，拒绝重跑: {outputs[0]}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建或执行 v3 realtime 实验阶段命令")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不检查运行时依赖或启动进程")
    parser.add_argument("--use-published-baselines", action="store_true",
                        help="dual 阶段使用发布包 results/seed0 中的基线报告")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed 必须为非负整数")
    if args.use_published_baselines and args.stage != "dual":
        parser.error("--use-published-baselines 仅适用于 dual 阶段")
    if args.use_published_baselines and args.seed != 0:
        parser.error("已发布基线只对应 seed0；其他 seed 必须先重做单任务基线")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = repository_root()
    try:
        command = build_command(args.stage, args.seed, repo=repo,
                                use_published_baselines=args.use_published_baselines)
        if args.stage == "strict-t":
            gate = strict_root(repo, args.seed) / "eval_F_preT" / "eval_Flappybird_greedy.json"
            if not args.dry_run and not gate_passed(gate):
                raise ExperimentError(f"F gate 未达到至少 40/50 且 score_mean>=10，阻止启动 T: {gate}")
        if args.dry_run:
            print(shlex.join(command))
            if args.stage == "dual":
                print("评测块之间请手动切换到对应 wechat_games_rt 客户端；Python 进程会保持运行。")
            return 0
        _preflight(args.stage, args.seed, repo,
                   use_published_baselines=args.use_published_baselines)
    except ExperimentError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    completed = subprocess.run(command, cwd=repo, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
