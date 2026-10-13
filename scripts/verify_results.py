#!/usr/bin/env python3
"""Recompute the published historical seed-0 results without running an agent."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual: float, expected: float, label: str) -> None:
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9),
            f"{label}: computed {actual}, recorded {expected}")


def verify(data_dir: Path, checkpoint: Path | None = None) -> None:
    report = read_json(data_dir / "dual_game_greedy.json")
    audit = read_json(data_dir / "completed_audit.json")
    require(report["seed"] == 0 and report["schedule"] == "strict", "Wrong run identity")
    require(report["policy_mode"] == "greedy", "Expected greedy evaluation")
    require(report["protocol"] == "wemi-rt-v1" and report["clock_mode"] == "realtime-v1",
            "Wrong runtime identity")
    require(len(audit["raw_episodes"]) == 100, "Expected 100 complete episodes")
    require(audit["all_checks_passed"] and all(audit["checks"].values()),
            "Historical audit contains failed checks")

    for game, score_threshold in (("Flappybird", 10), ("Timberman", 50)):
        rows = [r for r in audit["raw_episodes"] if r["game"] == game]
        require([r["episode"] for r in rows] == list(range(1, 51)),
                f"{game}: incomplete or reordered episode block")
        scores = [r["score"] for r in rows]
        recorded = report["games"][game]
        baseline = read_json(data_dir / f"baseline-{game}.json")
        require(baseline["game"] == game and baseline["policy_mode"] == "greedy"
                and baseline["episodes"] == 50, f"{game}: wrong baseline")
        close(statistics.mean(scores), recorded["score_mean"], f"{game} mean")
        close(statistics.median(scores), recorded["score_median"], f"{game} median")
        close(statistics.pstdev(scores), recorded["score_std"], f"{game} std")
        close(max(scores), recorded["score_max"], f"{game} max")
        require(all(r["length"] == 1000 and r["reason"] == "horizon" for r in rows),
                f"{game}: unexpected episode budget/termination")
        passed = sum(s >= score_threshold for s in scores)
        require(passed == recorded["pass_count"] == 50, f"{game}: wrong pass count")
        acceptance = report["acceptance"][game]
        close(baseline["score_mean"], acceptance["st_baseline_mean"], f"{game} baseline")
        close(statistics.mean(scores) / baseline["score_mean"], acceptance["relative_ratio"],
              f"{game} retention")
        require(passed >= 40 and statistics.mean(scores) >= 0.8 * baseline["score_mean"],
                f"{game}: acceptance failed")
        require(acceptance["passed"] and acceptance["passed_absolute"]
                and acceptance["passed_relative"], f"{game}: contradictory acceptance")
        require(recorded["episodes"] == 50 and recorded["truncated_count"] == 0,
                f"{game}: wrong episode count")
        legal = {"0", "1"} if game == "Flappybird" else {"2", "3"}
        require(set(recorded["token_counts"]) == legal
                and sum(recorded["token_counts"].values()) == 50000,
                f"{game}: wrong action accounting")
        print(f"{game}: {passed}/50，均分 {statistics.mean(scores):.2f}，统计与门槛通过")

    blocks = [r["game"] for r in audit["raw_episodes"]]
    require(blocks == ["Flappybird"] * 50 + ["Timberman"] * 50, "Wrong evaluation order")
    require(report["model_hash"] == report["model_hash_after"]
            == audit["checkpoint_model_digest"], "Model digest mismatch")
    require(report["parameter_hash_before"] == report["parameter_hash_after"],
            "Parameter digest mismatch")
    require(report["checkpoint_sha256"] == audit["checkpoint_sha256"],
            "Checkpoint identity mismatch")
    require(report["optimizer_steps"] == 0, "Evaluation updated the optimizer")
    require(all(report[k] for k in ("state_dict_unchanged", "private_generators_unchanged",
                                    "global_rng_unchanged")), "Evaluation mutated state")
    if checkpoint is not None:
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        require(digest.hexdigest() == report["checkpoint_sha256"], "Checkpoint SHA-256 mismatch")
        print("Checkpoint 文件 SHA-256 与历史正式评测一致")
    print("历史 seed-0 的 100 局与评测不变性记录核验通过；未启动训练或评测。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "results/seed0")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    try:
        verify(args.data_dir, args.checkpoint)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"核验失败：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
