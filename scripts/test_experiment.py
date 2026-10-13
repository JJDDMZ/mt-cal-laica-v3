"""Offline tests for the experiment command wrapper."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).with_name("experiment.py")
SPEC = importlib.util.spec_from_file_location("experiment", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


class ExperimentCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name) / "mt-cal-laica-v3"

    def test_dry_run_builds_without_executing_child(self) -> None:
        with mock.patch.object(experiment, "repository_root", return_value=self.repo), \
                mock.patch.object(experiment, "_preflight") as preflight, \
                mock.patch.object(experiment.subprocess, "run") as child, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.main(["baseline-f", "--seed", "3", "--dry-run"]), 0)
        preflight.assert_not_called()
        child.assert_not_called()

    def test_strict_t_resumes_same_strict_run_from_f_checkpoint(self) -> None:
        root = experiment.strict_root(self.repo, 7)
        command = experiment.build_command("strict-t", 7, repo=self.repo)
        self.assertEqual(Path(command[command.index("--run-root") + 1]), root.resolve())
        self.assertEqual(Path(command[command.index("--resume") + 1]),
                         (root / "stage_Flappybird.pt").resolve())
        self.assertEqual(command[command.index("--seed") + 1], "7")

    def test_dual_has_one_evaluator_invocation_and_selects_baselines(self) -> None:
        command = experiment.build_command("dual", 0, repo=self.repo)
        self.assertEqual(command[command.index("-m") + 1], "wemi_cal.evaluate_rt_dual")
        self.assertEqual(command.count("--checkpoint"), 1)
        baseline_path = (experiment.baseline_root(self.repo, "Flappybird", 0) / "eval" /
                         "eval_Flappybird_greedy.json").resolve()
        self.assertIn(str(baseline_path), command)
        published = experiment.build_command("dual", 0, repo=self.repo,
                                             use_published_baselines=True)
        self.assertIn(str((self.repo / "results/seed0/baseline-Flappybird.json").resolve()), published)
        with self.assertRaises(experiment.ExperimentError):
            experiment.build_command("dual", 1, repo=self.repo,
                                     use_published_baselines=True)

    def test_strict_t_preflight_rejects_missing_checkpoint(self) -> None:
        self._make_runtime_tree()
        with self.assertRaisesRegex(experiment.ExperimentError, "缺少 F checkpoint"):
            experiment._preflight("strict-t", 0, self.repo)

    def test_strict_t_requires_passed_gate_not_only_report_presence(self) -> None:
        self._make_runtime_tree()
        root = experiment.strict_root(self.repo, 0)
        root.mkdir(parents=True)
        (root / "stage_Flappybird.pt").touch()
        report = root / "eval_F_preT" / "eval_Flappybird_greedy.json"
        report.parent.mkdir()
        report.write_text(json.dumps({"game": "Flappybird", "policy_mode": "greedy",
                                      "episodes": 50, "pass_count": 39, "score_mean": 20}),
                          encoding="utf-8")
        with self.assertRaisesRegex(experiment.ExperimentError, "F gate"):
            experiment._preflight("strict-t", 0, self.repo)

    def test_gate_uses_verified_report_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "gate.json"
            report.write_text(json.dumps({"game": "Flappybird", "policy_mode": "greedy",
                                          "episodes": 50, "pass_count": 40, "score_mean": 8}),
                              encoding="utf-8")
            self.assertTrue(experiment.gate_passed(report))
            for invalid in (
                {"game": "Timberman", "policy_mode": "greedy", "episodes": 50,
                 "pass_count": 50, "score_mean": 100},
                {"game": "Flappybird", "policy_mode": "stochastic_target", "episodes": 50,
                 "pass_count": 50},
                {"game": "Flappybird", "policy_mode": "greedy", "episodes": 50,
                 "pass_count": 51},
                {"game": "Flappybird", "policy_mode": "greedy", "episodes": 49,
                 "pass_count": 49},
            ):
                report.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaises(experiment.ExperimentError):
                    experiment.gate_passed(report)

    def test_gate_validation_allows_low_mean_when_40_episodes_pass(self) -> None:
        self._make_runtime_tree()
        root = experiment.strict_root(self.repo, 0)
        root.mkdir(parents=True)
        (root / "stage_Flappybird.pt").touch()
        report = root / "eval_F_preT" / "eval_Flappybird_greedy.json"
        report.parent.mkdir()
        report.write_text(json.dumps({"game": "Flappybird", "policy_mode": "greedy",
                                      "episodes": 50, "pass_count": 40, "score_mean": 8}),
                          encoding="utf-8")
        experiment._preflight("strict-t", 0, self.repo)

    def test_preflight_uses_published_wemi_env_nested_layout(self) -> None:
        self._make_runtime_tree()
        strict = experiment.strict_root(self.repo, 0)
        strict.mkdir(parents=True)
        (strict / "stage_Flappybird.pt").touch()
        gate = strict / "eval_F_preT" / "eval_Flappybird_greedy.json"
        gate.parent.mkdir()
        gate.write_text(json.dumps({"game": "Flappybird", "policy_mode": "greedy",
                                    "episodes": 50, "pass_count": 40}), encoding="utf-8")
        experiment._preflight("strict-t", 0, self.repo)

    def test_preflight_rejects_partial_eval_directories(self) -> None:
        self._make_runtime_tree()
        baseline = experiment.baseline_root(self.repo, "Flappybird", 0)
        baseline.mkdir(parents=True)
        (baseline / "stage_Flappybird.pt").touch()
        (baseline / "eval").mkdir()
        with self.assertRaisesRegex(experiment.ExperimentError, "输出目录已存在"):
            experiment._preflight("eval-baseline-f", 0, self.repo)

    def test_strict_t_rejects_any_prior_t_stage_artifact(self) -> None:
        self._make_runtime_tree()
        root = experiment.strict_root(self.repo, 0)
        root.mkdir(parents=True)
        (root / "stage_Flappybird.pt").touch()
        report = root / "eval_F_preT" / "eval_Flappybird_greedy.json"
        report.parent.mkdir()
        report.write_text(json.dumps({"game": "Flappybird", "policy_mode": "greedy",
                                      "episodes": 50, "pass_count": 40}), encoding="utf-8")
        (root / "stage_Timberman_partial.json").touch()
        with self.assertRaisesRegex(experiment.ExperimentError, "阶段产物已存在"):
            experiment._preflight("strict-t", 0, self.repo)

    def test_preflight_rejects_existing_fresh_stage_outputs(self) -> None:
        self._make_runtime_tree()
        root = experiment.baseline_root(self.repo, "Flappybird", 0)
        root.mkdir(parents=True)
        (root / "train_log.jsonl").write_text("partial run\n", encoding="utf-8")
        with self.assertRaisesRegex(experiment.ExperimentError, "拒绝覆盖"):
            experiment._preflight("baseline-f", 0, self.repo)

        strict = experiment.strict_root(self.repo, 0)
        strict.mkdir(parents=True)
        (strict / "old_train_log.jsonl").touch()
        with self.assertRaisesRegex(experiment.ExperimentError, "拒绝覆盖"):
            experiment._preflight("strict-f", 0, self.repo)

    def test_negative_and_unknown_seed_or_stage_rejected(self) -> None:
        with self.assertRaises(experiment.ExperimentError):
            experiment.build_command("baseline-f", -1, repo=self.repo)
        with self.assertRaises(experiment.ExperimentError):
            experiment.build_command("something-else", 0, repo=self.repo)
        with self.assertRaises(SystemExit):
            experiment.parse_args(["baseline-f", "--seed", "-1"])

    def test_baseline_f_is_fresh_training_not_resume(self) -> None:
        command = experiment.build_command("baseline-f", 0, repo=self.repo)
        self.assertNotIn("--resume", command)
        self.assertNotIn("--evaluate", command)
        self.assertEqual(command[command.index("--steps") + 1], "50000")
        self.assertEqual(command[command.index("--connect-timeout") + 1], "900")

    def _make_runtime_tree(self) -> None:
        (self.repo / "wemi_cal/configs").mkdir(parents=True)
        (self.repo / "wemi_cal/configs/mt_cal.yaml").touch()
        env_package = self.repo / "WemiEnv/WemiEnv"
        env_package.mkdir(parents=True)
        for filename in ("WemiEnv.py", "Flappybird.py", "Timberman.py", "interact.py"):
            (env_package / filename).touch()
        for game in ("Flappybird", "Timberman"):
            directory = self.repo / "wechat_games_rt" / game
            directory.mkdir(parents=True)
            (directory / "project.config.json").touch()


if __name__ == "__main__":
    unittest.main()
