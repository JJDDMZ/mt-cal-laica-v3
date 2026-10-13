"""Standard-library tests for the pinned runtime assembler."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).with_name("prepare_runtime.py")
MODULE_SPEC = importlib.util.spec_from_file_location("prepare_runtime", SCRIPT)
assert MODULE_SPEC and MODULE_SPEC.loader
prepare_runtime = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(prepare_runtime)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RuntimePreparationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.upstream = self.base / "upstream"
        self.repo = self.base / "public"
        self.upstream.mkdir()
        self.repo.mkdir()
        (self.repo / "tests/runtime-fixtures/wemi_cal").mkdir(parents=True)
        (self.repo / "tests/runtime-fixtures/wemi_cal/fake.py").write_text(
            "fixture = True\n", encoding="utf-8"
        )
        self.source_bytes = b"upstream module\n"
        self.archive_bytes = self.make_archive()
        (self.upstream / "WemiEnv").mkdir()
        (self.upstream / "WemiEnv/module.py").write_bytes(self.source_bytes)
        (self.upstream / "vchat_minigame").mkdir()
        (self.upstream / "vchat_minigame/game.zip").write_bytes(self.archive_bytes)
        subprocess.run(["git", "init", "-q", str(self.upstream)], check=True)
        subprocess.run(
            ["git", "-C", str(self.upstream), "config", "user.email", "test@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.upstream), "config", "user.name", "Runtime Test"],
            check=True,
        )
        subprocess.run(["git", "-C", str(self.upstream), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(self.upstream), "commit", "-qm", "fixture"], check=True
        )
        self.commit = subprocess.run(
            ["git", "-C", str(self.upstream), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        self.spec: dict[str, Any] = {
            "spec_version": 1,
            "upstream_commit": self.commit,
            "archives": {
                "Game": {
                    "path": "vchat_minigame/game.zip",
                    "sha256": digest(self.archive_bytes),
                    "output": "wechat_games_rt/Game",
                    "cal_output": "wechat_games_cal/Game",
                    "patches": {
                        "js/main.js": [
                            {"old": "7891", "new": "7911", "count": 1}
                        ]
                    },
                    "project_config": {
                        "appid": "touristappid",
                        "isGameTourist": True,
                        "libVersion": "3.17.3",
                    },
                }
            },
            "python_sources": {
                "module.py": {
                    "input_sha256": digest(self.source_bytes),
                    "output_sha256": digest(b"runtime module\n"),
                    "patches": [
                        {"old": "upstream", "new": "runtime", "count": 1}
                    ],
                }
            },
            "runtime": {
                "python_output_dir": "WemiEnv/WemiEnv",
                "fixture_overlay": "tests/runtime-fixtures",
                "fixture_overlay_target": "wechat_games_cal",
                "manifest": "runtime-manifest.json",
                "no_overwrite": ["WemiEnv", "wechat_games_rt", "wechat_games_cal"],
                "generated_cal_files": {
                    "Game/lib/adapter.js": {
                        "content": "module.exports = 'project adapter';\n",
                        "sha256": digest(b"module.exports = 'project adapter';\n"),
                        "source": "test-owned adapter fixture",
                    }
                },
            },
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def make_archive(extra: tuple[str, bytes] | None = None) -> bytes:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("js/main.js", "connect('ws://localhost:7891')")
            archive.writestr("project.config.json", '{"appid":"private"}')
            if extra:
                archive.writestr(extra[0], extra[1])
        return output.getvalue()

    def test_build_hashes_patches_overlay_and_safe_config(self) -> None:
        manifest_path = prepare_runtime.prepare_runtime(
            self.upstream, self.repo, self.spec
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        module = self.repo / "WemiEnv/WemiEnv/module.py"
        self.assertEqual(module.read_bytes(), b"runtime module\n")
        self.assertEqual(
            (self.repo / "wechat_games_rt/Game/js/main.js").read_text(),
            "connect('ws://localhost:7911')",
        )
        config = json.loads(
            (self.repo / "wechat_games_rt/Game/project.config.json").read_text()
        )
        self.assertEqual(config["appid"], "touristappid")
        self.assertTrue(config["isGameTourist"])
        self.assertEqual(config["libVersion"], "3.17.3")
        self.assertTrue((self.repo / "wechat_games_cal/wemi_cal/fake.py").is_file())
        self.assertEqual(
            (self.repo / "wechat_games_cal/Game/lib/adapter.js").read_text(),
            "module.exports = 'project adapter';\n",
        )
        self.assertEqual(
            manifest["generated_cal_files"]["Game/lib/adapter.js"]["source"],
            "test-owned adapter fixture",
        )
        self.assertEqual(
            manifest["project_config_overrides"]["Game"]["libVersion"], "3.17.3"
        )
        self.assertEqual(
            manifest["generated_files"]["WemiEnv/WemiEnv/module.py"],
            digest(b"runtime module\n"),
        )
        self.assertEqual(manifest["upstream_commit"], self.commit)

    def test_wrong_commit_is_rejected_before_creating_outputs(self) -> None:
        self.spec["upstream_commit"] = "0" * 40
        with self.assertRaisesRegex(prepare_runtime.PreparationError, "HEAD is"):
            prepare_runtime.prepare_runtime(self.upstream, self.repo, self.spec)
        self.assertFalse((self.repo / "WemiEnv").exists())

    def test_archive_hash_mismatch_is_rejected(self) -> None:
        self.spec["archives"]["Game"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(prepare_runtime.PreparationError, "archive hash mismatch"):
            prepare_runtime.prepare_runtime(self.upstream, self.repo, self.spec)

    def test_zip_path_traversal_is_rejected(self) -> None:
        malicious = self.make_archive(("../escape.txt", b"no"))
        with zipfile.ZipFile(io.BytesIO(malicious)) as archive:
            with self.assertRaisesRegex(prepare_runtime.PreparationError, "unsafe ZIP member"):
                prepare_runtime.safe_zip_members(archive)

    def test_existing_outputs_are_never_overwritten(self) -> None:
        target = self.repo / "wechat_games_rt"
        target.mkdir()
        marker = target / "keep.txt"
        marker.write_text("user data", encoding="utf-8")
        with self.assertRaisesRegex(prepare_runtime.PreparationError, "refusing to overwrite"):
            prepare_runtime.prepare_runtime(self.upstream, self.repo, self.spec)
        self.assertEqual(marker.read_text(encoding="utf-8"), "user data")

    def test_second_prepare_is_explicitly_refused_without_changes(self) -> None:
        prepare_runtime.prepare_runtime(self.upstream, self.repo, self.spec)
        before = (self.repo / "runtime-manifest.json").read_bytes()
        with self.assertRaisesRegex(prepare_runtime.PreparationError, "refusing to overwrite"):
            prepare_runtime.prepare_runtime(self.upstream, self.repo, self.spec)
        self.assertEqual((self.repo / "runtime-manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
