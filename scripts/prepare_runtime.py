#!/usr/bin/env python3
"""Build the ignored local WemiEnv/game runtime from a pinned upstream checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent
SPEC_PATH = SCRIPT_DIR / "runtime-spec.json"


class PreparationError(RuntimeError):
    """An input failed validation or a safe output could not be assembled."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot read runtime spec {path}: {exc}") from exc
    if spec.get("spec_version") != 1:
        raise PreparationError("unsupported runtime spec version")
    return spec


def git_bytes(upstream: Path, commit: str, relative_path: str) -> bytes:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise PreparationError(f"unsafe upstream path in spec: {relative_path}")
    try:
        return subprocess.run(
            ["git", "-C", str(upstream), "show", f"{commit}:{path.as_posix()}"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreparationError(
            f"cannot read {relative_path} from pinned upstream commit {commit}"
        ) from exc


def validate_upstream(upstream: Path, spec: dict[str, Any]) -> None:
    if not upstream.is_dir():
        raise PreparationError(f"upstream directory does not exist: {upstream}")
    try:
        head = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreparationError("upstream must be a Git checkout") from exc
    if head != spec["upstream_commit"]:
        raise PreparationError(
            f"upstream HEAD is {head}; expected {spec['upstream_commit']}"
        )

    for game in spec["archives"].values():
        archive = git_bytes(upstream, spec["upstream_commit"], game["path"])
        actual = sha256(archive)
        if actual != game["sha256"]:
            raise PreparationError(
                f"pinned archive hash mismatch for {game['path']}: {actual}"
            )

    for filename, entry in spec["python_sources"].items():
        source = git_bytes(
            upstream, spec["upstream_commit"], f"WemiEnv/{filename}"
        )
        actual = sha256(source)
        if actual != entry["input_sha256"]:
            raise PreparationError(
                f"pinned Python source hash mismatch for {filename}: {actual}"
            )


def apply_patches(data: bytes, entry: dict[str, Any], label: str) -> bytes:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreparationError(f"expected UTF-8 text in {label}") from exc
    if entry.get("normalize_newlines"):
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    for patch in entry.get("patches", []):
        old = patch["old"]
        count = text.count(old)
        expected = patch.get("count", 1)
        if count != expected:
            raise PreparationError(
                f"patch for {label} expected {expected} match(es), found {count}"
            )
        text = text.replace(old, patch["new"])
    result = text.encode("utf-8")
    expected_hash = entry.get("output_sha256")
    if expected_hash and sha256(result) != expected_hash:
        raise PreparationError(
            f"assembled source hash mismatch for {label}: {sha256(result)}"
        )
    return result


def safe_zip_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members: list[zipfile.ZipInfo] = []
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        mode = info.external_attr >> 16
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or (mode and stat.S_ISLNK(mode))
        ):
            raise PreparationError(f"unsafe ZIP member path: {name!r}")
        members.append(info)
    return members


def extract_archive(data: bytes, destination: Path) -> None:
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(data)) as archive:
            members = safe_zip_members(archive)
            for info in members:
                if info.is_dir():
                    continue
                target = destination.joinpath(*PurePosixPath(info.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, PreparationError):
            raise
        raise PreparationError(f"cannot safely extract game archive: {exc}") from exc


def sanitize_project_config(game_dir: Path, overrides: dict[str, Any]) -> None:
    path = game_dir / "project.config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"missing or invalid {path.name} in {game_dir}") from exc
    config.update(overrides)
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def overlay_fixtures(source: Path, destination: Path) -> list[Path]:
    if not source.is_dir():
        raise PreparationError(
            f"required fake-client fixtures are missing: {source}"
        )
    copied: list[Path] = []
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        if item.is_symlink():
            raise PreparationError(f"fixture symlinks are not supported: {relative}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise PreparationError(f"unsupported fixture entry: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, target)
        copied.append(target)
    return copied


def file_inventory(root: Path, relative_to: Path) -> dict[str, str]:
    return {
        path.relative_to(relative_to).as_posix(): sha256(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def prepare_runtime(
    upstream: Path,
    repo_root: Path = DEFAULT_REPO_ROOT,
    spec: dict[str, Any] | None = None,
) -> Path:
    """Assemble local runtime paths without modifying the upstream checkout."""
    spec = spec or read_spec()
    upstream = upstream.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    validate_upstream(upstream, spec)

    output_paths = [repo_root / name for name in spec["runtime"]["no_overwrite"]]
    manifest_path = repo_root / spec["runtime"]["manifest"]
    existing = [str(path) for path in output_paths if path.exists()]
    if manifest_path.exists():
        existing.append(str(manifest_path))
    if existing:
        raise PreparationError(
            "refusing to overwrite existing runtime output: " + ", ".join(existing)
        )

    repo_root.mkdir(parents=True, exist_ok=True)
    fixture_source = repo_root / spec["runtime"]["fixture_overlay"]
    if not fixture_source.is_dir():
        raise PreparationError(f"required fake-client fixtures are missing: {fixture_source}")

    staging = Path(tempfile.mkdtemp(prefix=".runtime-prepare-", dir=repo_root))
    try:
        staged_wemi = staging / "WemiEnv/WemiEnv"
        staged_wemi.mkdir(parents=True)
        output_python_hashes: dict[str, str] = {}
        for filename, entry in spec["python_sources"].items():
            source = git_bytes(upstream, spec["upstream_commit"], f"WemiEnv/{filename}")
            result = apply_patches(source, entry, f"WemiEnv/{filename}")
            target = staged_wemi / filename
            target.write_bytes(result)
            output_python_hashes[filename] = sha256(result)

        archive_bytes: dict[str, bytes] = {}
        for game_name, entry in spec["archives"].items():
            archive_bytes[game_name] = git_bytes(
                upstream, spec["upstream_commit"], entry["path"]
            )
            staged_rt = staging / entry["output"]
            extract_archive(archive_bytes[game_name], staged_rt)
            for relative, patches in entry.get("patches", {}).items():
                target = staged_rt / relative
                if not target.is_file():
                    raise PreparationError(f"archive is missing expected file {relative}")
                if isinstance(patches, dict):
                    patch_entry = {
                        "normalize_newlines": patches.get("normalize_newlines", False),
                        "patches": patches.get("operations", []),
                    }
                else:
                    patch_entry = {"patches": patches}
                target.write_bytes(
                    apply_patches(target.read_bytes(), patch_entry, relative)
                )
            sanitize_project_config(staged_rt, entry["project_config"])

            staged_cal = staging / entry["cal_output"]
            extract_archive(archive_bytes[game_name], staged_cal)
            sanitize_project_config(staged_cal, entry["project_config"])

        fixture_overlay_files = overlay_fixtures(
            fixture_source, staging / spec["runtime"]["fixture_overlay_target"]
        )
        generated_cal_files: dict[str, dict[str, str]] = {}
        cal_root = staging / spec["runtime"]["fixture_overlay_target"]
        for relative, entry in spec["runtime"].get("generated_cal_files", {}).items():
            relative_path = PurePosixPath(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise PreparationError(f"unsafe generated fixture path: {relative}")
            content = entry["content"].encode("utf-8")
            actual = sha256(content)
            if actual != entry["sha256"]:
                raise PreparationError(
                    f"generated fixture source hash mismatch for {relative}: {actual}"
                )
            target = cal_root.joinpath(*relative_path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            generated_cal_files[relative] = {
                "sha256": actual,
                "source": entry.get("source", "project-authored runtime file"),
            }

        # All outputs are assembled and validated before any destination appears.
        manifest: dict[str, Any] = {
            "manifest_version": 1,
            "upstream_commit": spec["upstream_commit"],
            "archives": {
                game: {
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                }
                for game, entry in spec["archives"].items()
            },
            "python_sources": {
                name: {
                    "upstream_sha256": spec["python_sources"][name]["input_sha256"],
                    "runtime_sha256": output_python_hashes[name],
                }
                for name in sorted(output_python_hashes)
            },
            "patches": {
                game: entry.get("patches", {})
                for game, entry in spec["archives"].items()
            },
            "project_config_overrides": {
                game: entry["project_config"]
                for game, entry in spec["archives"].items()
            },
            "generated_cal_files": generated_cal_files,
            "fixture_overlay_sha256": {
                path.relative_to(staging).as_posix(): sha256(path.read_bytes())
                for path in fixture_overlay_files
            },
            "generated_files": {},
        }
        for output_name in spec["runtime"]["no_overwrite"]:
            staged_output = staging / output_name
            manifest["generated_files"].update(
                file_inventory(staged_output, staging)
            )
        staged_manifest = staging / spec["runtime"]["manifest"]
        staged_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for output_name in spec["runtime"]["no_overwrite"]:
            source = staging / output_name
            destination = repo_root / output_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise PreparationError(f"output appeared during preparation: {destination}")
            os.replace(source, destination)
        if manifest_path.exists():
            raise PreparationError(f"manifest appeared during preparation: {manifest_path}")
        os.replace(staged_manifest, manifest_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ignored local WemiEnv and mini-game runtime files."
    )
    parser.add_argument(
        "--upstream",
        required=True,
        type=Path,
        help="local WemiEnv Git checkout at the pinned upstream commit",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="public repository root (defaults to this script's parent repository)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = prepare_runtime(args.upstream, args.repo_root)
    except PreparationError as exc:
        print(f"prepare_runtime: error: {exc}", file=sys.stderr)
        return 2
    print(f"Runtime prepared. Source manifest: {manifest}")
    print("Generated WemiEnv/ and wechat_games_* paths must remain locally ignored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
