"""可追溯身份清单（技术方案 §1.3 / §14 T0）。

记录 WemiEnv commit、dirty diff 摘要、游戏源码 hash、Python 与依赖版本、原文来源。
输出 manifest.json，作为所有运行的身份基线。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

# 需要纳入 hash 的游戏逻辑文件（相对仓库根）。env_hash 的输入（§13）。
GAME_LOGIC_FILES = {
    "Flappybird": [
        "Flappybird/lib/wxbird.js",
        "Flappybird/lib/envcore.js",
        "Flappybird/pages/bird/bird.js",
        "Flappybird/project.config.json",
    ],
    "Timberman": [
        "Timberman/js/main.js",
        "Timberman/js/envcore.js",
        "Timberman/js/base/npc.js",
        "Timberman/js/base/tree.js",
        "Timberman/js/base/databus.js",
        "Timberman/project.config.json",
    ],
}

# 只读输入的资料来源（§18）。
SOURCE_DOCUMENTS = [
    "WemiEnv_Cal-LAICA_双游戏单模型技术方案.md",
    "Cal-LAICA_高信息密度论文解读.md",
    "WemiEnv_高信息密度论文解读.md",
    "WemiEnv_论文总结与复现指南.md",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_git(args: Iterable[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_identity(repo: Path) -> Dict[str, Any]:
    """记录 commit 与工作区状态；不是 git 仓库时明确说明，不伪造。"""
    info: Dict[str, Any] = {"is_repo": False, "path": str(repo)}
    if not (repo / ".git").exists():
        return info
    commit = _run_git(["rev-parse", "HEAD"], repo)
    if commit is None:
        return info
    status = _run_git(["status", "--porcelain"], repo) or ""
    diff = _run_git(["diff", "HEAD"], repo) or ""
    info.update(
        is_repo=True,
        commit=commit,
        branch=_run_git(["branch", "--show-current"], repo),
        dirty=bool(status.strip()),
        status_porcelain=status.splitlines(),
        diff_sha256=hashlib.sha256(diff.encode()).hexdigest(),
        diff_bytes=len(diff.encode()),
    )
    return info


def game_source_hashes(root: Path, game: str) -> Dict[str, Optional[str]]:
    """逐文件 hash；缺失文件记 None（不静默跳过，调用方需自行核对）。"""
    out: Dict[str, Optional[str]] = {}
    for rel in GAME_LOGIC_FILES.get(game, []):
        p = root / rel
        out[rel] = sha256_file(p) if p.is_file() else None
    return out


def dependency_versions() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("torch", "numpy", "gymnasium", "yaml", "pytest"):
        try:
            m = __import__(mod)
        except ImportError:
            info[mod] = None
            continue
        info[mod] = getattr(m, "__version__", "unknown")
    try:
        import torch

        info["torch_num_threads"] = torch.get_num_threads()
    except Exception:  # pragma: no cover
        pass
    return info


def build_manifest(
    *,
    workspace: str | Path,
    wemi_env_repo: str | Path | None = None,
    games_root: str | Path | None = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ws = Path(workspace).resolve()
    repo = Path(wemi_env_repo).resolve() if wemi_env_repo else ws / "WemiEnv"
    games = Path(games_root).resolve() if games_root else ws / "wechat_games"

    manifest: Dict[str, Any] = {
        "manifest_version": 1,
        "workspace": str(ws),
        "wemi_env": git_identity(repo),
        "games_root": str(games),
        "game_source_sha256": {g: game_source_hashes(games, g) for g in GAME_LOGIC_FILES},
        "sources": {},
        "dependencies": dependency_versions(),
    }
    for rel in SOURCE_DOCUMENTS:
        p = ws / rel
        manifest["sources"][rel] = sha256_file(p) if p.is_file() else None
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(path: str | Path, manifest: Dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return p


if __name__ == "__main__":
    ws = Path(__file__).resolve().parent.parent
    m = build_manifest(workspace=ws)
    out = write_manifest(ws / "wemi_cal" / "manifest.json", m)
    print(f"wrote {out}")
    print(json.dumps({k: m[k] for k in ("wemi_env", "dependencies")}, indent=2, ensure_ascii=False)[:2000])
