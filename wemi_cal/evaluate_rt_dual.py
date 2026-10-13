"""使用同一份 checkpoint 和同一个模型顺序评测两款实时游戏。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from . import evaluate as EV
from .envs import features as F
from .envs.features import PolicyMode
from .real_env import LEGACY_PORT

DUAL_GAMES = ("Flappybird", "Timberman")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _generator_snapshot(model: Any) -> Dict[str, torch.Tensor]:
    if not hasattr(model, "generator_state"):
        raise TypeError("模型必须提供 generator_state() 以检查私有 RNG")
    return {name: value.detach().clone() for name, value in model.generator_state().items()}


def _global_rng_snapshot() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
    }


def _restore_global_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def _global_rng_equal(state: Mapping[str, Any]) -> bool:
    now = _global_rng_snapshot()
    return (now["python"] == state["python"]
            and now["numpy"][0] == state["numpy"][0]
            and np.array_equal(now["numpy"][1], state["numpy"][1])
            and now["numpy"][2:] == state["numpy"][2:]
            and torch.equal(now["torch"], state["torch"]))


def _validate_model_games(model: Any) -> None:
    if not hasattr(model, "phi"):
        raise TypeError("双游戏实时评测要求 continuous 模型")
    expected = {
        0: ("Flappybird", 0), 1: ("Flappybird", 1),
        2: ("Timberman", 0), 3: ("Timberman", 1),
    }
    actual = model.phi.registered
    if actual != expected:
        raise RuntimeError(f"checkpoint 的 global token 映射非法：{actual!r}，期望 {expected!r}")
    for game in DUAL_GAMES:
        if tuple(model.phi.registered_games()).count(game) != 1:
            raise RuntimeError(f"{game} 未在 checkpoint 中注册")
        if tuple(F.local_action(game, token) for token in F.legal_tokens(game)) != (0, 1):
            raise RuntimeError(f"{game} 的合法 global token 到 local action 映射异常")


def evaluate_two_games(*, cfg, model, env_factory, episodes: int = 50,
                       seed_start: int = 200000) -> Dict[str, Any]:
    """以一个 model 对象顺序评测两款游戏，且保证返回时模型/RNG 状态不变。"""
    if episodes <= 0:
        raise ValueError("episodes 必须为正")
    _validate_model_games(model)
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model_hash_before = _state_digest(state_before)
    parameter_hash_before = model.parameter_hash()
    generator_before = _generator_snapshot(model)
    global_rng_before = _global_rng_snapshot()
    reports: Dict[str, Any] = {}
    try:
        for game in DUAL_GAMES:
            env = env_factory(game)
            env.start_server()
            try:
                reports[game] = EV.evaluate_game(
                    cfg=cfg, model=model, game=game,
                    env_reset=lambda _game, seed, handle=env: handle.reset(seed),
                    env_step=lambda _game, action, handle=env: handle.step(action),
                    policy_mode=PolicyMode.GREEDY.value, episodes=episodes,
                    seed_start=seed_start, verbose=True,
                ).as_dict()
            finally:
                env.close()
    finally:
        _restore_global_rng(global_rng_before)
    global_rng_unchanged = _global_rng_equal(global_rng_before)

    state_after = model.state_dict()
    state_unchanged = (set(state_before) == set(state_after)
                       and all(torch.equal(state_before[k], state_after[k])
                               for k in state_before))
    generator_after = _generator_snapshot(model)
    generators_unchanged = (set(generator_before) == set(generator_after)
                            and all(torch.equal(generator_before[k], generator_after[k])
                                    for k in generator_before))
    model_hash_after = _state_digest(state_after)
    parameter_hash_after = model.parameter_hash()
    if not (state_unchanged and generators_unchanged and global_rng_unchanged):
        raise RuntimeError("双游戏评测修改了模型 state_dict、私有 generator 或全局 RNG 状态")
    return {
        "games": reports,
        "model_object_id": id(model),
        "model_hash_before": model_hash_before,
        "model_hash_after": model_hash_after,
        "parameter_hash_before": parameter_hash_before,
        "parameter_hash_after": parameter_hash_after,
        "private_generators_unchanged": generators_unchanged,
        "global_rng_unchanged": global_rng_unchanged,
        "state_dict_unchanged": state_unchanged,
        "optimizer_steps": 0,
    }


def load_baseline(path: Optional[str | Path]) -> Optional[float]:
    """从给定的单任务 JSON 报告读取 score_mean；不接受隐式或手填基线。"""
    if path is None:
        return None
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取单任务 baseline JSON {source}: {exc}") from exc
    if not isinstance(payload, dict) or "score_mean" not in payload:
        raise ValueError(f"单任务 baseline JSON 必须包含 score_mean: {source}")
    value = payload["score_mean"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"baseline score_mean 必须是有限数值: {source}")
    return float(value)


def add_acceptance(cfg, reports: Mapping[str, Mapping[str, Any]],
                   baselines: Mapping[str, Optional[float]]) -> Dict[str, Any]:
    """聚合绝对门槛和单任务均值的 80% 相对门槛。"""
    output = {}
    for game in DUAL_GAMES:
        raw = reports[game]
        report = (raw if isinstance(raw, EV.GameReport) else
                  EV.GameReport(**{key: raw[key] for key in EV.GameReport.__dataclass_fields__
                                   if key in raw}))
        baseline = baselines.get(game)
        result = EV.acceptance(cfg=cfg, report=report, st_baseline_mean=baseline).as_dict()
        ratio = None if baseline is None or baseline <= 0 else report.score_mean / baseline
        result["relative_evidence_available"] = baseline is not None
        result["relative_ratio"] = ratio
        result["relative_threshold"] = 0.8
        if baseline is None:
            result["passed_relative"] = False
            result["passed"] = False
        output[game] = result
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _result_path(run_root: str | Path) -> Path:
    """Resolve run-root and require it to remain inside this repo's runs_v3."""
    runs_root = (REPO_ROOT / "runs_v3").resolve()
    resolved = Path(run_root).expanduser().resolve()
    try:
        resolved.relative_to(runs_root)
    except ValueError as exc:
        raise SystemExit(f"--run-root 必须位于仓库 runs_v3/ 下: {resolved}") from exc
    output_path = resolved / "dual_game_greedy.json"
    if output_path.exists() or output_path.is_symlink():
        raise SystemExit(f"结果文件已存在，拒绝覆盖: {output_path}")
    return output_path


def _write_new_result_atomic(output_path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete same-directory temp file atomically without replacing."""
    temporary_path: Optional[Path] = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Same-directory hard link publishes the closed inode atomically and
            # fails with EEXIST instead of replacing a result from another run.
            os.link(temporary_path, output_path)
        except FileExistsError as exc:
            raise SystemExit(f"结果文件已存在，拒绝覆盖: {output_path}") from exc
        except OSError as exc:
            raise RuntimeError(f"无法原子发布评测结果 {output_path}: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"无法写入评测结果 {output_path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                raise RuntimeError(f"无法清理临时评测结果 {temporary_path}: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="单次加载同一模型顺序评测两款实时游戏")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--episodes", type=int, default=50,
                        help="每款游戏固定 50 局；其他值不能用于 acceptance")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--config", default="wemi_cal/configs/mt_cal.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schedule", choices=("strict", "rehearsal"), default="strict")
    parser.add_argument("--wemi-env-root", type=Path, default=Path("WemiEnv"))
    parser.add_argument("--port", type=int, default=LEGACY_PORT)
    parser.add_argument("--st-flappy-report", type=Path)
    parser.add_argument("--st-timber-report", type=Path)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.episodes != 50:
        raise SystemExit("strict acceptance 评测的 --episodes 必须恰为 50")
    if args.steps <= 0:
        raise SystemExit("--steps 必须为正数")
    if args.schedule != "strict":
        raise SystemExit("双游戏 CLI 仅支持 strict 实时训练身份")

    from . import checkpoint as CK
    from .models.continuous import ContinuousModel
    from .real_env import RealTimeGameEnv
    from .run_rt import build_cfg

    cfg = build_cfg(argparse.Namespace(
        config=args.config, variant="continuous", seed=args.seed,
        schedule=args.schedule, steps=args.steps, first_data=None,
        repr_updates=None, offline_updates=None, port=args.port,
    ))
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint 不存在: {args.checkpoint}")
    output_path = _result_path(args.run_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        raise SystemExit(f"结果文件已存在，拒绝覆盖: {output_path}")
    baseline_paths = {
        "Flappybird": args.st_flappy_report,
        "Timberman": args.st_timber_report,
    }
    baseline_info = {
        game: {
            "report_path": str(path.expanduser().resolve()) if path is not None else None,
            "score_mean": load_baseline(path),
        }
        for game, path in baseline_paths.items()
    }
    model = ContinuousModel(seed=cfg.seed)
    CK.load_checkpoint(str(args.checkpoint), cfg=cfg, model=model,
                       restore_replay=False, restore_rng=False)
    _validate_model_games(model)

    def env_factory(game):
        block = DUAL_GAMES.index(game) + 1
        print(f"评测块 {block}/2（{game}）：请在本块开始前打开 "
              f"wechat_games_rt/{game}，并只编译一次。", flush=True)
        return RealTimeGameEnv(game, wemi_env_root=args.wemi_env_root,
                               port=args.port, horizon=cfg.horizon)

    dual = evaluate_two_games(cfg=cfg, model=model, env_factory=env_factory,
                              episodes=args.episodes,
                              seed_start=int(cfg.eval["test_seed_start"]))
    baselines = {game: baseline_info[game]["score_mean"] for game in DUAL_GAMES}
    acceptance = add_acceptance(cfg, dual["games"], baselines)
    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "model_hash": dual["model_hash_before"],
        "protocol": cfg.env["protocol"],
        "clock_mode": cfg.env["clock_mode"],
        "seed": cfg.seed,
        "schedule": cfg.schedule,
        "identity_hash": cfg.identity_hash(),
        "config_path": str(Path(args.config).expanduser().resolve()),
        "baselines": baseline_info,
        "steps": args.steps,
        "episodes_per_game": args.episodes,
        "policy_mode": PolicyMode.GREEDY.value,
        "games": dual["games"],
        "acceptance": acceptance,
        "model_object_id": dual["model_object_id"],
        "model_hash_after": dual["model_hash_after"],
        "parameter_hash_before": dual["parameter_hash_before"],
        "parameter_hash_after": dual["parameter_hash_after"],
        "private_generators_unchanged": dual["private_generators_unchanged"],
        "global_rng_unchanged": dual["global_rng_unchanged"],
        "state_dict_unchanged": dual["state_dict_unchanged"],
        "optimizer_steps": dual["optimizer_steps"],
        "port": args.port,
    }
    _write_new_result_atomic(output_path, payload)
    print(f"双游戏 greedy 评测结果已写入: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
