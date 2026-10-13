"""双游戏联合评测：模型复用、只读状态与基线验收。"""

from __future__ import annotations

import copy
import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from wemi_cal import evaluate as EV
from wemi_cal.evaluate import GameReport
from wemi_cal.evaluate_rt_dual import evaluate_two_games
from wemi_cal.envs import features as F
from wemi_cal.envs.features import PolicyMode
from wemi_cal.models.continuous import ContinuousModel
from wemi_cal.run_rt import build_cfg
import wemi_cal.evaluate_rt_dual as DUAL


def cfg_for():
    return build_cfg(Namespace(
        config="wemi_cal/configs/mt_cal.yaml", variant="continuous",
        seed=0, schedule="strict", steps=5000, first_data=None,
        repr_updates=None, offline_updates=None, port=7911,
    ))


class FakeHandle:
    def __init__(self, game):
        self.game = game
        self.started = False
        self.closed = False
        self.actions = []

    def start_server(self):
        self.started = True

    def reset(self, seed):
        x = np.zeros(F.OBS_DIM, dtype=np.float32)
        x[26 + F.task_id(self.game)] = 1.0
        return x, {"episode_id": seed}

    def step(self, action):
        self.actions.append(action)
        x = np.zeros(F.OBS_DIM, dtype=np.float32)
        x[26 + F.task_id(self.game)] = 1.0
        return x, 0.0, True, False, {"score_raw": 0.0, "termination_reason": "done"}

    def close(self):
        self.closed = True


def clone_state_dict(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def assert_state_dict_equal(before, model):
    after = model.state_dict()
    assert set(before) == set(after)
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_dual_eval_reuses_one_model_and_never_mutates_it(monkeypatch):
    cfg = cfg_for()
    model = ContinuousModel(seed=0)
    model.register_game("Flappybird")
    model.register_game("Timberman")
    seen_ids = []
    handles = []

    def fake_evaluate_game(**kwargs):
        seen_ids.append(id(kwargs["model"]))
        assert kwargs["policy_mode"] == PolicyMode.GREEDY.value
        return GameReport(game=kwargs["game"], policy_mode="greedy", episodes=2)

    monkeypatch.setattr(EV, "evaluate_game", fake_evaluate_game)
    before_state = clone_state_dict(model)
    before_rng = model.generator_state()
    before_hash = model.parameter_hash()
    report = evaluate_two_games(
        cfg=cfg, model=model,
        env_factory=lambda game: handles.append(FakeHandle(game)) or handles[-1],
        episodes=2, seed_start=200000,
    )

    assert set(report["games"]) == {"Flappybird", "Timberman"}
    assert len(seen_ids) == 2 and set(seen_ids) == {id(model)}
    assert report["model_object_id"] == id(model)
    assert report["parameter_hash_before"] == before_hash
    assert report["parameter_hash_after"] == before_hash
    assert_state_dict_equal(before_state, model)
    assert all(torch.equal(before_rng[name], model.generator_state()[name])
               for name in before_rng)
    assert [h.game for h in handles] == ["Flappybird", "Timberman"]
    assert all(h.started and h.closed for h in handles)


def test_dual_eval_preserves_global_rng_states():
    cfg = cfg_for()
    model = ContinuousModel(seed=1)
    model.register_game("Flappybird")
    model.register_game("Timberman")
    handles = {}
    random.seed(813)
    np.random.seed(814)
    torch.manual_seed(815)
    py_before = random.getstate()
    np_before = copy.deepcopy(np.random.get_state())
    torch_before = torch.get_rng_state().clone()

    evaluate_two_games(
        cfg=cfg, model=model,
        env_factory=lambda game: handles.setdefault(game, FakeHandle(game)),
        episodes=1, seed_start=200000,
    )

    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0] and np.array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert set(handles["Flappybird"].actions).issubset({0, 1})
    assert set(handles["Timberman"].actions).issubset({0, 1})


def test_dual_eval_output_has_absolute_and_baseline_acceptance():
    from wemi_cal.evaluate_rt_dual import add_acceptance

    cfg = cfg_for()
    reports = {
        "Flappybird": GameReport(game="Flappybird", policy_mode="greedy", episodes=10,
                                 score_mean=8.0, pass_count=8, pass_rate=0.8),
        "Timberman": GameReport(game="Timberman", policy_mode="greedy", episodes=10,
                                score_mean=16.0, pass_count=7, pass_rate=0.7),
    }
    accepted = add_acceptance(cfg, reports, {"Flappybird": 10.0, "Timberman": 20.0})
    assert accepted["Flappybird"]["passed_absolute"] is True
    assert accepted["Flappybird"]["passed_relative"] is True
    assert accepted["Flappybird"]["relative_ratio"] == pytest.approx(0.8)
    assert accepted["Timberman"]["passed_absolute"] is False
    assert accepted["Timberman"]["passed_relative"] is True
    missing = add_acceptance(cfg, reports, {})
    assert missing["Flappybird"]["relative_evidence_available"] is False
    assert missing["Flappybird"]["passed_relative"] is False


def test_acceptance_marks_missing_relative_baseline_unavailable_and_failed():
    cfg = cfg_for()
    report = GameReport(game="Flappybird", policy_mode="greedy", episodes=10,
                        score_mean=10.0, pass_count=10, pass_rate=1.0)
    result = EV.acceptance(cfg=cfg, report=report, st_baseline_mean=None)
    assert result.passed_absolute is True
    assert result.passed_relative is False
    assert result.relative_evidence_available is False
    assert result.passed is False


def test_baseline_loader_reads_score_mean_from_explicit_json(tmp_path):
    from wemi_cal.evaluate_rt_dual import load_baseline

    report_path = tmp_path / "st-flappy.json"
    report_path.write_text(json.dumps({"score_mean": 12.5}), encoding="utf-8")
    assert load_baseline(report_path) == 12.5
    with pytest.raises(ValueError, match="score_mean"):
        report_path.write_text(json.dumps({"score": 12.5}), encoding="utf-8")
        load_baseline(report_path)


def prepare_cli(monkeypatch, tmp_path):
    """Patch CLI boundaries so its output contract can be tested without a client."""
    import wemi_cal.checkpoint as CK
    import wemi_cal.real_env as real_env
    import wemi_cal.run_rt as run_rt

    cfg = build_cfg(Namespace(
        config="wemi_cal/configs/mt_cal.yaml", variant="continuous",
        seed=17, schedule="strict", steps=50000, first_data=None,
        repr_updates=None, offline_updates=None, port=7911,
    ))
    monkeypatch.setattr(DUAL, "REPO_ROOT", tmp_path, raising=False)
    calls = {"load": 0, "env": [], "env_ports": [], "evaluate": 0, "cfg_args": None}
    monkeypatch.setattr(run_rt, "build_cfg", lambda args: calls.update(cfg_args=args) or cfg)

    def load_checkpoint(_path, *, model, **_kwargs):
        calls["load"] += 1
        model.register_game("Flappybird")
        model.register_game("Timberman")

    class FakeRealtimeEnv:
        def __init__(self, game, **_kwargs):
            self.game = game
            calls["env"].append(game)
            calls["env_ports"].append(_kwargs.get("port"))

        def start_server(self):
            pass

        def close(self):
            pass

    def fake_evaluate_two_games(*, cfg, model, env_factory, episodes, seed_start):
        calls["evaluate"] += 1
        assert seed_start == cfg.eval["test_seed_start"]
        for game in DUAL_GAMES_FOR_TEST:
            env_factory(game)
        reports = {
            game: GameReport(game=game, policy_mode="greedy", episodes=50,
                             score_mean=10.0, pass_count=45, pass_rate=0.9).as_dict()
            for game in DUAL_GAMES_FOR_TEST
        }
        return {
            "games": reports, "model_object_id": id(model),
            "model_hash_before": "model-hash", "model_hash_after": "model-hash",
            "parameter_hash_before": "param-hash", "parameter_hash_after": "param-hash",
            "private_generators_unchanged": True, "global_rng_unchanged": True,
            "state_dict_unchanged": True, "optimizer_steps": 0,
        }

    monkeypatch.setattr(CK, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(real_env, "RealTimeGameEnv", FakeRealtimeEnv)
    monkeypatch.setattr(DUAL, "evaluate_two_games", fake_evaluate_two_games)
    monkeypatch.setattr(DUAL, "REPO_ROOT", tmp_path, raising=False)

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    flappy_baseline = tmp_path / "flappy.json"
    timber_baseline = tmp_path / "timber.json"
    flappy_baseline.write_text(json.dumps({"score_mean": 12}), encoding="utf-8")
    timber_baseline.write_text(json.dumps({"score_mean": 15}), encoding="utf-8")
    return cfg, calls, checkpoint, flappy_baseline, timber_baseline


DUAL_GAMES_FOR_TEST = ("Flappybird", "Timberman")


def test_cli_records_realtime_identity_and_baseline_sources(monkeypatch, tmp_path):
    cfg, calls, checkpoint, flappy, timber = prepare_cli(monkeypatch, tmp_path)
    run_root = tmp_path / "runs_v3" / "trial"
    result = DUAL.main([
        "--checkpoint", str(checkpoint), "--run-root", str(run_root),
        "--st-flappy-report", str(flappy), "--st-timber-report", str(timber),
    ])
    assert result == 0
    payload = json.loads((run_root / "dual_game_greedy.json").read_text(encoding="utf-8"))
    assert payload["protocol"] == "wemi-rt-v1"
    assert payload["clock_mode"] == "realtime-v1"
    assert payload["seed"] == 17
    assert payload["schedule"] == "strict"
    assert payload["identity_hash"] == cfg.identity_hash()
    assert payload["config_path"] == str(Path("wemi_cal/configs/mt_cal.yaml").resolve())
    assert payload["baselines"] == {
        "Flappybird": {"report_path": str(flappy.resolve()), "score_mean": 12.0},
        "Timberman": {"report_path": str(timber.resolve()), "score_mean": 15.0},
    }
    assert calls["load"] == 1
    assert calls["env"] == list(DUAL_GAMES_FOR_TEST)


def test_cli_records_missing_baselines_as_null(monkeypatch, tmp_path):
    _cfg, _calls, checkpoint, _flappy, _timber = prepare_cli(monkeypatch, tmp_path)
    run_root = tmp_path / "runs_v3" / "trial"
    DUAL.main(["--checkpoint", str(checkpoint), "--run-root", str(run_root)])
    payload = json.loads((run_root / "dual_game_greedy.json").read_text(encoding="utf-8"))
    assert payload["baselines"] == {
        "Flappybird": {"report_path": None, "score_mean": None},
        "Timberman": {"report_path": None, "score_mean": None},
    }


def test_dual_cli_accepts_port_and_records_it(monkeypatch, tmp_path):
    cfg, calls, checkpoint, _flappy, _timber = prepare_cli(monkeypatch, tmp_path)
    run_root = tmp_path / "runs_v3" / "port"
    DUAL.main(["--checkpoint", str(checkpoint), "--run-root", str(run_root),
               "--port", "8123"])
    payload = json.loads((run_root / "dual_game_greedy.json").read_text(encoding="utf-8"))
    assert calls["cfg_args"].port == 8123
    assert calls["env_ports"] == [8123, 8123]
    assert payload["port"] == 8123


@pytest.mark.parametrize("run_root_kind", ["outside", "escape"])
def test_cli_rejects_output_roots_outside_runs_v3(monkeypatch, tmp_path, run_root_kind):
    _cfg, calls, checkpoint, flappy, timber = prepare_cli(monkeypatch, tmp_path)
    if run_root_kind == "outside":
        run_root = tmp_path / "runs_other" / "trial"
    else:
        run_root = tmp_path / "runs_v3" / ".." / "escaped"
    with pytest.raises(SystemExit, match="runs_v3"):
        DUAL.main([
            "--checkpoint", str(checkpoint), "--run-root", str(run_root),
            "--st-flappy-report", str(flappy), "--st-timber-report", str(timber),
        ])
    assert calls["load"] == 0
    assert calls["evaluate"] == 0


def test_cli_refuses_to_overwrite_existing_result(monkeypatch, tmp_path):
    _cfg, calls, checkpoint, flappy, timber = prepare_cli(monkeypatch, tmp_path)
    run_root = tmp_path / "runs_v3" / "trial"
    run_root.mkdir(parents=True)
    output = run_root / "dual_game_greedy.json"
    output.write_text("keep this result", encoding="utf-8")
    with pytest.raises(SystemExit, match="已存在"):
        DUAL.main([
            "--checkpoint", str(checkpoint), "--run-root", str(run_root),
            "--st-flappy-report", str(flappy), "--st-timber-report", str(timber),
        ])
    assert output.read_text(encoding="utf-8") == "keep this result"
    assert calls["load"] == 0
    assert calls["evaluate"] == 0


def test_cli_requires_exactly_50_episodes(monkeypatch, tmp_path):
    _cfg, calls, checkpoint, flappy, timber = prepare_cli(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="恰为 50"):
        DUAL.main([
            "--checkpoint", str(checkpoint), "--run-root", str(tmp_path / "runs_v3" / "trial"),
            "--episodes", "49", "--st-flappy-report", str(flappy),
            "--st-timber-report", str(timber),
        ])
    assert calls["load"] == 0
    assert calls["evaluate"] == 0


def test_atomic_publish_keeps_final_path_absent_until_complete_temp_is_linked(monkeypatch, tmp_path):
    output = tmp_path / "dual_game_greedy.json"
    payload = {"complete": True, "score_mean": 42.0}
    real_link = DUAL.os.link
    publish_observations = []

    def observe_then_link(source, target):
        source_path = Path(source)
        target_path = Path(target)
        assert not target_path.exists()
        assert json.loads(source_path.read_text(encoding="utf-8")) == payload
        publish_observations.append((target_path.exists(), source_path.exists()))
        real_link(source, target)

    monkeypatch.setattr(DUAL.os, "link", observe_then_link)
    DUAL._write_new_result_atomic(output, payload)

    assert publish_observations == [(False, True)]
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob(".dual_game_greedy.json.*.tmp")) == []


def test_atomic_publish_preserves_existing_target_and_cleans_temp(monkeypatch, tmp_path):
    output = tmp_path / "dual_game_greedy.json"
    output.write_text("existing result", encoding="utf-8")
    real_link = DUAL.os.link
    link_called = []

    def record_existing_target(source, target):
        link_called.append((Path(source), Path(target)))
        real_link(source, target)

    monkeypatch.setattr(DUAL.os, "link", record_existing_target)
    with pytest.raises(SystemExit, match="已存在"):
        DUAL._write_new_result_atomic(output, {"replacement": True})

    assert len(link_called) == 1
    assert output.read_text(encoding="utf-8") == "existing result"
    assert not link_called[0][0].exists()
    assert list(tmp_path.glob(".dual_game_greedy.json.*.tmp")) == []


def test_atomic_publish_cleans_temp_when_serialization_fails(monkeypatch, tmp_path):
    output = tmp_path / "dual_game_greedy.json"

    def fail_serialization(*_args, **_kwargs):
        raise TypeError("cannot encode test payload")

    monkeypatch.setattr(DUAL.json, "dump", fail_serialization)
    with pytest.raises(TypeError, match="cannot encode"):
        DUAL._write_new_result_atomic(output, {"data": object()})

    assert not output.exists()
    assert list(tmp_path.glob(".dual_game_greedy.json.*.tmp")) == []


def test_atomic_publish_normalizes_os_link_failure_and_cleans_temp(monkeypatch, tmp_path):
    output = tmp_path / "dual_game_greedy.json"

    def fail_link(_source, _target):
        raise OSError(5, "simulated filesystem failure")

    monkeypatch.setattr(DUAL.os, "link", fail_link)
    with pytest.raises(RuntimeError, match="无法原子发布评测结果"):
        DUAL._write_new_result_atomic(output, {"complete": True})

    assert not output.exists()
    assert list(tmp_path.glob(".dual_game_greedy.json.*.tmp")) == []
