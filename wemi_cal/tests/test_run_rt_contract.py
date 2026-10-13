import copy
import dataclasses
import math
import random
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wemi_cal.config import ConfigError, load_config
from wemi_cal.envs import features as F
from wemi_cal.learner import UpdateStats
from wemi_cal.replay import ReplayError, make_transition
from wemi_cal.run_rt import RealTimeTrainer, build_cfg


def args_for(**overrides):
    values = dict(
        config="wemi_cal/configs/mt_cal.yaml", variant="continuous",
        seed=0, schedule="strict", steps=5000, first_data=None,
        repr_updates=None, offline_updates=None, port=7911,
    )
    values.update(overrides)
    return Namespace(**values)


class DummyLegacy:
    """Contract tests never call the real simulator."""


def make_trainer(tmp_path, *, game="Flappybird", steps=5000):
    cfg = build_cfg(args_for(steps=steps))
    return RealTimeTrainer(
        cfg, game, DummyLegacy(), run_root=tmp_path / game, seed=0
    )


def add_completed_episode(trainer, *, game, episode_id=0, length=200):
    legal = F.legal_tokens(game)
    for i in range(length):
        trainer.replay.add_confirmed(make_transition(
            buffer=trainer.replay, game=game, episode_id=episode_id,
            request_id=i, x=np.zeros(F.OBS_DIM, dtype=np.float32),
            next_x=np.zeros(F.OBS_DIM, dtype=np.float32),
            e_behavior=np.zeros(2, dtype=np.float32),
            global_token=legal[i % len(legal)], reward_train=0.0,
            log_b_action=-math.log(len(legal)), behavior_kernel="uniform",
            behavior_mapping_version=1, terminated=(i == length - 1),
            truncated=False, termination_reason=("collision" if i == length - 1 else "none"),
            env_hash=f"{game}-test", clock_mode="realtime-v1",
        ), game=game)
    trainer.replay.finish_episode(
        game=game, episode_id=episode_id, terminated=True,
        truncated=False, termination_reason="collision",
    )


def valid_update_stats(batch):
    return UpdateStats(
        td_loss=1.0, cal_regularizer=0.1, q1_loss=0.6, q2_loss=0.5,
        actor_loss=0.2, grad_norm_q1=1.0, grad_norm_q2=1.0,
        grad_norm_actor=1.0, w_mean=1.0, w_max=1.0,
        ess_over_batch=0.5, target_mean=0.0, q_data_mean=0.0,
        n_transitions=int(batch["x"].shape[0]),
        reference_coverage={0: 1.0}, q_minus_reference_frac={0: 0.25},
    )


def test_real_time_config_records_real_protocol_and_clock():
    cfg = build_cfg(args_for())
    assert cfg.env["protocol"] == "wemi-rt-v1"
    assert cfg.env["clock_mode"] == "realtime-v1"
    assert cfg.env["ports"] == {"Flappybird": 7911, "Timberman": 7911}


def test_steps_override_means_per_game_not_total():
    cfg = build_cfg(args_for(steps=5000))
    assert cfg.steps_for_game("Flappybird") == 5000
    assert cfg.steps_for_game("Timberman") == 5000
    assert cfg.total_train_steps == 10000


def test_realtime_build_cfg_forwards_custom_port():
    cfg = build_cfg(args_for(port=8123))
    assert cfg.env["ports"] == {"Flappybird": 8123, "Timberman": 8123}


def test_explicit_production_budget_keeps_production_identity():
    assert build_cfg(args_for(steps=50000)).test_only is False
    assert build_cfg(args_for(steps=5000)).test_only is True


def test_realtime_and_lockstep_protocols_have_distinct_identities():
    realtime = build_cfg(args_for(steps=50000))
    lockstep = load_config("wemi_cal/configs/mt_cal.yaml")
    assert realtime.identity_hash() != lockstep.identity_hash()


def test_rehearsal_rejects_steps_override():
    with pytest.raises(ConfigError, match="rehearsal 使用 YAML 中的固定 30k/30k/10k/10k/10k/10k 预算"):
        build_cfg(args_for(schedule="rehearsal", steps=5000))


def test_non_positive_steps_override_is_rejected():
    with pytest.raises(ConfigError, match="--steps 必须为正，且表示每个游戏的交互步数"):
        build_cfg(args_for(steps=0))


def test_make_transition_requires_explicit_clock_mode(tmp_path):
    from wemi_cal.replay import ReplayBuffer
    buf = ReplayBuffer()
    with pytest.raises(TypeError, match="clock_mode"):
        make_transition(
            buffer=buf, game="Flappybird", episode_id=1, request_id=0,
            x=np.zeros(F.OBS_DIM, dtype=np.float32),
            next_x=np.zeros(F.OBS_DIM, dtype=np.float32),
            e_behavior=np.zeros(2, dtype=np.float32), global_token=0,
            reward_train=0.0, log_b_action=-0.693, behavior_kernel="uniform",
            behavior_mapping_version=1,
        )


def test_realtime_online_transitions_and_checkpoint_keep_realtime_clock(tmp_path):
    import wemi_cal.checkpoint as CK

    trainer = make_trainer(tmp_path)
    trainer.model.register_game_as_first("Flappybird")

    class OneStepLegacy:
        def reset(self, *, seed):
            return np.zeros(F.OBS_DIM, dtype=np.float32), {"episode_id": 0}

        def step(self, action):
            return (np.zeros(F.OBS_DIM, dtype=np.float32), 0.0, True, False,
                    SimpleNamespace(latency_s=0.0, termination_reason="collision",
                                    score_raw=0.0, episode_return_train=0.0,
                                    steps_taken=1))

    trainer.env = OneStepLegacy()
    trainer.online(1, update_every=5, save_every=0)
    transition = trainer.replay.online[0][0].transitions[0]
    assert transition.clock_mode == "realtime-v1"
    checkpoint = trainer.save(tmp_path / "clock.pt", stage="Flappybird")
    restored = CK.load_checkpoint(checkpoint, cfg=trainer.cfg,
                                  model=trainer.model, restore_rng=False)
    assert restored["replay"].online[0][0].transitions[0].clock_mode == "realtime-v1"


def test_periodic_online_resume_runs_only_remaining_stage_budget(tmp_path):
    source = make_trainer(tmp_path)
    source.model.register_game_as_first("Flappybird")

    class CountingLegacy:
        def __init__(self):
            self.step_count = 0
            self.ids = []

        def reset(self, *, seed):
            self.ids.append(0)
            self.step_count = 0
            return np.zeros(F.OBS_DIM, dtype=np.float32), {"episode_id": 0}

        def step(self, action):
            self.step_count += 1
            terminal = self.step_count == 2
            return (np.zeros(F.OBS_DIM, dtype=np.float32), 0.0, terminal, False,
                    SimpleNamespace(latency_s=0.0,
                                    termination_reason="collision" if terminal else "none",
                                    score_raw=0.0, episode_return_train=0.0,
                                    steps_taken=self.step_count))

    env = CountingLegacy()
    source.env = env
    source.online(2, update_every=50, save_every=2)
    checkpoint = source.run_root / "ckpt_latest.pt"
    completed_ids = [ep.episode_id for ep in source.replay.historical[0]]

    resumed = make_trainer(tmp_path)
    resumed.restore(checkpoint)
    resumed.env = env
    resumed.online(5, update_every=50, save_every=0)

    assert resumed.stats["online_steps"] == 5
    assert len(resumed.replay.online[0]) + len(resumed.replay.historical[0]) == 3
    episode_ids = [ep.episode_id for ep in resumed.replay.historical[0] + resumed.replay.online[0]]
    assert len(episode_ids) == len(set(episode_ids))
    assert set(completed_ids).issubset(episode_ids)


def test_restore_other_game_resets_stage_stats_and_episode_counter(tmp_path):
    source = make_trainer(tmp_path, game="Flappybird")
    source.model.register_game_as_first("Flappybird")
    add_completed_episode(source, game="Flappybird", episode_id=0, length=2)
    source.stats["online_steps"] = 9
    source.stats["random_steps"] = 3
    source.next_episode_ids = {0: 44, 1: 2}
    source.stage_phase = "online"
    source.stage_target_online_steps = 9
    source.offline_done = True
    source.completed_tasks = {0}
    checkpoint = source.save(source.run_root / "flappy.pt", stage="Flappybird")

    target = make_trainer(tmp_path, game="Timberman")
    target.restore(checkpoint)

    assert target.stats["online_steps"] == 0
    assert target.stats["random_steps"] == 0
    assert target.next_episode_ids[1] == 2
    assert target.offline_done is True
    assert target.completed_tasks == {0}

    class FourStepLegacy:
        def __init__(self):
            self.n = 0

        def reset(self, *, seed):
            self.n = 0
            return np.zeros(F.OBS_DIM, dtype=np.float32), {"episode_id": 0}

        def step(self, action):
            self.n += 1
            terminal = self.n == 2
            return (np.zeros(F.OBS_DIM, dtype=np.float32), 0.0, terminal, False,
                    SimpleNamespace(latency_s=0.0,
                                    termination_reason="collision" if terminal else "none",
                                    score_raw=0.0, episode_return_train=0.0,
                                    steps_taken=self.n))

    target.model.register_game("Timberman")
    target.env = FourStepLegacy()
    target.online(4, update_every=20, save_every=0)
    assert target.stats["online_steps"] == 4
    assert [ep.episode_id for ep in target.replay.online[1]] == [2, 3]


def test_run_root_must_resolve_inside_repository_runs_v3(monkeypatch, tmp_path):
    import wemi_cal.run_rt as RT
    monkeypatch.setattr(RT, "REPO_ROOT", tmp_path)
    allowed = RT._resolve_run_root(tmp_path / "runs_v3" / "trial")
    assert allowed == (tmp_path / "runs_v3" / "trial").resolve()
    for path in (tmp_path / "runs_v3" / ".." / "runs_other" / "trial",
                 tmp_path / "runs_v2" / "trial", tmp_path.parent / "outside"):
        with pytest.raises(SystemExit, match="runs_v3"):
            RT._resolve_run_root(path)


def test_existing_stage_artifacts_are_refused_before_client_start(monkeypatch, tmp_path):
    import wemi_cal.run_rt as RT
    monkeypatch.setattr(RT, "REPO_ROOT", tmp_path)
    run_root = tmp_path / "runs_v3" / "trial"
    run_root.mkdir(parents=True)
    target = run_root / "stage_Flappybird.pt"
    target.write_bytes(b"preserve")
    started = []

    class NoStartEnv:
        def __init__(self, *args, **kwargs):
            started.append("constructed")

        def start_server(self):
            started.append("started")

    monkeypatch.setattr(RT, "RealTimeGameEnv", NoStartEnv)
    with pytest.raises(SystemExit, match="已存在"):
        RT.main(["--game", "Flappybird", "--steps", "5000",
                 "--run-root", str(run_root)])
    assert started == []
    assert target.read_bytes() == b"preserve"


def test_json_publish_never_replaces_existing_output_and_cleans_temp(tmp_path):
    import wemi_cal.run_rt as RT
    target = tmp_path / "summary.json"
    target.write_text("existing", encoding="utf-8")
    with pytest.raises(SystemExit, match="已存在"):
        RT._publish_json_no_overwrite(target, {"complete": True})
    assert target.read_text(encoding="utf-8") == "existing"
    assert list(tmp_path.glob(".summary.json.*.tmp")) == []


def test_json_publish_cleans_temp_after_serialization_failure(monkeypatch, tmp_path):
    import wemi_cal.run_rt as RT
    target = tmp_path / "summary.json"

    def fail_dump(*_args, **_kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(RT.json, "dump", fail_dump)
    with pytest.raises(OSError, match="simulated"):
        RT._publish_json_no_overwrite(target, {"complete": True})
    assert not target.exists()
    assert list(tmp_path.glob(".summary.json.*.tmp")) == []


def test_realtime_trainer_restore_recovers_owned_state(tmp_path):
    source = make_trainer(tmp_path, game="Flappybird")
    source.stage_phase = "online"
    source.stage_target_online_steps = 9
    source.mapping_version = 4
    source.offline_done = True
    source.completed_tasks = {0, 1}
    source.rng.random()
    source.replay.rng.random()
    for parameter in source.learner.model.beta.parameters():
        parameter.grad = parameter.new_ones(parameter.shape)
    source.learner.opt_beta.step()
    expected_rng = source.rng.bit_generator.state
    expected_replay_rng = source.replay.rng.bit_generator.state
    checkpoint = source.save(source.run_root / "resume.pt", stage="Flappybird@9")

    resumed = make_trainer(tmp_path, game="Flappybird")
    resumed.restore(checkpoint)
    assert resumed.mapping_version == 4
    assert resumed.offline_done is True
    assert resumed.completed_tasks == {0, 1}
    assert resumed.rng.bit_generator.state == expected_rng
    assert resumed.replay.rng.bit_generator.state == expected_replay_rng
    assert resumed.learner.opt_beta.state_dict()["state"]


@pytest.mark.parametrize("corruption", [
    "trainer_state", "replay", "empty_episode_summary", "nonfinite_episode_score",
    "invalid_episode_length", "invalid_episode_reason",
])
def test_failed_restore_does_not_partially_apply_checkpoint(tmp_path, corruption):
    import wemi_cal.checkpoint as CK

    source = make_trainer(tmp_path, game="Flappybird")
    source.model.register_game_as_first("Flappybird")
    if corruption in {"trainer_state", "replay"}:
        add_completed_episode(source, game="Flappybird", episode_id=0, length=2)
        source.next_episode_ids.update({0: 1, 1: 1})
    source.stats["episodes"] = [{"episode_id": 0, "score_raw": 1.0,
                                 "return_train": 2.0, "length": 2,
                                 "reason": "collision"}]
    with torch.no_grad():
        next(source.model.parameters()).add_(3.0)
    for parameter in source.learner.model.beta.parameters():
        parameter.grad = torch.ones_like(parameter)
    source.learner.opt_beta.step()
    source.stage_phase = "online"
    source.stage_target_online_steps = 20
    checkpoint = source.save(source.run_root / "atomic-resume.pt", stage="Flappybird@20")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if corruption == "trainer_state":
        del payload["trainer_state"]["stage_phase"]
    elif corruption == "replay":
        payload["replay"]["online"]["0"][0]["transitions"][0].pop("clock_mode")
    else:
        summary = {
            "empty_episode_summary": {},
            "nonfinite_episode_score": {"score_raw": float("nan")},
            "invalid_episode_length": {"length": True},
            "invalid_episode_reason": {"reason": "exploded"},
        }[corruption]
        for owner in (payload["trainer_state"]["stats"], payload["extra"]["stats"]):
            if corruption == "empty_episode_summary":
                owner["episodes"] = [{}]
            else:
                owner["episodes"][0].update(summary)
    torch.save(payload, checkpoint)

    target = make_trainer(tmp_path, game="Flappybird")
    model_before = {k: v.detach().clone() for k, v in target.model.state_dict().items()}
    optimizer_before = copy.deepcopy(target.learner.opt_beta.state_dict())
    replay_before = target.replay
    replay_rng_before = copy.deepcopy(target.replay.rng.bit_generator.state)
    trainer_fields_before = {
        "mapping_version": target.mapping_version,
        "offline_done": target.offline_done,
        "completed_tasks": set(target.completed_tasks),
        "next_episode_ids": dict(target.next_episode_ids),
        "stage_game": target.stage_game,
        "stage_phase": target.stage_phase,
        "stage_target_online_steps": target.stage_target_online_steps,
        "stats": copy.deepcopy(target.stats),
        "trainer_rng": copy.deepcopy(target.rng.bit_generator.state),
    }
    py_rng_before = random.getstate()
    np_rng_before = np.random.get_state()
    torch_rng_before = torch.get_rng_state().clone()

    expected_error = ("episodes" if corruption in {"empty_episode_summary",
                      "nonfinite_episode_score", "invalid_episode_length",
                      "invalid_episode_reason"} else None)
    with pytest.raises(CK.CheckpointError, match=expected_error):
        target.restore(checkpoint)

    assert all(torch.equal(value, target.model.state_dict()[key])
               for key, value in model_before.items())
    assert target.learner.opt_beta.state_dict() == optimizer_before
    assert target.replay is replay_before
    assert target.replay.total_transitions() == 0
    assert target.replay.rng.bit_generator.state == replay_rng_before
    assert target.mapping_version == trainer_fields_before["mapping_version"]
    assert target.offline_done == trainer_fields_before["offline_done"]
    assert target.completed_tasks == trainer_fields_before["completed_tasks"]
    assert target.next_episode_ids == trainer_fields_before["next_episode_ids"]
    assert target.stage_game == trainer_fields_before["stage_game"]
    assert target.stage_phase == trainer_fields_before["stage_phase"]
    assert target.stage_target_online_steps == trainer_fields_before["stage_target_online_steps"]
    assert target.stats == trainer_fields_before["stats"]
    assert target.rng.bit_generator.state == trainer_fields_before["trainer_rng"]
    assert random.getstate() == py_rng_before
    np_rng_after = np.random.get_state()
    assert np_rng_after[0] == np_rng_before[0]
    assert np.array_equal(np_rng_after[1], np_rng_before[1])
    assert np_rng_after[2:] == np_rng_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng_before)


def test_initial_offline_uses_replay_minibatches_and_runs_once(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    add_completed_episode(trainer, game="Flappybird")
    batches = []

    def fake_update(batch):
        batches.append(batch)
        assert batch["x"].shape[0] == 128
        assert set(batch["task_id"].tolist()) == {0}
        return valid_update_stats(batch)

    monkeypatch.setattr(trainer.learner, "update", fake_update)

    assert trainer.initial_offline(3) == 3
    assert trainer.offline_done is True
    assert trainer.stats["offline_updates"] == 3
    assert len(batches) == 3
    with pytest.raises(RuntimeError, match="只能执行一次"):
        trainer.initial_offline(1)


def test_initial_offline_failure_does_not_mark_experiment_complete(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    add_completed_episode(trainer, game="Flappybird")
    calls = 0

    def fail_on_second_update(batch):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("update failed")
        return valid_update_stats(batch)

    monkeypatch.setattr(trainer.learner, "update", fail_on_second_update)
    with pytest.raises(RuntimeError, match="update failed"):
        trainer.initial_offline(3)

    assert calls == 2
    assert trainer.offline_done is False
    assert trainer.stats["offline_updates"] == 0


def test_initial_offline_rejects_missing_replay_without_marking_complete(tmp_path):
    trainer = make_trainer(tmp_path)

    with pytest.raises(ReplayError, match="replay 为空"):
        trainer.initial_offline(1)

    assert trainer.offline_done is False
    assert trainer.stats["offline_updates"] == 0


def test_finalize_stage_freezes_completed_flappybird_data(tmp_path):
    trainer = make_trainer(tmp_path, game="Flappybird")
    add_completed_episode(trainer, game="Flappybird", episode_id=3)

    trainer.finalize_stage()

    assert trainer.completed_tasks == {0}
    assert trainer.replay.online[0] == []
    assert [ep.episode_id for ep in trainer.replay.historical[0]] == [3]
    assert not any(task == 0 for task, _ in trainer.replay.pending)


def test_timberman_joint_batch_uses_exact_historical_f_and_online_t(tmp_path):
    trainer = make_trainer(tmp_path, game="Timberman")
    add_completed_episode(trainer, game="Flappybird", episode_id=10)
    trainer.replay.freeze_historical(0)
    add_completed_episode(trainer, game="Timberman", episode_id=20)
    trainer.completed_tasks.add(0)

    batch = trainer.make_learner_batch(1)

    assert batch["batch_task_counts"] == {0: 64, 1: 64}
    assert batch["size"] == 128
    assert set(batch["task_id"].tolist()) == {0, 1}
    assert set(batch["games"]) == {"Flappybird", "Timberman"}


def test_timberman_joint_batch_fails_when_historical_source_has_under_64(tmp_path):
    trainer = make_trainer(tmp_path, game="Timberman")
    add_completed_episode(trainer, game="Flappybird", episode_id=10, length=63)
    trainer.replay.freeze_historical(0)
    add_completed_episode(trainer, game="Timberman", episode_id=20)
    trainer.completed_tasks.add(0)

    with pytest.raises(ReplayError, match="来源不足"):
        trainer.make_learner_batch(1)


def test_online_closes_budget_tail_as_invalid_truncated_history(tmp_path):
    trainer = make_trainer(tmp_path, game="Flappybird")

    class OneStepLegacy:
        def reset(self, *, seed):
            return np.zeros(F.OBS_DIM, dtype=np.float32), {"episode_id": 77}

        def step(self, action):
            return (np.ones(F.OBS_DIM, dtype=np.float32), 0.25, False, False,
                    SimpleNamespace(latency_s=0.0, termination_reason="none"))

    trainer.env = OneStepLegacy()
    trainer.model.register_game_as_first("Flappybird")

    trainer.online(1, update_every=2, save_every=0)
    trainer.finalize_stage()

    episode = trainer.replay.historical[0][0]
    assert episode.completed is True
    assert episode.mc_filled is False
    assert episode.transitions[0].truncated is True
    assert episode.transitions[0].ref_valid is False
    assert trainer.completed_tasks == {0}
    assert trainer.replay.online[0] == []


def test_timberman_joint_batch_rejects_wrong_configured_batch_size(tmp_path):
    trainer = make_trainer(tmp_path, game="Timberman")
    add_completed_episode(trainer, game="Flappybird", episode_id=10)
    trainer.replay.freeze_historical(0)
    add_completed_episode(trainer, game="Timberman", episode_id=20)
    trainer.completed_tasks.add(0)
    trainer.replay.batch_size = 64

    with pytest.raises(ReplayError, match="batch size 必须为 128"):
        trainer.make_learner_batch(1)


def test_production_cannot_disable_initial_offline():
    with pytest.raises(ConfigError, match="--no-initial-offline 仅允许 test_only 配置"):
        build_cfg(args_for(steps=50000, no_initial_offline=True))


def test_online_rejects_invalid_update_before_next_environment_step(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)

    class CountingLegacy:
        def __init__(self):
            self.steps = 0
            self.episode_id = 80

        def reset(self, *, seed):
            self.episode_id += 1
            self.steps = 0
            return np.zeros(F.OBS_DIM, dtype=np.float32), {"episode_id": self.episode_id}

        def step(self, action):
            self.steps += 1
            terminated = self.steps == 8
            return (np.zeros(F.OBS_DIM, dtype=np.float32), 0.0, terminated, False,
                    SimpleNamespace(latency_s=0.0,
                                    termination_reason="collision" if terminated else "none",
                                    score_raw=0.0, episode_return_train=0.0,
                                    steps_taken=self.steps))

    env = CountingLegacy()
    trainer.env = env
    trainer.model.register_game_as_first("Flappybird")
    monkeypatch.setattr(
        trainer.learner, "update",
        lambda batch: dataclasses.replace(valid_update_stats(batch), ess_over_batch=0.0),
    )

    with pytest.raises(RuntimeError, match="step=8.*ESS"):
        trainer.online(20, update_every=1, save_every=0)
    assert env.steps == 8


def test_every_fiftieth_online_update_logs_actual_update_evidence(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)

    class CountingLegacy:
        def __init__(self):
            self.steps = 0
            self.episode_id = 81

        def reset(self, *, seed):
            self.episode_id += 1
            self.steps = 0
            return np.zeros(F.OBS_DIM, dtype=np.float32), {"episode_id": self.episode_id}

        def step(self, action):
            self.steps += 1
            terminated = self.steps == 8
            return (np.zeros(F.OBS_DIM, dtype=np.float32), 0.0, terminated, False,
                    SimpleNamespace(latency_s=0.0,
                                    termination_reason="collision" if terminated else "none",
                                    score_raw=0.0, episode_return_train=0.0,
                                    steps_taken=self.steps))

    trainer.env = CountingLegacy()
    trainer.model.register_game_as_first("Flappybird")
    monkeypatch.setattr(trainer.learner, "update", valid_update_stats)

    trainer.online(57, update_every=1, save_every=0)

    import json
    records = [json.loads(line) for line in trainer.log_path.read_text().splitlines()]
    updates = [record for record in records if record.get("event") == "learner_update"]
    assert len(updates) == 1
    record = updates[0]
    assert record["batch_task_counts"] == {"0": 128}
    assert record["td_loss"] == 1.0
    assert record["calibration_regularizer"] == 0.1
    assert record["actor_loss"] == 0.2
    assert record["reference_coverage"] == {"0": 1.0}
    assert record["q_minus_reference_frac"] == {"0": 0.25}
    assert record["grad_norm_q1"] == 1.0
    assert record["grad_norm_q2"] == 1.0
    assert record["grad_norm_actor"] == 1.0
    assert record["w_mean"] == 1.0
    assert record["w_max"] == 1.0
    assert record["ess_over_batch"] == 0.5
