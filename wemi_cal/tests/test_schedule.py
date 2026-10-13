"""T7 验收：阶段状态机、一次初始离线预训练、预算截断与冻结表（§9，§14 T7）。

用假环境 + 假回调跑完整状态机：不需要模拟器。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from wemi_cal.config import load_config, validate_config
from wemi_cal.envs import features as F
from wemi_cal.models.continuous import ContinuousModel
from wemi_cal.schedule import (
    FROZEN_BY_SUBPHASE,
    ScheduleError,
    SubPhase,
    Trainer,
    TrainerCallbacks,
    assert_frozen,
    set_trainable,
    trainable_modules,
)

CONFIG_PATH = "wemi_cal/configs/mt_cal.yaml"


def small_cfg(schedule="strict", **train_overrides):
    import yaml

    raw = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    raw["train"]["random_steps_first_visit"] = 20
    raw["train"]["repr_updates_first_visit"] = 5
    raw["train"]["initial_offline_updates"] = 5
    raw["train"]["stage_schedules"]["strict"] = [
        {"game": "Flappybird", "steps": 100}, {"game": "Timberman", "steps": 100},
    ]
    raw["train"]["stage_schedules"]["rehearsal"] = [
        {"game": "Flappybird", "steps": 60}, {"game": "Timberman", "steps": 60},
        {"game": "Flappybird", "steps": 40}, {"game": "Timberman", "steps": 40},
        {"game": "Flappybird", "steps": 20}, {"game": "Timberman", "steps": 20},
    ]
    raw["train"].update(train_overrides)
    raw["schedule"] = schedule
    return validate_config(raw, test_only=True)


class FakeEnvRunner:
    """记录每个游戏被采样了多少步、是否在非首次阶段出现新采样。"""

    def __init__(self, episode_len=7):
        self.episode_len = episode_len
        self.step_counts = {0: 0, 1: 0}
        self.step_counts_by_stage: list = []
        self.reset_counts = {0: 0, 1: 0}
        self.episode_t = 0
        self.episode_id = -1
        self.current_game = None
        self.episode_log: list = []

    def reset(self, game, seed):
        task = F.task_id(game)
        self.current_game = game
        self.episode_t = 0
        self.episode_id += 1
        self.reset_counts[task] += 1
        x = np.zeros(F.OBS_DIM, dtype=np.float32)
        x[26 + task] = 1.0
        return x, {"episode_id": self.episode_id}

    def step(self, game, local):
        task = F.task_id(game)
        self.step_counts[task] += 1
        self.episode_t += 1
        done = self.episode_t >= self.episode_len
        x = np.zeros(F.OBS_DIM, dtype=np.float32)
        x[26 + task] = 1.0
        r = 1.0 if not done else -1.0
        info = {
            "episode_id": self.episode_id,
            "termination_reason": "collision" if done else "none",
            "episode_return_train": 1.0 * (self.episode_t - 1) - 1.0,
        }
        return x, r, done, False, info

    def choose(self, game, mapping_version, rollout_id):
        token = F.legal_tokens(game)[0]
        return token, F.local_action(game, token), math.log(0.5), np.zeros(2, dtype=np.float32), 0


def build_trainer(cfg, runner=None):
    runner = runner or FakeEnvRunner()
    model = ContinuousModel(seed=0)
    counters = {"transitions": 0, "adapt_calls": [], "offline_calls": [], "updates": 0,
                "episode_ends": [], "first_data_steps": []}

    def _register_first_time(game):
        """首个接入的游戏必须允许 token=2/3（ST-T）；后续游戏按固定表追加。"""
        if game in model.phi.registered_games():
            return
        if model.phi.num_tokens == 0:
            model.register_game_as_first(game)
        else:
            model.register_game(game)

    def collect_first_data(game, steps, mapping_version):
        # 首次采集：均匀动作，不调用尚未扩展的 φ
        counters["first_data_steps"].append((game, steps))
        runner.step_counts[F.task_id(game)] += steps
        _register_first_time(game)

    def adapt_representation(game, updates):
        counters["adapt_calls"].append((game, updates))
        _register_first_time(game)
        return len(counters["adapt_calls"])

    def initial_offline(game, updates):
        counters["offline_calls"].append((game, updates))
        # 离线阶段必须只更新 β/Q：这里断言冻结表已生效
        assert_frozen(model, SubPhase.INITIAL_OFFLINE, game=game)
        return updates

    def learner_update(mapping_version):
        counters["updates"] += 1
        assert_frozen(model, SubPhase.ONLINE, game="Flappybird")
        return {"td_loss": 0.1, "actor_loss": 0.2, "ess_over_batch": 0.9}

    def on_transition(**kw):
        counters["transitions"] += 1

    def on_episode_end(game, episode_id, term, trunc, reason):
        counters["episode_ends"].append((game, term, trunc, reason))

    cb = TrainerCallbacks(
        collect_first_data=collect_first_data,
        adapt_representation=adapt_representation,
        learner_update=learner_update,
        initial_offline=initial_offline,
        env_reset=runner.reset,
        env_step=runner.step,
        choose_action=runner.choose,
        on_episode_end=on_episode_end,
        on_transition=on_transition,
    )
    tr = Trainer(cfg=cfg, model=model, callbacks=cb, horizon=1000)
    return tr, model, runner, counters


# ================================================================ 冻结表
def test_frozen_table_matches_plan():
    """冻结表语义与 §9.3 一致，且 C / D 各有自己的模块名。"""
    T = trainable_modules
    assert set(T(SubPhase.FIRST_DATA, "continuous")) == set()
    assert set(T(SubPhase.FIRST_DATA, "discrete")) == set()
    assert set(T(SubPhase.ADAPT, "continuous")) == {"phi", "psi"}
    # D 的表示是码本 + ψD（§16.1），对应 C 的 φ + ψ
    assert set(T(SubPhase.ADAPT, "discrete")) == {"codebook", "psi"}
    assert set(T(SubPhase.INITIAL_OFFLINE, "continuous")) == {"beta", "q"}
    assert set(T(SubPhase.INITIAL_OFFLINE, "discrete")) == {"actor", "q"}
    assert set(T(SubPhase.ONLINE, "continuous")) == {"beta", "q"}
    assert set(T(SubPhase.ONLINE, "discrete")) == {"actor", "q"}
    assert FROZEN_BY_SUBPHASE[SubPhase.ONLINE]["new_game_interaction"] == "current_stage_game_only"
    assert set(T(SubPhase.EVAL, "continuous")) == set()
    assert set(T(SubPhase.EVAL, "discrete")) == set()


def test_set_trainable_enforces_freeze_table():
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    set_trainable(m, SubPhase.ADAPT)
    assert all(p.requires_grad for p in m.phi.parameters())
    assert all(p.requires_grad for p in m.psi.parameters())
    assert all(not p.requires_grad for p in m.beta.parameters())
    assert all(not p.requires_grad for p in m.q1.parameters())

    set_trainable(m, SubPhase.ONLINE)
    assert all(not p.requires_grad for p in m.phi.parameters())
    assert all(not p.requires_grad for p in m.psi.parameters())
    assert all(p.requires_grad for p in m.beta.parameters())
    assert all(p.requires_grad for p in m.q1.parameters())

    set_trainable(m, SubPhase.FIRST_DATA)
    assert not any(p.requires_grad for p in m.parameters())


def test_assert_frozen_catches_violation():
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    # 在在线阶段把 φ 设成可训练 → 必须报错
    set_trainable(m, SubPhase.ONLINE)
    for p in m.phi.parameters():
        p.requires_grad_(True)
    with pytest.raises(ScheduleError, match="不应可训练"):
        assert_frozen(m, SubPhase.ONLINE, game="Flappybird")


# ================================================================ strict
def test_strict_runs_two_stages_with_exact_budget():
    cfg = small_cfg("strict")
    tr, model, runner, counters = build_trainer(cfg)
    out = tr.run()
    assert len(out["stages"]) == 2
    # 每个阶段：随机 20 步 + 在线 80 步 = 100
    for st in out["stages"]:
        assert st["first_data_steps"] == 20
        assert st["online_steps"] == 80
    assert out["total_online_steps"] == 160
    assert out["total_first_data_steps"] == 40
    assert runner.step_counts[0] == 100
    # strict 第二阶段不与 F 进行新的训练交互
    assert runner.step_counts[1] == 100


def test_strict_s2_does_not_sample_flappy_again():
    """§14 T7：strict S2 无 F 新训练样本。"""
    cfg = small_cfg("strict")
    tr, model, runner, counters = build_trainer(cfg)
    tr.run_stage("Flappybird", 100, first_visit=True)
    f_steps_after_s1 = runner.step_counts[0]
    tr.run_stage("Timberman", 100, first_visit=True)
    assert runner.step_counts[0] == f_steps_after_s1, "第二阶段不应再与 F 交互"


def test_adapt_and_offline_counts():
    """§14 T7：两次适配、一次初始离线。"""
    cfg = small_cfg("strict")
    tr, model, runner, counters = build_trainer(cfg)
    out = tr.run()
    assert len(counters["adapt_calls"]) == 2
    assert len(counters["offline_calls"]) == 1
    assert out["adapt_events"] == 2
    assert counters["offline_calls"][0][0] == "Flappybird"
    assert out["offline_updates"] == 5


def test_learner_updates_every_four_online_steps():
    cfg = small_cfg("strict")
    tr, model, runner, counters = build_trainer(cfg)
    out = tr.run()
    # 在线共 160 步 → 160/4 = 40 次 learner update
    assert out["counters"]["learner_updates"] == 40
    assert counters["updates"] == 40


def test_no_learner_update_during_first_data_or_adapt():
    cfg = small_cfg("strict")
    tr, model, runner, counters = build_trainer(cfg)
    tr.run_stage("Flappybird", 100, first_visit=True)
    # 随机 20 步不产生 learner update；在线 80 步产生 20 次
    assert counters["updates"] == 20


def test_mapping_version_increments_per_first_visit():
    cfg = small_cfg("strict")
    tr, model, runner, counters = build_trainer(cfg)
    out = tr.run()
    assert out["mapping_version"] == 2
    assert out["stages"][0]["mapping_version"] == 1
    assert out["stages"][1]["mapping_version"] == 2


def test_offline_pretrain_only_once_across_stages():
    cfg = small_cfg("strict")
    tr, model, runner, counters = build_trainer(cfg)
    tr.run()
    assert len(counters["offline_calls"]) == 1


def test_rehearsal_six_stages_no_extra_adapt_or_offline():
    """§14 T7：rehearsal 六阶段正确；返回旧游戏不重做适配/不重新初始化。"""
    cfg = small_cfg("rehearsal")
    tr, model, runner, counters = build_trainer(cfg)
    out = tr.run()
    assert len(out["stages"]) == 6
    assert len(counters["adapt_calls"]) == 2          # 只有两次首次接入
    assert len(counters["offline_calls"]) == 1
    steps = [st["online_steps"] for st in out["stages"]]
    assert steps == [40, 40, 40, 40, 20, 20]
    assert out["counters"]["repr_updates"] == 10      # 2 × 5


def test_rehearsal_totals_match_strict_online_steps():
    cfg_s = small_cfg("strict")
    cfg_r = small_cfg("rehearsal")
    tr_s, _, r_s, _ = build_trainer(cfg_s)
    tr_r, _, r_r, _ = build_trainer(cfg_r)
    out_s = tr_s.run()
    out_r = tr_r.run()
    # strict: 2×80=160 在线；rehearsal: 40+40+40+40+20+20=200 在线
    # 注意 rehearsal 各阶段步数不同（60/40/20），随机采集只在前两次
    assert out_r["total_first_data_steps"] == 40
    assert out_s["total_first_data_steps"] == 40
    # 在线总步数 = 总预算 - 随机采集
    assert out_r["total_online_steps"] == (60 + 60 + 40 + 40 + 20 + 20) - 40
    assert out_s["total_online_steps"] == (100 + 100) - 40


def test_budget_truncation_does_not_over_sample():
    """§14 T7：预算到点截断，不多采。"""
    cfg = small_cfg("strict")
    # episode_len=7 与在线预算 80 不整除（80/7=11.43）
    runner = FakeEnvRunner(episode_len=7)
    tr, model, runner, counters = build_trainer(cfg, runner)
    tr.run_stage("Flappybird", 100, first_visit=True)
    assert runner.step_counts[0] == 100          # 恰好 20 + 80，没有多一步
    reasons = [r for r in runner.episode_log]
    # 最后一步应当是操作性截断
    assert counters["episode_ends"][-1][2] is True or counters["episode_ends"][-1][1] is True


def test_stage_requires_enough_steps_for_first_data():
    cfg = small_cfg("strict")
    tr, model, runner, counters = build_trainer(cfg)
    with pytest.raises(ScheduleError, match="不足以覆盖"):
        tr.run_stage("Flappybird", 10, first_visit=True)


def test_online_loop_terminates_episodes_and_logs_reasons():
    cfg = small_cfg("strict")
    runner = FakeEnvRunner(episode_len=5)
    tr, model, runner, counters = build_trainer(cfg, runner)
    res = tr.run_stage("Flappybird", 100, first_visit=True)
    assert res.terminations.get("collision", 0) > 0
    assert len(res.episode_returns) > 0


def test_log_records_contain_phase_and_counters():
    cfg = small_cfg("strict")
    tr, model, runner, counters = build_trainer(cfg)
    tr.run()
    events = [r["event"] for r in tr.log_records if "event" in r]
    assert events.count("first_data_done") == 2
    assert events.count("adapt_done") == 2
    assert events.count("initial_offline_done") == 1
    assert events.count("learner_update") == 40
    assert all("random_steps" in r for r in tr.log_records)


def test_mapping_snapshot_before_and_after_adapt():
    """§9.4：适配前后的 β/Q 快照应保持一致（适配只动 φ/ψ）。"""
    cfg = small_cfg("strict")
    tr, model, runner, counters = build_trainer(cfg)
    before = model.subset_hash(["beta.", "q1.", "q2.", "target_q1.", "target_q2."])
    tr.run_stage("Flappybird", 100, first_visit=True)
    after = model.subset_hash(["beta.", "q1.", "q2.", "target_q1.", "target_q2."])
    # 本测试的假回调不真正更新 β/Q，因此 hash 应完全相同
    assert before == after


def test_single_game_schedule_uses_one_stage():
    import yaml

    raw = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    raw["train"]["random_steps_first_visit"] = 20
    raw["train"]["repr_updates_first_visit"] = 5
    raw["train"]["initial_offline_updates"] = 5
    raw["mode"] = "single"
    raw["schedule"] = "single"
    raw["game"] = "Timberman"
    cfg = validate_config(raw, test_only=True)
    tr, model, runner, counters = build_trainer(cfg)
    out = tr.run()
    assert len(out["stages"]) == 1
    assert out["stages"][0]["game"] == "Timberman"
    assert out["stages"][0]["first_data_steps"] == 20
    assert len(counters["offline_calls"]) == 1
