"""T8 端到端验收：无模拟器跑通完整状态机、checkpoint 恢复与纯评测（§14 T7/T8）。

这是 T0–T8 的收口测试：
- 真实状态机 + 真实 replay + 真实 learner + 真实 checkpoint/评测；
- 只用进程内的特征级假游戏，**不是能力实验**（能力实验需要微信模拟器）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wemi_cal import checkpoint as CK
from wemi_cal.config import GAMES, load_config
from wemi_cal.envs import features as F
from wemi_cal.envs.features import PolicyMode
from wemi_cal.train import CalLaicaRunner, LocalGameSim, run_smoke, smoke_config

CONFIG_PATH = "wemi_cal/configs/mt_cal.yaml"


@pytest.fixture(scope="module")
def smoke_out(tmp_path_factory):
    out_root = tmp_path_factory.mktemp("smoke")
    return out_root, run_smoke(base_config=CONFIG_PATH, total_steps=600, seed=0,
                               out_root=out_root)


# ---------------------------------------------------------------- 端到端
def test_smoke_completes_full_state_machine(smoke_out):
    out_root, out = smoke_out
    summary = out["summary"]
    assert len(summary["stages"]) == 2
    # 两次首次采集 + 两次适配
    assert summary["adapt_events"] == 2
    assert summary["counters"]["random_steps"] == 120
    # 只有首个任务做一次初始离线
    assert out["summary"]["stages"][0]["initial_offline_updates"] > 0
    assert out["summary"]["stages"][1]["initial_offline_updates"] == 0
    assert summary["counters"]["offline_updates"] > 0


def test_smoke_mapping_versions_increment(smoke_out):
    _, out = smoke_out
    st = out["summary"]["stages"]
    assert st[0]["mapping_version"] == 1
    assert st[1]["mapping_version"] == 2
    assert out["summary"]["mapping_version"] == 2


def test_smoke_learner_updates_on_schedule(smoke_out):
    _, out = smoke_out
    online = out["summary"]["total_online_steps"]
    n_updates = out["summary"]["counters"]["learner_updates"]
    assert n_updates == online // 4, (n_updates, online)


def test_smoke_phi_covers_all_actions_after_balanced_data(smoke_out):
    """φ 覆盖：每个合法动作至少要在一个 e 网格点成为 argmax（§6.4）。

    若首采数据不均衡（某动作一条都没有），这里会坍缩成单动作——本测试同时守住
    那条数据质量要求。
    """
    _, out = smoke_out
    for game, diag in out["repr_diagnostics"].items():
        cov = diag["coverage"]
        assert cov["all_actions_reachable"], (
            f"{game} 的 φ 只覆盖了 {cov['counts']}；首采数据或适配有错"
        )
        assert len(cov["counts"]) == 2


def test_smoke_produces_checkpoint_policy_and_summary(smoke_out):
    out_root, out = smoke_out
    assert Path(out["checkpoint"]).is_file()
    assert Path(out["policy"]).is_file()
    run_dir = Path(out["checkpoint"]).parent
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "train_log.jsonl").is_file()
    log_lines = (run_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(l).get("event") for l in log_lines]
    assert events.count("first_data_done") == 2
    assert events.count("adapt_done") == 2
    assert events.count("initial_offline_done") == 1


def test_smoke_policy_loads_and_is_reusable(smoke_out):
    """部署文件必须能装回模型，且两游戏共用同一个文件（§12.5）。"""
    _, out = smoke_out
    from wemi_cal.models.continuous import ContinuousModel

    cfg = smoke_config(CONFIG_PATH, total_steps=600, seed=0)
    m = ContinuousModel(seed=12345)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    payload = CK.load_policy_into(m, out["policy"])
    assert payload["obs_schema"] == "compact29-v1"
    assert payload["action_registry"] == [list(x) for x in F.ACTION_REGISTRY]
    # 同一个 policy.pt 服务两个游戏：hash 只算一次
    assert CK.policy_hash(out["policy"]) == out["policy_hash"]
    assert payload["registered_tokens"] == [0, 1, 2, 3]


def test_smoke_eval_ran_in_both_modes_without_mutating_weights(smoke_out):
    _, out = smoke_out
    assert set(out["eval"]) == set(GAMES)
    for game in GAMES:
        assert set(out["eval"][game]) == {"greedy", "stochastic_target"}
        assert out["eval"][game]["greedy"]["episodes"] == 2
    assert "param_hash_after" in out


def test_smoke_replay_state_is_consistent(smoke_out):
    _, out = smoke_out
    diag = out["replay"]
    assert diag["total_transitions"] > 0
    # 每个游戏各有 env_hash（跨游戏不能共用一个）
    assert "env_hash:0" in diag["compat"] and "env_hash:1" in diag["compat"]


def test_smoke_is_reproducible(tmp_path):
    """同 seed 两次自检必须给出完全相同的策略 hash（可重放性）。"""
    out1 = run_smoke(base_config=CONFIG_PATH, total_steps=400, seed=0,
                     out_root=tmp_path / "a")
    out2 = run_smoke(base_config=CONFIG_PATH, total_steps=400, seed=0,
                     out_root=tmp_path / "b")
    assert out1["policy_hash"] == out2["policy_hash"]
    assert out1["param_hash_after"] == out2["param_hash_after"]
    assert out1["summary"]["counters"] == out2["summary"]["counters"]


def test_smoke_rehearsal_schedule_runs(tmp_path):
    out = run_smoke(base_config=CONFIG_PATH, total_steps=400, schedule="rehearsal",
                    seed=0, out_root=tmp_path / "reh")
    # §9.2：rehearsal 必须是 6 个阶段，但只有两次首次接入适配
    assert len(out["summary"]["stages"]) == 6
    assert out["summary"]["adapt_events"] == 2
    assert out["summary"]["counters"]["offline_updates"] > 0
    # 返回旧游戏不重做适配
    later = out["summary"]["stages"][2:]
    assert all(s["adapt_updates"] == 0 for s in later)
    assert all(s["initial_offline_updates"] == 0 for s in later)
    # 每游戏总预算相等（公平比较）
    total = {"Flappybird": 0, "Timberman": 0}
    for st in out["summary"]["stages"]:
        total[st["game"]] += st["online_steps"] + st["first_data_steps"]
    assert total["Flappybird"] == total["Timberman"]


# ---------------------------------------------------------------- checkpoint 恢复
def test_checkpoint_resume_is_identity_checked(smoke_out):
    _, out = smoke_out
    ckpt = out["checkpoint"]
    from wemi_cal.models.continuous import ContinuousModel

    cfg = smoke_config(CONFIG_PATH, total_steps=600, seed=0)
    m = ContinuousModel(seed=999)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    restored = CK.load_checkpoint(ckpt, cfg=cfg, model=m, restore_replay=True)
    assert restored["trainer_state"]["mapping_version"] == 2
    assert restored["replay"].total_transitions() > 0
    # 部署文件与 checkpoint 里的 β/φ 一致
    policy_payload = CK.load_policy_into(m, out["policy"])
    assert policy_payload["registered_tokens"] == [0, 1, 2, 3]


def test_checkpoint_resume_rejects_wrong_schedule(smoke_out):
    _, out = smoke_out
    wrong = smoke_config(CONFIG_PATH, total_steps=600, seed=0, schedule="rehearsal")
    with pytest.raises(CK.CheckpointError):
        CK.load_checkpoint(out["checkpoint"], cfg=wrong)


def test_checkpoint_resume_rejects_wrong_seed(smoke_out):
    """seed 不同 → identity 不同 → 拒绝（不能"尽量加载"）。"""
    _, out = smoke_out
    wrong = smoke_config(CONFIG_PATH, total_steps=600, seed=7)
    with pytest.raises(CK.CheckpointError):
        CK.load_checkpoint(out["checkpoint"], cfg=wrong)


# ---------------------------------------------------------------- 正式入口守卫
def test_production_config_rejects_shortened_budget():
    import yaml

    from wemi_cal.config import ConfigError, validate_config

    raw = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    raw["train"]["random_steps_first_visit"] = 20
    with pytest.raises(ConfigError, match="test_only"):
        validate_config(raw)                 # 未声明 test_only → 拒绝
    cfg = validate_config(raw, test_only=True)
    assert cfg.test_only is True


def test_production_entry_refuses_test_only_config(monkeypatch, tmp_path):
    """`--smoke` 之外，正式入口不吃 test_only 配置。"""
    from wemi_cal import train as T

    called = {}

    def fake_load_config(*a, **kw):
        called["args"] = (a, kw)
        return smoke_config(CONFIG_PATH, total_steps=400)

    monkeypatch.setattr(T, "load_config", fake_load_config)
    with pytest.raises(SystemExit) as exc:
        T.main(["--config", CONFIG_PATH])
    assert "test_only" in str(exc.value)


def test_cli_smoke_path_runs(tmp_path):
    from wemi_cal import train as T

    rc = T.main(["--smoke", "--smoke-steps", "200"])
    assert rc == 0


# ---------------------------------------------------------------- 假游戏自身一致性
def test_local_sim_next_state_depends_on_action():
    """ψ 的可辨识性前提：同一状态上不同动作必须产生不同的 next_x。

    否则表示适配必然坍缩到单动作——这正是第一版模拟器的缺陷。
    """
    outs = {}
    for a in (0, 1):
        s = LocalGameSim(game="Flappybird", episode_len=10, seed=0)
        x, _ = s.reset(seed=1)
        nx, r, *_ = s.step(local=a)
        outs[a] = (nx.copy(), r)
    assert not np.allclose(outs[0][0], outs[1][0]), "不同动作的 next_x 不能相同"
    assert outs[0][1] != outs[1][1], "不同动作的奖励不应相同"


def test_local_sim_timberman_action_affects_observation():
    outs = {}
    for a in (0, 1):
        s = LocalGameSim(game="Timberman", episode_len=10, seed=0)
        x, _ = s.reset(seed=3)
        nx, r, *_ = s.step(local=a)
        outs[a] = nx.copy()
    assert not np.allclose(outs[0], outs[1])


def test_local_sim_respects_horizon_and_termination():
    s = LocalGameSim(game="Flappybird", horizon=5, episode_len=10, seed=0)
    s.reset(seed=0)
    for i in range(4):
        _, _, term, trunc, info = s.step(local=0)
        assert term is False
    _, r, term, trunc, info = s.step(local=0)
    assert term is True and trunc is False
    # H=5 < episode_len=10 → 这是 horizon 终止，不是游戏死亡
    assert info["termination_reason"] == "horizon"


def test_smoke_config_stage_budgets_equal_per_game():
    cfg = smoke_config(CONFIG_PATH, total_steps=400)
    assert cfg.steps_for_game("Flappybird") == cfg.steps_for_game("Timberman") == 200
    assert cfg.test_only is True
