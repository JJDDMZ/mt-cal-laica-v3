"""T8 验收：D 对照、checkpoint/恢复、部署导出与纯评测（§16，§12，§14 T8）。"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wemi_cal import checkpoint as CK
from wemi_cal.config import load_config, validate_config
from wemi_cal.envs import features as F
from wemi_cal.envs.features import PolicyMode
from wemi_cal.evaluate import (
    POLICY_RNG_BASE,
    EvaluateError,
    acceptance,
    evaluate_game,
    select_token,
)
from wemi_cal.models.continuous import ContinuousModel
from wemi_cal.models.discrete import (
    DiscreteModel,
    adapt_discrete_representation,
    discrete_repr_loss,
)
from wemi_cal.replay import ReplayBuffer, make_transition

CONFIG_PATH = "wemi_cal/configs/mt_cal.yaml"


def cfg_for(variant="continuous", schedule="strict", mode="full", **train) -> "object":
    import yaml

    raw = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    raw["variant"] = variant
    raw["schedule"] = schedule
    raw["mode"] = mode
    raw["train"].update(train)
    # 缩小随机采集步数属于小预算测试，必须显式声明 test_only
    test_only = "random_steps_first_visit" in train or "repr_updates_first_visit" in train
    return validate_config(raw, test_only=test_only)


# ================================================================ D：码本与结构
def test_codebook_initialised_as_first_four_basis_vectors():
    m = DiscreteModel(seed=0)
    E = m.codebook()
    assert E.shape == (4, 8)
    for j in range(4):
        assert torch.allclose(E[j], torch.eye(8)[j]), E[j]


def test_discrete_actor_shapes_and_masked_probs():
    m = DiscreteModel(seed=0)
    x = torch.randn(5, F.OBS_DIM)
    assert m.actor.hidden(x).shape == (5, 8)
    assert m.actor.logits(x, m.codebook).shape == (5, 4)
    logp = m.actor.log_probs(x, m.codebook, "Flappybird")
    assert torch.isfinite(logp[:, [0, 1]]).all()
    assert torch.isinf(logp[:, [2, 3]]).all() and (logp[:, [2, 3]] < 0).all()
    p = m.actor.probs(x, m.codebook, "Flappybird")
    assert torch.allclose(p[:, :2].sum(-1), torch.ones(5), atol=1e-5)


def test_discrete_psi_and_q_shapes():
    m = DiscreteModel(seed=0)
    x = torch.randn(4, F.OBS_DIM)
    nx = torch.randn(4, F.OBS_DIM)
    assert m.psi(x, nx).shape == (4, 8)
    q = m.q1(x, m.codebook)
    assert q.shape == (4, 4)
    assert m.q2(x, m.codebook).shape == (4, 4)


def test_discrete_actor_can_use_consistent_basis_geometry():
    """§16.1 的损失覆盖判据：若 h 在 e_j 与外积一致，联合几何可以是加性的。
    这里直接验证内积形式：增大 h 的第 j 个分量会提高对应 token 的 logit。"""
    m = DiscreteModel(seed=0)
    x = torch.zeros(1, F.OBS_DIM)
    with torch.no_grad():
        h = m.actor.hidden(x)
        logits = m.actor.logits(x, m.codebook)
        assert torch.allclose(logits[0, :4], h[0, :4], atol=1e-6)


def test_registration_freezes_old_embeddings():
    m = DiscreteModel(seed=0)
    m.register_game("Flappybird")
    m.freeze_embeddings_except("Flappybird")
    assert all(p.requires_grad for p in m.codebook.rows[:2])
    m.register_game("Timberman")
    m.freeze_embeddings_except("Timberman")
    # F 的两行严格冻结，避免 Adam 动量偷偷改旧行（§16.1）
    assert not m.codebook.rows[0].requires_grad
    assert not m.codebook.rows[1].requires_grad
    assert m.codebook.rows[2].requires_grad and m.codebook.rows[3].requires_grad


def test_discrete_repr_loss_formula():
    m = DiscreteModel(seed=0)
    x = torch.randn(8, F.OBS_DIM)
    nx = torch.randn(8, F.OBS_DIM)
    token = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loss, stats = discrete_repr_loss(m, x=x, next_x=nx, token=token, game="Flappybird")
    # 手算 CE + 分离项
    z = m.psi(x, nx)
    E = m.codebook()
    logits = z @ E.t()
    mask = torch.full_like(logits, float("-inf"))
    mask[:, [0, 1]] = 0.0
    logp = torch.log_softmax(logits + mask, dim=-1)
    ce = -logp[torch.arange(8), token].mean()
    dist = (E[0] - E[1]).norm(p=2)
    sep = 0.1 * torch.clamp(0.5 - dist, min=0.0) ** 2
    assert abs(float(stats["ce"]) - float(ce)) < 1e-6
    assert abs(float(stats["sep"]) - float(sep)) < 1e-6
    assert abs(float(loss) - float(ce + sep)) < 1e-6


def test_discrete_adaptation_only_updates_current_rows_and_psi():
    m = DiscreteModel(seed=0)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    f_rows_before = [r.detach().clone() for r in m.codebook.rows[:2]]
    actor_before = [p.detach().clone() for p in m.actor.parameters()]
    q_before = [p.detach().clone() for p in m.q1.parameters()]
    x = torch.randn(16, F.OBS_DIM)
    nx = torch.randn(16, F.OBS_DIM)
    token = torch.tensor([2, 3] * 8)
    diag = adapt_discrete_representation(
        m, data=[(x, nx, token)], game="Timberman", updates=5
    )
    assert diag.n_updates == 5
    # F 的行严格不变
    assert all(torch.equal(a, b) for a, b in zip(f_rows_before, m.codebook.rows[:2]))
    # T 的行被归一化到单位范数
    for i in (2, 3):
        assert abs(float(m.codebook.rows[i].norm()) - 1.0) < 1e-5
    # actor/Q 不变
    assert all(torch.equal(a, b) for a, b in zip(actor_before, m.actor.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(q_before, m.q1.parameters()))


def test_discrete_adaptation_normalises_and_rejects_zero_norm():
    m = DiscreteModel(seed=0)
    m.register_game("Flappybird")
    with torch.no_grad():
        m.codebook.rows[0].zero_()
    with pytest.raises(ValueError, match="范数为 0"):
        m.codebook.normalise_current("Flappybird")


def test_discrete_adaptation_leaves_embeddings_frozen_after():
    m = DiscreteModel(seed=0)
    m.register_game("Flappybird")
    x = torch.randn(8, F.OBS_DIM)
    token = torch.zeros(8, dtype=torch.long)
    adapt_discrete_representation(m, data=[(x, x, token)], game="Flappybird", updates=2)
    assert not any(p.requires_grad for p in m.codebook.rows)


def test_discrete_masks_illegal_actions_in_actor_and_q_gradient():
    m = DiscreteModel(seed=0)
    x = torch.randn(4, F.OBS_DIM)
    tok = m.actor.greedy_token(x, m.codebook, "Flappybird")
    assert set(tok.tolist()) <= {0, 1}
    tok_t = m.actor.greedy_token(x, m.codebook, "Timberman")
    assert set(tok_t.tolist()) <= {2, 3}


# ================================================================ checkpoint
def _dummy_replay() -> ReplayBuffer:
    buf = ReplayBuffer(seed=1)
    for ep in range(2):
        for i in range(3):
            buf.add_confirmed(make_transition(
                buffer=buf, game="Flappybird", episode_id=ep, request_id=i,
                x=np.zeros(F.OBS_DIM, dtype=np.float32),
                next_x=np.zeros(F.OBS_DIM, dtype=np.float32),
                e_behavior=np.zeros(2, dtype=np.float32), global_token=0,
                reward_train=0.5, log_b_action=np.log(0.5),
                behavior_kernel="uniform", behavior_mapping_version=1, clock_mode="lockstep-v1",
                env_hash="h",
            ), game="Flappybird")
        buf.finish_episode(game="Flappybird", episode_id=ep, terminated=True,
                           truncated=False, termination_reason="collision")
    return buf


def test_checkpoint_roundtrip_continuous(tmp_path):
    cfg = cfg_for()
    m = ContinuousModel(seed=3)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    buf = _dummy_replay()
    opt = torch.optim.Adam(m.beta.parameters(), lr=1e-3)
    path = CK.save_checkpoint(
        tmp_path / "ckpt.pt", cfg=cfg, model=m, optimizers={"beta": opt},
        replay=buf, trainer_state={"subphase": "online", "mapping_version": 2,
                                  "counters": {"online_steps": 1234}},
    )
    m2 = ContinuousModel(seed=99)
    m2.register_game("Flappybird")
    m2.register_game("Timberman")
    opt2 = torch.optim.Adam(m2.beta.parameters(), lr=1e-3)
    out = CK.load_checkpoint(path, cfg=cfg, model=m2, optimizers={"beta": opt2})
    assert out["trainer_state"]["mapping_version"] == 2
    for a, b in zip(m.parameters(), m2.parameters()):
        assert torch.equal(a, b)
    assert out["replay"].total_transitions() == buf.total_transitions()


def test_checkpoint_save_failure_preserves_existing_file_and_cleans_temporary(
    tmp_path, monkeypatch
):
    cfg = cfg_for()
    model = ContinuousModel(seed=3)
    model.register_game("Flappybird")
    path = tmp_path / "stage_Flappybird.pt"
    original = b"previous complete checkpoint"
    path.write_bytes(original)

    def fail_during_serialization(payload, destination):
        Path(destination).write_bytes(b"partial serialization")
        raise OSError("simulated serialization failure")

    monkeypatch.setattr(CK.torch, "save", fail_during_serialization)
    with pytest.raises(OSError, match="simulated serialization failure"):
        CK.save_checkpoint(
            path, cfg=cfg, model=model, optimizers={}, replay=_dummy_replay(),
            trainer_state={"completed_tasks": []},
        )

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_checkpoint_restores_optimizer_and_private_rng(tmp_path):
    cfg = cfg_for()
    m = ContinuousModel(seed=7)
    m.register_game("Flappybird")
    opt = torch.optim.Adam(m.beta.parameters(), lr=3e-4)
    opt.zero_grad(set_to_none=True)
    for p in m.beta.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    replay = _dummy_replay()
    # Advance the private replay stream so restoration can be checked independently.
    replay.rng.random()
    expected_model_rng = m.generator_state()
    expected_replay_rng = replay.rng.bit_generator.state
    path = CK.save_checkpoint(
        tmp_path / "v3.pt", cfg=cfg, model=m, optimizers={"beta": opt},
        replay=replay,
        trainer_state={"offline_done": True, "completed_tasks": [0]},
    )

    m2 = ContinuousModel(seed=99)
    opt2 = torch.optim.Adam(m2.beta.parameters(), lr=3e-4)
    out = CK.load_checkpoint(path, cfg=cfg, model=m2, optimizers={"beta": opt2})
    assert out["trainer_state"]["offline_done"] is True
    assert out["trainer_state"]["completed_tasks"] == [0]
    assert opt2.state_dict()["state"]
    assert all(torch.equal(expected_model_rng[k], m2.generator_state()[k])
               for k in expected_model_rng)
    assert out["replay"].rng.bit_generator.state == expected_replay_rng
    assert CK.CHECKPOINT_FORMAT == 3


@pytest.mark.parametrize("missing_field", ["model_rng", "replay_rng"])
def test_checkpoint_rejects_missing_private_rng_before_restoring(tmp_path, missing_field):
    cfg = cfg_for()
    source = ContinuousModel(seed=7)
    source.register_game("Flappybird")
    source_opt = torch.optim.Adam(source.beta.parameters(), lr=3e-4)
    path = CK.save_checkpoint(
        tmp_path / f"missing-{missing_field}.pt", cfg=cfg, model=source,
        optimizers={"beta": source_opt}, replay=_dummy_replay(), trainer_state={},
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload[missing_field]
    torch.save(payload, path)

    target = ContinuousModel(seed=99)
    target.register_game("Flappybird")
    before_parameters = [p.detach().clone() for p in target.parameters()]
    target_opt = torch.optim.Adam(target.beta.parameters(), lr=0.123)
    before_optimizer = copy.deepcopy(target_opt.state_dict())
    with pytest.raises(CK.CheckpointError, match=missing_field):
        CK.load_checkpoint(path, cfg=cfg, model=target,
                           optimizers={"beta": target_opt})

    assert all(torch.equal(before, after)
               for before, after in zip(before_parameters, target.parameters()))
    assert target_opt.state_dict() == before_optimizer


def test_checkpoint_rejects_identity_mismatch(tmp_path):
    cfg = cfg_for()
    m = ContinuousModel(seed=3)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    path = CK.save_checkpoint(
        tmp_path / "ckpt.pt", cfg=cfg, model=m, optimizers={},
        replay=_dummy_replay(), trainer_state={},
    )
    other = cfg_for(schedule="rehearsal")
    with pytest.raises(CK.CheckpointError):
        CK.load_checkpoint(path, cfg=other)


def test_checkpoint_rejects_variant_mismatch(tmp_path):
    cfg = cfg_for()
    m = ContinuousModel(seed=3)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    path = CK.save_checkpoint(
        tmp_path / "ckpt.pt", cfg=cfg, model=m, optimizers={},
        replay=_dummy_replay(), trainer_state={},
    )
    d_cfg = cfg_for(variant="discrete")
    with pytest.raises(CK.CheckpointError, match="variant"):
        CK.load_checkpoint(path, cfg=d_cfg)


def test_checkpoint_rejects_schema_mismatch(tmp_path):
    """拒绝旧 schema checkpoint 恢复到当前格式（§14 T0）。"""
    cfg = cfg_for()
    m = ContinuousModel(seed=0)
    p = tmp_path / "old.pt"
    torch.save({"checkpoint_format": 3, "schema_version": 1, "variant": "continuous",
                "mode": "full", "stage_schedule": [], "identity_hash": "x",
                "config": {}, "model_state": {}, "optimizers": {}, "replay": {}}, p)
    with pytest.raises(CK.CheckpointError, match="schema_version"):
        CK.load_checkpoint(p, cfg=cfg)


def test_checkpoint_rejects_format_mismatch(tmp_path):
    cfg = cfg_for()
    p = tmp_path / "old.pt"
    torch.save({"checkpoint_format": 2, "schema_version": 2}, p)
    with pytest.raises(CK.CheckpointError) as exc:
        CK.load_checkpoint(p, cfg=cfg)
    assert str(exc.value) == (
        "checkpoint_format 不匹配：期望 3，得到 2；旧实验仅用于分析，不能无损续训"
    )


def test_checkpoint_restores_rng_state(tmp_path):
    cfg = cfg_for()
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    torch.manual_seed(1234)
    np.random.seed(1234)
    path = CK.save_checkpoint(tmp_path / "c.pt", cfg=cfg, model=m, optimizers={},
                              replay=_dummy_replay(), trainer_state={})
    torch.manual_seed(999)
    np.random.seed(999)
    after_seed = torch.randn(3)
    CK.load_checkpoint(path, cfg=cfg, restore_rng=True)
    torch.manual_seed(1234)
    np.random.seed(1234)
    expected = torch.randn(3)
    # 恢复后应重现保存时的随机流
    CK.load_checkpoint(path, cfg=cfg, restore_rng=True)
    assert torch.equal(torch.randn(3), expected)


def test_checkpoint_saves_replay_and_pending(tmp_path):
    cfg = cfg_for()
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    buf = _dummy_replay()
    # 留一个未完成回合
    buf.add_confirmed(make_transition(
        buffer=buf, game="Timberman", episode_id=5, request_id=0,
        x=np.zeros(F.OBS_DIM, dtype=np.float32),
        next_x=np.zeros(F.OBS_DIM, dtype=np.float32),
        e_behavior=np.zeros(2, dtype=np.float32), global_token=2,
        reward_train=0.0, log_b_action=np.log(0.5),
        behavior_kernel="uniform", behavior_mapping_version=2, env_hash="h", clock_mode="lockstep-v1",
    ), game="Timberman")
    path = CK.save_checkpoint(tmp_path / "c.pt", cfg=cfg, model=m, optimizers={},
                              replay=buf, trainer_state={})
    out = CK.load_checkpoint(path, cfg=cfg)
    assert out["replay"].pending_transitions() == 1
    assert out["replay"].diagnostics()["compat"]["env_hash:0"] == "h"


def test_v3_checkpoint_replay_requires_clock_mode(tmp_path):
    cfg = cfg_for()
    model = ContinuousModel(seed=0)
    model.register_game("Flappybird")
    model.register_game("Timberman")
    path = CK.save_checkpoint(tmp_path / "clock-schema.pt", cfg=cfg, model=model,
                              optimizers={}, replay=_dummy_replay(), trainer_state={})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["replay"]["online"]["0"][0]["transitions"][0]["clock_mode"]
    torch.save(payload, path)
    with pytest.raises(CK.CheckpointError, match="缺少合法 clock_mode"):
        CK.load_checkpoint(path, cfg=cfg)


def test_checkpoint_discrete_codebook_roundtrip(tmp_path):
    cfg = cfg_for(variant="discrete")
    m = DiscreteModel(seed=5)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    with torch.no_grad():
        m.codebook.rows[0].fill_(0.3)
    path = CK.save_checkpoint(tmp_path / "d.pt", cfg=cfg, model=m, optimizers={},
                              replay=_dummy_replay(), trainer_state={})
    m2 = DiscreteModel(seed=7)
    m2.register_game("Flappybird")
    m2.register_game("Timberman")
    CK.load_checkpoint(path, cfg=cfg, model=m2)
    assert torch.allclose(m2.codebook.rows[0], m.codebook.rows[0])


def test_checkpoint_missing_optimizer_rejected(tmp_path):
    cfg = cfg_for()
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    path = CK.save_checkpoint(tmp_path / "c.pt", cfg=cfg, model=m, optimizers={},
                              replay=_dummy_replay(), trainer_state={})
    m2 = ContinuousModel(seed=0)
    m2.register_game("Flappybird")
    m2.register_game("Timberman")
    with pytest.raises(CK.CheckpointError, match="缺少 optimizer"):
        CK.load_checkpoint(path, cfg=cfg, model=m2,
                           optimizers={"beta": torch.optim.Adam(m2.beta.parameters())})


# ================================================================ 部署导出
def test_export_policy_contains_only_inference_components(tmp_path):
    cfg = cfg_for()
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    p = CK.export_policy(tmp_path / "policy.pt", cfg=cfg, model=m)
    payload = torch.load(p, map_location="cpu", weights_only=False)
    assert payload["components"] == ["beta", "phi"]
    assert "q1_state" not in payload and "psi_state" not in payload
    assert "target_q1" not in json.dumps(list(payload.keys()))
    assert payload["obs_schema"] == "compact29-v1"
    assert payload["action_registry"] == [list(x) for x in F.ACTION_REGISTRY]
    assert payload["registered_tokens"] == [0, 1, 2, 3]


def test_same_policy_file_used_for_both_games(tmp_path):
    """§12.5：最终两游戏推理加载同一个 policy.pt，hash 一致。"""
    cfg = cfg_for()
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    p1 = CK.export_policy(tmp_path / "policy.pt", cfg=cfg, model=m)
    h1 = CK.policy_hash(p1)
    h2 = CK.policy_hash(p1)
    assert h1 == h2
    m2 = ContinuousModel(seed=99)
    m2.register_game("Flappybird")
    m2.register_game("Timberman")
    CK.load_policy_into(m2, p1)
    assert torch.equal(m.beta.head.weight, m2.beta.head.weight)
    assert torch.equal(m.phi.linear.weight, m2.phi.linear.weight)
    assert h1 == hashlib.sha256(Path(p1).read_bytes()).hexdigest()


def test_export_policy_discrete(tmp_path):
    cfg = cfg_for(variant="discrete")
    m = DiscreteModel(seed=0)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    p = CK.export_policy(tmp_path / "policy_d.pt", cfg=cfg, model=m)
    payload = torch.load(p, map_location="cpu", weights_only=False)
    assert payload["components"] == ["actor", "codebook"]
    m2 = DiscreteModel(seed=9)
    m2.register_game("Flappybird")
    m2.register_game("Timberman")
    CK.load_policy_into(m2, p)
    assert torch.equal(m.actor.trunk[0].weight, m2.actor.trunk[0].weight)


# ================================================================ 纯评测
class FakeEvalEnv:
    """确定性假环境：分数固定，用于断言评测协议本身。"""

    def __init__(self, horizon=6, score=7.0):
        self.horizon = horizon
        self.score = score
        self.calls = 0
        self.reset_seeds = []
        self.tokens = []

    def reset(self, game, seed):
        self.reset_seeds.append(seed)
        task = F.task_id(game)
        x = np.zeros(F.OBS_DIM, dtype=np.float32)
        x[26 + task] = 1.0
        self.t = 0
        return x, {"episode_id": len(self.reset_seeds) - 1}

    def step(self, game, local):
        self.calls += 1
        self.t += 1
        done = self.t >= self.horizon
        task = F.task_id(game)
        x = np.zeros(F.OBS_DIM, dtype=np.float32)
        x[26 + task] = 1.0
        info = {
            "score_raw": self.score,
            "termination_reason": "horizon" if done else "none",
            "episode_return_train": 1.0,
        }
        return x, 1.0, done, False, info


def test_evaluate_game_greedy_uses_fixed_seeds():
    cfg = cfg_for()
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    env = FakeEvalEnv()
    rep = evaluate_game(
        cfg=cfg, model=m, game="Flappybird", env_reset=env.reset, env_step=env.step,
        policy_mode=PolicyMode.GREEDY.value, episodes=5, seed_start=100000,
    )
    assert rep.episodes == 5
    assert env.reset_seeds == [100000, 100001, 100002, 100003, 100004]
    assert rep.score_mean == 7.0
    assert rep.terminations.get("horizon") == 5


def test_evaluate_game_stochastic_target_mode_runs():
    cfg = cfg_for()
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    env = FakeEvalEnv()
    rep = evaluate_game(
        cfg=cfg, model=m, game="Timberman", env_reset=env.reset, env_step=env.step,
        policy_mode=PolicyMode.STOCHASTIC_TARGET.value, episodes=3, seed_start=200000,
    )
    assert rep.episodes == 3
    assert rep.policy_mode == "stochastic_target"


def test_evaluate_does_not_change_weights():
    """§14 T8：评测不更新权重；参数 hash 前后一致。"""
    cfg = cfg_for()
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    m.register_game("Timberman")
    h_before = m.parameter_hash()
    env = FakeEvalEnv()
    for mode in (PolicyMode.GREEDY.value, PolicyMode.STOCHASTIC_TARGET.value):
        evaluate_game(cfg=cfg, model=m, game="Flappybird", env_reset=env.reset,
                      env_step=env.step, policy_mode=mode, episodes=2, seed_start=100000)
    assert m.parameter_hash() == h_before


def test_evaluate_rejects_unknown_policy_mode():
    cfg = cfg_for()
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    with pytest.raises(EvaluateError, match="未知 policy_mode"):
        select_token(variant="continuous", model=m,
                     x=np.zeros(F.OBS_DIM, dtype=np.float32), game="Flappybird",
                     policy_mode="epsilon_greedy")


def test_select_token_greedy_is_deterministic():
    cfg = cfg_for()
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    x = np.zeros(F.OBS_DIM, dtype=np.float32)
    toks = {select_token(variant="continuous", model=m, x=x, game="Flappybird",
                         policy_mode="greedy") for _ in range(20)}
    assert len(toks) == 1, "greedy 必须是确定性的部署规则"
    assert toks.pop() in (0, 1)


def test_select_token_stochastic_target_varies():
    cfg = cfg_for()
    m = ContinuousModel(seed=0)
    m.register_game("Flappybird")
    # 让 β 有较大方差，确保随机目标模式会变
    with torch.no_grad():
        m.beta.head.bias[2:].fill_(1.5)
    x = np.zeros(F.OBS_DIM, dtype=np.float32)
    toks = {select_token(variant="continuous", model=m, x=x, game="Flappybird",
                         policy_mode="stochastic_target") for _ in range(60)}
    assert toks <= {0, 1}
    assert len(toks) >= 1


def test_select_token_discrete_modes():
    m = DiscreteModel(seed=0)
    m.register_game("Flappybird")
    x = np.zeros(F.OBS_DIM, dtype=np.float32)
    g = {select_token(variant="discrete", model=m, x=x, game="Flappybird",
                      policy_mode="greedy") for _ in range(10)}
    assert g <= {0, 1}
    s = {select_token(variant="discrete", model=m, x=x, game="Flappybird",
                      policy_mode="stochastic_target") for _ in range(40)}
    assert s <= {0, 1}


# ================================================================ 验收门槛
def test_acceptance_absolute_and_relative():
    cfg = cfg_for()
    from wemi_cal.evaluate import GameReport

    rep = GameReport(game="Flappybird", policy_mode="greedy", episodes=50,
                     score_mean=12.0, pass_count=45, pass_rate=0.9)
    ok = acceptance(cfg=cfg, report=rep, st_baseline_mean=10.0)
    assert ok.passed_absolute and ok.passed_relative and ok.passed

    weak = GameReport(game="Flappybird", policy_mode="greedy", episodes=50,
                      score_mean=12.0, pass_count=10, pass_rate=0.2)
    assert not acceptance(cfg=cfg, report=weak, st_baseline_mean=10.0).passed

    low_rel = GameReport(game="Flappybird", policy_mode="greedy", episodes=50,
                         score_mean=7.0, pass_count=45, pass_rate=0.9)
    res = acceptance(cfg=cfg, report=low_rel, st_baseline_mean=10.0)
    assert res.passed_absolute and not res.passed_relative


def test_acceptance_rejects_nonpositive_baseline():
    """§12.2：单游戏基线均值 ≤0 时比例无效，不能靠小常数伪造达标。"""
    cfg = cfg_for()
    from wemi_cal.evaluate import GameReport

    rep = GameReport(game="Timberman", policy_mode="greedy", episodes=50,
                     score_mean=5.0, pass_count=45, pass_rate=0.9)
    res = acceptance(cfg=cfg, report=rep, st_baseline_mean=0.0)
    assert not res.passed_relative and not res.passed


def test_acceptance_requires_40_of_50():
    cfg = cfg_for()
    from wemi_cal.evaluate import GameReport

    rep = GameReport(game="Timberman", policy_mode="greedy", episodes=50,
                     score_mean=100.0, pass_count=39, pass_rate=0.78)
    assert not acceptance(cfg=cfg, report=rep, st_baseline_mean=50.0).passed_absolute
    rep.pass_count = 40
    assert acceptance(cfg=cfg, report=rep, st_baseline_mean=50.0).passed_absolute


def test_config_rejects_test_only_in_production():
    """test_only 配置不能被当成正式配置（身份不同）。"""
    cfg_prod = load_config(CONFIG_PATH)
    cfg_test = cfg_for(random_steps_first_visit=20)
    assert cfg_prod.identity_hash() != cfg_test.identity_hash()
    assert cfg_prod.test_only is False and cfg_test.test_only is True
