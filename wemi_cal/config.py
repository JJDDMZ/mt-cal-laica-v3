"""配置加载、校验与运行身份（技术方案 §10，任务 T0）。

设计约束：
- ``variant`` / ``schedule`` / ``mode`` 三个维度独立，加载时校验。
- 完整配置写入 checkpoint，并生成稳定的身份哈希；身份不匹配的恢复必须被拒绝。
- 这里不猜测任何数值：所有阈值都来自 configs/mt_cal.yaml。
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

SCHEMA_VERSION = 2

VARIANTS = ("continuous", "discrete")
SCHEDULES = ("strict", "rehearsal", "single")
# §10：full/replay/uncalibrated 对应 §7 三种正则设置；single 为单游戏单阶段。
MODES = ("full", "replay", "uncalibrated", "single")

GAMES = ("Flappybird", "Timberman")
TASK_ID = {"Flappybird": 0, "Timberman": 1}

# λcal 按 mode 决定；replay 取 0，uncalibrated 用相同 valid 行但改 max 操作（见 learner.py）。
CALIBRATION_MODE = {
    "full": "max",
    "replay": "off",
    "uncalibrated": "linear",
    "single": "off",
}


class ConfigError(ValueError):
    """配置不合法：不静默修正，直接让调用方看到。"""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


@dataclass
class Config:
    """已校验的完整配置。"""

    raw: Dict[str, Any]
    variant: str
    schedule: str
    mode: str
    seed: int
    game: Optional[str] = None  # 仅 mode=single 使用
    source_path: Optional[str] = None
    stage_schedule: List[Dict[str, Any]] = field(default_factory=list)
    test_only: bool = False

    # ---- 常用只读视图 ----
    @property
    def env(self) -> Dict[str, Any]:
        return self.raw["env"]

    @property
    def continuous(self) -> Dict[str, Any]:
        return self.raw["continuous"]

    @property
    def discrete(self) -> Dict[str, Any]:
        return self.raw["discrete"]

    @property
    def replay(self) -> Dict[str, Any]:
        return self.raw["replay"]

    @property
    def train(self) -> Dict[str, Any]:
        return self.raw["train"]

    @property
    def eval(self) -> Dict[str, Any]:
        return self.raw["eval"]

    @property
    def horizon(self) -> int:
        return int(self.env["horizon"])

    @property
    def gamma(self) -> float:
        return float(self.train["gamma"])

    @property
    def lambda_cal(self) -> float:
        return 0.0 if self.mode in ("replay", "single") else float(self.train["lambda_cal"])

    @property
    def calibration_mode(self) -> str:
        return CALIBRATION_MODE[self.mode]

    @property
    def total_train_steps(self) -> int:
        return sum(int(s["steps"]) for s in self.stage_schedule)

    def steps_for_game(self, game: str) -> int:
        return sum(int(s["steps"]) for s in self.stage_schedule if s["game"] == game)

    def schema_fingerprint(self) -> Dict[str, Any]:
        """影响 replay / checkpoint 兼容性的结构字段（§13：env_hash 与 mapping_version 分开）。"""
        c = self.continuous
        d = self.discrete
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": self.env["protocol"],
            "clock_mode": self.env["clock_mode"],
            "obs_schema": self.env["obs_schema"],
            "reward_schema": self.env["reward_schema"],
            "horizon": self.env["horizon"],
            "obs_dim": 29,
            "latent_dim": int(c["latent_dim"]),
            "hidden_sizes": list(c["hidden_sizes"]),
            "decoder": c["decoder"],
            "embedding_dim": int(d["embedding_dim"]),
            "variant": self.variant,
        }

    def identity_hash(self) -> str:
        """运行身份哈希：variant/schedule/mode/seed + 结构字段。

        恢复时必须比对；任何一项不匹配都拒绝加载，不能靠“网络维度恰好相同”放行。
        """
        payload = {
            "experiment": self.raw["experiment"],
            "variant": self.variant,
            "schedule": self.schedule,
            "mode": self.mode,
            "seed": self.seed,
            "game": self.game,
            "stage_schedule": self.stage_schedule,
            "test_only": self.test_only,
            "training_constants": {k: int(self.train[k]) for k in PRODUCTION_CONSTANTS},
            "schema": self.schema_fingerprint(),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.raw)


def load_config(
    path: str | Path,
    *,
    variant: Optional[str] = None,
    schedule: Optional[str] = None,
    mode: Optional[str] = None,
    seed: Optional[int] = None,
    game: Optional[str] = None,
    test_only: bool = False,
) -> Config:
    """读取 YAML + 应用 CLI override + 校验 + 展开 stage_schedule。"""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"配置文件不存在: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - 取决于输入文件
        raise ConfigError(f"YAML 解析失败: {exc}") from exc
    _require(isinstance(raw, dict), "配置根节点必须是映射")

    if variant is not None:
        raw["variant"] = variant
    if schedule is not None:
        raw["schedule"] = schedule
    if mode is not None:
        raw["mode"] = mode
    if seed is not None:
        raw["seed"] = seed

    return validate_config(raw, game=game, source_path=str(p), test_only=test_only)


# 正式配置里被钉死的常量；test_only 配置允许在给定区间内缩小，便于无模拟器的小预算测试。
PRODUCTION_CONSTANTS = {
    "random_steps_first_visit": 2000,
    "repr_updates_first_visit": 1000,
    "initial_offline_updates": 1000,
    "update_every_online_steps": 4,
}
TEST_ONLY_RANGES = {
    "random_steps_first_visit": (2, 2000),
    "repr_updates_first_visit": (1, 1000),
    "initial_offline_updates": (1, 1000),
    "update_every_online_steps": (1, 64),
}


def validate_config(
    raw: Mapping[str, Any],
    *,
    game: Optional[str] = None,
    source_path: Optional[str] = None,
    test_only: bool = False,
) -> Config:
    """校验配置并展开阶段表。所有失败都抛 ConfigError。

    ``test_only=True`` 显式声明这是小预算测试配置：允许缩小
    ``random_steps_first_visit`` / ``repr_updates_first_visit`` /
    ``initial_offline_updates`` / ``update_every_online_steps``，
    但这些值会真实写入身份哈希与 checkpoint。**绝不能**把 test_only 配置交给
    正式实验入口（``train.py`` 会拒绝）。
    """
    cfg_raw = copy.deepcopy(dict(raw))
    cfg_raw["_test_only"] = bool(test_only)

    for key in ("experiment", "variant", "schedule", "mode", "seed", "env",
                "continuous", "discrete", "replay", "train", "eval"):
        _require(key in cfg_raw, f"配置缺少字段: {key}")

    _require(int(cfg_raw.get("schema_version", 0)) == SCHEMA_VERSION,
             f"schema_version 必须是 {SCHEMA_VERSION}（拒绝 v1 checkpoint 配置）")

    variant = cfg_raw["variant"]
    schedule = cfg_raw["schedule"]
    mode = cfg_raw["mode"]
    seed = int(cfg_raw["seed"])

    _require(variant in VARIANTS, f"variant 必须是 {VARIANTS}，得到 {variant!r}")
    _require(schedule in SCHEDULES, f"schedule 必须是 {SCHEDULES}，得到 {schedule!r}")
    _require(mode in MODES, f"mode 必须是 {MODES}，得到 {mode!r}")

    # ---- 环境块 ----
    env = cfg_raw["env"]
    protocol = str(env["protocol"])
    clock_mode = str(env["clock_mode"])
    _require(protocol in ("wemi-cal-v1", "wemi-rt-v1"), f"未知 protocol: {protocol}")
    _require(clock_mode in ("lockstep-v1", "realtime-v1"), f"未知 clock_mode: {clock_mode}")
    _require(
        (protocol, clock_mode) in {
            ("wemi-cal-v1", "lockstep-v1"),
            ("wemi-rt-v1", "realtime-v1"),
        },
        f"protocol/clock_mode 不匹配: {protocol}/{clock_mode}",
    )
    _require(env["obs_schema"] == "compact29-v1", f"obs_schema 必须是 compact29-v1，得到 {env['obs_schema']!r}")
    _require(env["reward_schema"] == "bounded-v1", f"reward_schema 必须是 bounded-v1，得到 {env['reward_schema']!r}")
    _require(int(env["horizon"]) > 0, "horizon 必须为正")
    ports = env["ports"]
    _require(set(ports.keys()) == set(GAMES), f"ports 必须恰好覆盖 {GAMES}，得到 {sorted(ports)}")
    if clock_mode == "lockstep-v1":
        _require(len(set(ports.values())) == len(ports), "lockstep 两游戏端口必须不同")
    else:
        _require(len(set(ports.values())) == 1, "实时客户端必须共用一个明确记录的端口")
    for g, port in ports.items():
        _require(0 < int(port) < 65536, f"{g} 端口越界: {port}")
    _require(int(env["flappy_ticks_per_step"]) >= 1, "flappy_ticks_per_step 至少为 1")
    _require(int(env["timber_ticks_per_step"]) >= 1, "timber_ticks_per_step 至少为 1")

    # ---- 连续块 ----
    cont = cfg_raw["continuous"]
    _require(int(cont["latent_dim"]) == 2, "本版 latent_dim 固定为 2")
    _require(list(cont["hidden_sizes"]) == [64, 64], "本版 hidden_sizes 固定为 [64, 64]")
    _require(cont["decoder"] == "linear_softmax", f"decoder 必须是 linear_softmax，得到 {cont['decoder']!r}")
    _require(0.0 <= float(cont["decoder_behavior_uniform_mix"]) < 1.0,
             "decoder_behavior_uniform_mix 必须在 [0,1)")
    _require(float(cont["log_std_min"]) < float(cont["log_std_max"]), "log_std_min 必须小于 log_std_max")
    _require(float(cont["kl_weight"]) > 0, "kl_weight 必须为正")
    _require(float(cont["repr_lr"]) > 0, "repr_lr 必须为正")
    _require(cont["importance_correction"] == "exact_decoder_ratio",
             "importance_correction 必须是 exact_decoder_ratio")
    _require(cont["importance_clip"] is None, "主配置不裁剪重要性权重（§8.2）")

    # ---- 离散块 ----
    disc = cfg_raw["discrete"]
    _require(int(disc["embedding_dim"]) == 8, "本版离散 embedding_dim 固定为 8")
    _require(disc["repr_mode"] == "learned_codebook", "D 的 repr_mode 必须是 learned_codebook")

    # ---- replay 块 ----
    rep = cfg_raw["replay"]
    batch = int(rep["batch_size"])
    _require(batch == 128, f"batch_size 必须是 128，得到 {batch}")
    _require(int(rep["per_task_capacity"]) > 0, "per_task_capacity 必须为正")
    _require(int(rep["max_stored_transitions"]) >= 2 * batch,
             "max_stored_transitions 至少要能装下两个 batch")
    _require(rep["task_balanced"] is True, "task_balanced 必须为 true")
    _require(0.0 < float(rep["active_task_history_fraction"]) < 1.0,
             "active_task_history_fraction 必须在 (0,1)")

    # ---- train 块 ----
    tr = cfg_raw["train"]
    _require(float(tr["gamma"]) == 0.99, f"gamma 固定为 0.99，得到 {tr['gamma']}")
    _require(tr["backup_entropy"] is False, "主 critic 不含熵 backup（§3）")
    _require(float(tr["tau"]) == 0.005, "tau 固定为 0.005")
    _require(float(tr["grad_clip"]) > 0, "grad_clip 必须为正")
    _require(float(tr["entropy_alpha"]) == 0.02, "entropy_alpha 固定为 0.02")
    _require(float(tr["lambda_cal"]) == 0.1, "lambda_cal 固定为 0.1")
    _require(int(tr["target_samples"]) == 2, "target_samples 固定为 2")
    _require(int(tr["calibration_samples"]) == 10, "calibration_samples 固定为 10")
    for key, prod in PRODUCTION_CONSTANTS.items():
        val = int(tr[key])
        if test_only:
            lo, hi = TEST_ONLY_RANGES[key]
            _require(lo <= val <= hi,
                     f"test_only 配置的 {key} 必须在 [{lo},{hi}]，得到 {val}")
        else:
            _require(val == prod, f"{key} 固定为 {prod}（test_only 模式才可缩小）")
    _require(tr["initial_data"] == "self_collected_uniform", "默认自采均匀数据")
    _require(tr["offline_pretrain_once"] is True, "初始离线预训练只做一次")
    _require(float(tr["actor_lr"]) > 0 and float(tr["critic_lr"]) > 0, "学习率必须为正")

    # ---- eval 块 ----
    ev = cfg_raw["eval"]
    _require(ev["primary_policy_mode"] == "greedy", "主部署评测模式固定为 greedy")
    _require(ev["diagnostic_policy_mode"] == "stochastic_target", "诊断模式固定为 stochastic_target")
    _require(int(ev["test_episodes"]) == 50, "最终测试固定每游戏 50 回合")
    _require(int(ev["validation_episodes"]) == 10, "验证固定每游戏 10 回合")
    _require(int(ev["validation_seed_start"]) == 100000, "验证 seed 起点固定为 100000")
    _require(int(ev["test_seed_start"]) == 200000, "测试 seed 起点固定为 200000")
    _require(0.0 < float(ev["required_pass_rate"]) <= 1.0, "required_pass_rate 必须在 (0,1]")

    # ---- schedule / mode 组合与阶段表 ----
    stage_schedule, resolved_game = _expand_stage_schedule(cfg_raw, game=game)

    cfg = Config(
        raw=cfg_raw,
        variant=variant,
        schedule=schedule,
        mode=mode,
        seed=seed,
        game=resolved_game,
        source_path=source_path,
        stage_schedule=stage_schedule,
        test_only=bool(test_only),
    )
    _validate_budget(cfg)
    return cfg


def _expand_stage_schedule(raw: Mapping[str, Any], *, game: Optional[str]) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """把 schedule 展开成 [(game, steps), ...]。single 由 mode 决定。"""
    mode = raw["mode"]
    schedule = raw["schedule"]
    schedules = raw["train"]["stage_schedules"]

    if mode == "single":
        _require(schedule == "single", "mode=single 必须搭配 schedule=single")
        if game is None:
            game = raw.get("game")
        _require(game in GAMES, f"mode=single 必须指定 game ∈ {GAMES}，得到 {game!r}")
        return [{"game": game, "steps": 50000}], game

    _require(schedule in ("strict", "rehearsal"),
             f"mode={mode} 必须搭配 schedule=strict 或 rehearsal")
    if game is not None:
        raise ConfigError(f"mode={mode} 不接受 --game（只有 single 需要）")
    _require(schedule in schedules, f"stage_schedules 缺少 {schedule}")
    stages = schedules[schedule]
    _require(isinstance(stages, list) and stages, f"stage_schedules.{schedule} 必须是非空列表")
    out: List[Dict[str, Any]] = []
    for i, st in enumerate(stages):
        _require(isinstance(st, dict) and "game" in st and "steps" in st,
                 f"stage_schedules.{schedule}[{i}] 需要 game 与 steps")
        g = st["game"]
        steps = int(st["steps"])
        _require(g in GAMES, f"stage_schedules.{schedule}[{i}].game 非法: {g!r}")
        _require(steps > 0, f"stage_schedules.{schedule}[{i}].steps 必须为正")
        out.append({"game": g, "steps": steps})

    expected_order = (
        ["Flappybird", "Timberman"] if schedule == "strict"
        else ["Flappybird", "Timberman", "Flappybird", "Timberman", "Flappybird", "Timberman"]
    )
    _require([s["game"] for s in out] == expected_order,
             f"{schedule} 阶段顺序必须是 {expected_order}，得到 {[s['game'] for s in out]}")
    return out, None


def _validate_budget(cfg: Config) -> None:
    """§0.1/§11：预算必须与方案一致，避免悄悄改预算后仍沿用同一实验身份。"""
    if cfg.test_only:
        # 小预算测试：只要求预算为正、每游戏相等、且不超过正式预算
        _require(cfg.total_train_steps > 0, "测试预算必须为正")
        for g in GAMES:
            _require(cfg.steps_for_game(g) <= 50000,
                     f"test_only 的 {g} 预算不能超过 50000")
        if cfg.mode != "single":
            _require(len({cfg.steps_for_game(g) for g in GAMES}) == 1,
                     "test_only 双任务比较要求每游戏预算相等（公平比较）")
        return
    if cfg.mode == "single":
        _require(cfg.total_train_steps == 50000,
                 f"单游戏预算必须是 50000，得到 {cfg.total_train_steps}")
        _require(cfg.steps_for_game(cfg.game) == 50000, "单游戏预算必须全部给该游戏")
        return

    _require(cfg.total_train_steps == 100000,
             f"双任务总预算必须是 100000，得到 {cfg.total_train_steps}")
    for g in GAMES:
        _require(cfg.steps_for_game(g) == 50000,
                 f"{g} 的预算必须是 50000，得到 {cfg.steps_for_game(g)}")
    if cfg.schedule == "rehearsal":
        _require(len(cfg.stage_schedule) == 6, "rehearsal 必须是 6 个阶段")
        _require([s["steps"] for s in cfg.stage_schedule] == [30000, 30000, 10000, 10000, 10000, 10000],
                 "rehearsal 各阶段步数必须是 [30000,30000,10000,10000,10000,10000]")
    else:
        _require(len(cfg.stage_schedule) == 2, "strict 必须是 2 个阶段")
        _require([s["steps"] for s in cfg.stage_schedule] == [50000, 50000],
                 "strict 各阶段步数必须是 [50000,50000]")


def run_dir(cfg: Config, root: str | Path = "runs_cal/v2") -> Path:
    """§10 输出路径：runs_cal/v2/<variant>/<schedule>/<mode>/<seed>/。"""
    root = Path(root)
    if cfg.mode == "single":
        return root / cfg.variant / "single" / str(cfg.game) / str(cfg.seed)
    return root / cfg.variant / cfg.schedule / cfg.mode / str(cfg.seed)


if __name__ == "__main__":  # 便于人工核对
    import sys

    c = load_config(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "mt_cal.yaml")
    print(json.dumps({
        "identity_hash": c.identity_hash(),
        "total_train_steps": c.total_train_steps,
        "stage_schedule": c.stage_schedule,
        "run_dir": str(run_dir(c)),
    }, indent=2, ensure_ascii=False))
