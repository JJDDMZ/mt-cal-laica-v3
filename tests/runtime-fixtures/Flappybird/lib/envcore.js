/**
 * Flappybird lockstep 环境核心（技术方案 §4.3/§4.4，任务 T2）。
 *
 * 与原 wxbird.js 的差别，全部是为了可重放与 lockstep：
 *  1. 不再用 setInterval(drawall1, 40) 驱动物理：物理只在 stepAgent/advanceTick 里推进。
 *  2. 管道高度用局部 xorshift32 PRNG，不用 Math.random（不改全局 Math.random）。
 *  3. 渲染与逻辑彻底分离：draw() 只读状态，绝不改 by/px/score/stoped。
 *  4. 死亡当 tick 立即停止。
 * 物理公式、点击行为、计分与管道生成次序与源码逐行一致（含 repeat 死代码也保留，见 fsm）。
 */

'use strict';

var WORLD_WIDTH = 400;
var WORLD_HEIGHT = 600;
var GROUND_Y = 550;           // ground 在源码里由 Background(ctx,0,550,...) 创建
var PHYSICS_TICK_MS = 40;     // 源码 setInterval(..., 40)

/** §4.4：局部 xorshift32，seed=0 初始化为 1。 */
function makeRng(seed) {
    var s = (seed >>> 0);
    if (s === 0) { s = 1; }
    return {
        nextUint: function () {
            s ^= s << 13; s >>>= 0;
            s ^= s >>> 17;
            s ^= s << 5;  s >>>= 0;
            return s >>> 0;
        },
        nextFloat: function () { return this.nextUint() / 4294967296; }
    };
}

function FlappyEngine(seed) {
    this.reset(seed);
}

FlappyEngine.prototype.reset = function (seed) {
    this.rng = makeRng(seed);
    this.seed = (seed >>> 0);
    this.stoped = false;
    this.ver1 = 10;
    this.gravity = 2;
    this.pipe_height = 200;
    this.velocity = 10;
    this.score = 0;
    this.isScore = false;
    this.pubnish = 0;
    this.deathTick = -1;
    // 与 startGame1 的复位值一致
    this.bird = { bx: 80, by: 300, bwidth: 40, bheight: 40 };
    this.up_pipe = { px: 0, py: 0, pwidth: 50, pheight: 200 };
    this.down_pipe = { px: 0, py: 400, pwidth: 50, pheight: 200 };
    this.ticks = 0;
    this.fsm = 0;   // 本版本没有 ver2 声明 → 置位即死（见 tick）
    this.actionCount = 0;
};

/** 源码 keyup()：立即上移 60，且夹到 0。 */
FlappyEngine.prototype.keyup = function () {
    this.bird.by -= 60;
    if (this.bird.by < 0) { this.bird.by = 0; }
};

/** 一次 agent 决策（对应一次点击/不点击）。 */
FlappyEngine.prototype.stepAgent = function (action) {
    if (action !== 0 && action !== 1) {
        throw new Error('非法 Flappybird 本地动作: ' + action);
    }
    if (this.stoped) {
        // 死亡后不再改变逻辑状态；agent 步只增加计数
        this.actionCount += 1;
        return { dead: true, applied: false };
    }
    this.actionCount += 1;
    if (action === 1) { this.keyup(); }
    return { dead: false, applied: true };
};

/** 单个物理 tick，等价原 drawall1 里 calculator1 的推进。 */
FlappyEngine.prototype.advanceTick = function () {
    if (this.stoped) { return false; }

    // ---- 碰撞检测（源码 calculator1 开头）----
    var b = this.bird, up = this.up_pipe, down = this.down_pipe;
    if (b.by + b.bheight > GROUND_Y ||
        ((b.bx + b.bwidth > up.px) && (b.by > up.py) &&
         (b.bx + b.bwidth < up.px + up.pwidth) && (b.by < up.py + up.pheight)) ||
        ((b.bx + b.bwidth > up.px) && (b.by > up.py) &&
         (b.bx + b.bwidth < up.px + up.pwidth) && (b.by < up.py + up.pheight)) ||
        ((b.bx > down.px) && (b.by > down.py) &&
         (b.bx < down.px + down.pwidth) && (b.by < down.py + down.pheight)) ||
        ((b.bx > down.px) && (b.by + b.bheight > down.py) &&
         (b.bx < down.px + down.pwidth) && (b.by + b.bheight < down.py + down.pheight))) {
        this.stoped = true;
        this.pubnish = -50;
        this.deathTick = this.ticks;
        return false;
    }

    // ---- 下落（源码：ver2 = ver1 + gravity; by += (ver2 + ver1) * 0.25）----
    this.ver2 = this.ver1 + this.gravity;   // 仅作为属性存在，不进入后续计算
    this.bird.by += (this.ver2 + this.ver1) * 0.25;

    // ---- 管道水平移动 / 生成 ----
    if (this.up_pipe.px + this.up_pipe.pwidth > 0) {
        this.up_pipe.px -= this.velocity;
        this.down_pipe.px -= this.velocity;
    } else {
        this.up_pipe.px = 400;
        this.down_pipe.px = 400;
        this.up_pipe.pheight = 100 + this.rng.nextFloat() * 200;
        this.down_pipe.py = this.up_pipe.pheight + this.pipe_height;
        this.down_pipe.pheight = 600 - this.down_pipe.py;
        this.isScore = true;
    }

    // ---- 计分 ----
    if (this.isScore && this.bird.bx > this.up_pipe.px + this.up_pipe.pwidth) {
        this.score += 1;
        this.isScore = false;
    }

    this.ticks += 1;
    return true;
};

/**
 * lockstep：先应用一次动作，再推进 ticks 个物理 tick；死亡当 tick 立即停。
 * 这是 §4.3 里 “一次 agent step = 点击 + 最多 3 个原物理 tick” 的执行体。
 */
FlappyEngine.prototype.stepLockstep = function (action, ticks) {
    if (!(ticks >= 1)) { throw new Error('ticks 必须 >= 1'); }
    var applied = this.stepAgent(action);
    var done = 0;
    for (var i = 0; i < ticks; i++) {
        if (!this.advanceTick()) { break; }
        done += 1;
    }
    return { dead: this.stoped, ticks: done, applied: applied.applied };
};

/** 主 schema 的 4 维特征（§5.2）：by, gap_top, gap_bottom, pipe_x，按 world 尺度归一。 */
FlappyEngine.prototype.observation = function () {
    return {
        bird_y: this.bird.by / WORLD_HEIGHT,
        gap_top: this.up_pipe.pheight / WORLD_HEIGHT,
        gap_bottom: this.down_pipe.py / WORLD_HEIGHT,
        pipe_x: this.up_pipe.px / WORLD_WIDTH,
        score: this.score,
        stopped: this.stoped
    };
};

/** §5.5 r_F：死亡 -1；否则 clip(0.01*min(by-gap_top, gap_bottom-by), -1, 1)，边界相等取 0。 */
FlappyEngine.prototype.reward = function (died) {
    if (died) { return -1; }
    var d1 = this.bird.by - this.up_pipe.pheight;
    var d2 = this.down_pipe.py - this.bird.by;
    var d = Math.min(d1, d2);
    var r = 0.01 * d;
    if (r > 1) { r = 1; }
    if (r < -1) { r = -1; }
    return r;
};

/** 只读渲染用状态；不修改任何逻辑字段。 */
FlappyEngine.prototype.renderState = function () {
    return {
        bird: this.bird,
        up_pipe: this.up_pipe,
        down_pipe: this.down_pipe,
        ground: { bgy: GROUND_Y },
        score: this.score,
        stoped: this.stoped,
        pubnish: this.pubnish
    };
};

module.exports = {
    FlappyEngine: FlappyEngine,
    makeRng: makeRng,
    WORLD_WIDTH: WORLD_WIDTH,
    WORLD_HEIGHT: WORLD_HEIGHT,
    GROUND_Y: GROUND_Y,
    PHYSICS_TICK_MS: PHYSICS_TICK_MS
};
