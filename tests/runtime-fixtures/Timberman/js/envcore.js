/**
 * Timberman lockstep 环境核心（技术方案 §4.3/§4.4，任务 T2）。
 *
 * 与 main.js 的差别，全部是为了可重放与 lockstep：
 *  1. 每个决策（换边/可能砍树）是唯一的 agent step；不再由 RAF 循环驱动 run()。
 *  2. 扣血从 renderLifebar 移到这里：每逻辑 tick -16（源码每帧 -16）。
 *  3. 树的随机生成用局部 xorshift32 PRNG，且保持「每读一棵消耗恰好一个随机数」的次序。
 *  4. 保留原 run() 次序：碰撞检测 → 判断换边（换边即 return）→ 砍树计分。
 *  5. 观测直接读树方位/npc 方位/blood，不再依赖 render 初始化坐标。
 */

'use strict';

var MAX_TREES = 11;
var BLOOD_MAX = 6000;
var BLOOD_DRAIN_PER_TICK = 16;
var BLOOD_GAIN_PER_CUT = 160;
var TICK_MS = 1000 / 60;      // §4.3：Timberman 每 tick 代表 1/60 秒
var AGENT_STEP_MS = 100;      // 一次 agent step 代表 100ms 虚拟时间

// 方位编码（§5.2）：false → -1，'center' → 0，true → +1
var SIDE_LEFT = -1;
var SIDE_CENTER = 0;
var SIDE_RIGHT = 1;

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

function encodeSide(raw) {
    if (raw === 'center') { return SIDE_CENTER; }
    if (raw === true) { return SIDE_RIGHT; }
    if (raw === false) { return SIDE_LEFT; }
    throw new Error('未知树方位: ' + JSON.stringify(raw));
}

function TimberEngine(seed) {
    this.reset(seed);
}

TimberEngine.prototype.reset = function (seed) {
    this.rng = makeRng(seed);
    this.seed = (seed >>> 0);
    this.score = 0;
    this.gameOver = false;
    this.npcSide = true;         // npc.posi 初值 true
    this.blood = BLOOD_MAX;
    this.trees = [];             // 每项 { raw: true/false/'center' }
    this.touchSide = null;       // 本次决策的点击侧（true=右屏，false=左屏）
    this.ticks = 0;
    this.agentSteps = 0;
    this.deathReason = null;
    for (var i = 1; i < 12; i++) {       // init() 里 for(i=1;i<12;i++)
        this.trees.push(this.randomTree());
    }
};

/**
 * 源码 randomTree()：随机数消耗次序必须原样保留——每次调用恰好消耗一个 nextFloat()，
 * 无论走哪个分支。返回原始布尔/字符串（不在这里编码成数值）。
 */
TimberEngine.prototype.randomTree = function () {
    var random = this.rng.nextFloat();
    var last = true;
    if (this.trees.length > 0) {
        last = this.trees[this.trees.length - 1].raw;
    }
    if (last === 'center') {
        if (random <= 0.3334) { return { raw: true }; }        // treeRight
        if (random <= 0.6667) { return { raw: false }; }       // treeLeft
        return { raw: 'center' };                              // treeCen
    }
    return { raw: 'center' };
};

TimberEngine.prototype.frontSide = function () {
    return encodeSide(this.trees[0].raw);
};

/**
 * 一次 agent 决策：等价于 simulate_touchCuttree(touchposi) → run()。
 * 源码 run() 的次序：碰撞检测 → 若已结束 return → 判断是否换边（换了就 return）
 *                → 否则计分、回血、shiftTree + pushTree。
 */
TimberEngine.prototype.stepAgent = function (localAction) {
    if (localAction !== 0 && localAction !== 1) {
        throw new Error('非法 Timberman 本地动作: ' + localAction);
    }
    if (this.gameOver) {
        this.agentSteps += 1;
        return { dead: true, cut: false, switched: false };
    }
    this.agentSteps += 1;
    // touchposi == 1 → 右半屏；== 0 → 左半屏（§5.4）
    this.touchSide = (localAction === 1);
    var tap = this.touchSide;

    // 1) 碰撞检测（源码 collisionDetection: this.npc.posi == databus.trees[0].posiDr）
    //    必须在规范化空间比较：源码里 npc.posi 是 bool，posiDr 可能是 bool 或 'center'，
    //    在 JS 里 false == 'center' 为真 → 站在 false 侧且最前树是 center 时也算碰撞。
    //    统一编码后 false≡-1、true≡+1，就与该行为完全一致。
    var npcCode = this.npcSide ? SIDE_RIGHT : SIDE_LEFT;
    if (npcCode === this.frontSide()) {
        this.gameOver = true;
        this.deathReason = 'collision';
        return { dead: true, cut: false, switched: false };
    }

    // 2) 换边判定：与当前所在侧相同 → 保持不动，本步不砍树（源码 !(posi == tap) 取反后 return）
    if (!(this.npcSide === tap)) {
        this.npcSide = !this.npcSide;
        return { dead: false, cut: false, switched: true };
    }

    // 3) 砍树：计分 + 回血（源码 run()）
    this.score += 1;
    this.blood = (this.blood + BLOOD_GAIN_PER_CUT >= BLOOD_MAX)
        ? BLOOD_MAX
        : this.blood + BLOOD_GAIN_PER_CUT;

    // 木头池：弹出最前，推入新树（净长度不变）
    this.trees.shift();
    this.trees.push(this.randomTree());
    return { dead: false, cut: true, switched: false };
};

/**
 * 逻辑 tick：只做时间流逝（扣血），绝不改方位/分数/树。
 * 源码里这段在 renderLifebar() 中，每渲染帧执行一次。
 */
TimberEngine.prototype.advanceTick = function () {
    if (this.gameOver) { return false; }
    this.blood -= BLOOD_DRAIN_PER_TICK;
    this.ticks += 1;
    if (this.blood < 0.017) {          // 源码 loop() 的判定
        this.gameOver = true;
        this.blood = this.blood < 0 ? this.blood : this.blood;
        this.deathReason = 'blood';
        return false;
    }
    return true;
};

/**
 * lockstep：一次决策 + 最多 ticks 个逻辑 tick；血尽或碰撞立即停。
 */
TimberEngine.prototype.stepLockstep = function (localAction, ticks) {
    if (!(ticks >= 1)) { throw new Error('ticks 必须 >= 1'); }
    var res = this.stepAgent(localAction);
    var done = 0;
    if (!this.gameOver) {
        for (var i = 0; i < ticks; i++) {
            if (!this.advanceTick()) { break; }
            done += 1;
        }
    }
    return {
        dead: this.gameOver,
        cut: res.cut,
        switched: res.switched,
        ticks: done
    };
};

/**
 * 主 schema 的 13 维特征（§5.2）：
 *   t[0:11] = 11 棵树从近到远的 branch_side（false→-1, 'center'→0, true→+1）
 *   t[11]   = npc_side（false→-1, true→+1）
 *   t[12]   = blood / 6000
 * 以原始 bool/字符串解码，拒绝未知字符串。
 */
TimberEngine.prototype.observation = function () {
    var feat = [];
    for (var i = 0; i < MAX_TREES; i++) {
        if (i < this.trees.length) {
            feat.push(encodeSide(this.trees[i].raw));
        } else {
            feat.push(0);   // 树不足 11 棵时的填充值；mask 仍为 1（Timberman 全 1）
        }
    }
    feat.push(this.npcSide ? SIDE_RIGHT : SIDE_LEFT);
    feat.push(this.blood / BLOOD_MAX);
    return {
        features: feat,
        score: this.score,
        blood: this.blood,
        stopped: this.gameOver
    };
};

/** 只读渲染用状态；不修改任何逻辑字段。 */
TimberEngine.prototype.renderState = function () {
    return {
        trees: this.trees,
        npcSide: this.npcSide,
        blood: this.blood,
        score: this.score,
        gameOver: this.gameOver
    };
};

module.exports = {
    TimberEngine: TimberEngine,
    makeRng: makeRng,
    encodeSide: encodeSide,
    MAX_TREES: MAX_TREES,
    BLOOD_MAX: BLOOD_MAX,
    BLOOD_DRAIN_PER_TICK: BLOOD_DRAIN_PER_TICK,
    BLOOD_GAIN_PER_CUT: BLOOD_GAIN_PER_CUT,
    TICK_MS: TICK_MS,
    AGENT_STEP_MS: AGENT_STEP_MS
};
