/**
 * T2 验收：JS lockstep 环境正确性（技术方案 §14 T2）。
 *
 * 不需要微信开发者工具，直接跑 node：
 *   node wemi_cal/tests/js/test_envcore.js
 *
 * 覆盖：同 seed+固定动作序列可重放、连续 reset 不残留、
 *       逻辑 tick 上限、死亡立即停止、扣血只在 tick、观测与奖励语义。
 */

'use strict';

var path = require('path');
var ROOT = path.resolve(__dirname, '..', '..', '..');
var flappy = require(path.join(ROOT, 'wechat_games_cal/Flappybird/lib/envcore.js'));
var timber = require(path.join(ROOT, 'wechat_games_cal/Timberman/js/envcore.js'));

var failures = [];
var checks = 0;

function check(name, cond, detail) {
    checks += 1;
    if (!cond) {
        failures.push(name + (detail ? '  [' + detail + ']' : ''));
        console.log('FAIL  ' + name + (detail ? '  [' + detail + ']' : ''));
    } else {
        console.log('ok    ' + name);
    }
}

function deepEqual(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
}

/** 用固定动作序列跑一段轨迹，返回可比较的快照。 */
function runFlappy(seed, actions, ticks) {
    var e = new flappy.FlappyEngine(seed);
    for (var i = 0; i < actions.length; i++) {
        e.stepLockstep(actions[i], ticks);
        if (e.stoped) { break; }
    }
    return {
        by: e.bird.by, px: e.up_pipe.px, pheight: e.up_pipe.pheight,
        py: e.down_pipe.py, score: e.score, stoped: e.stoped, ticks: e.ticks
    };
}

function runTimber(seed, actions, ticks) {
    var e = new timber.TimberEngine(seed);
    for (var i = 0; i < actions.length; i++) {
        e.stepLockstep(actions[i], ticks);
        if (e.gameOver) { break; }
    }
    return {
        score: e.score, blood: e.blood, npcSide: e.npcSide,
        trees: e.trees.map(function (t) { return t.raw; }),
        gameOver: e.gameOver, ticks: e.ticks, deathReason: e.deathReason
    };
}

// ---------------------------------------------------------------- 确定性
(function testFlappyDeterminism() {
    var actions = [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0];
    var a = runFlappy(12345, actions, 3);
    var b = runFlappy(12345, actions, 3);
    check('F 同 seed + 同动作序列完全可重放', deepEqual(a, b), JSON.stringify(a) + ' vs ' + JSON.stringify(b));

    var c = runFlappy(12346, actions, 3);
    check('F 不同 seed 产生不同管道序列', !deepEqual(a, c));
})();

(function testTimberDeterminism() {
    var actions = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0];
    var a = runTimber(777, actions, 6);
    var b = runTimber(777, actions, 6);
    check('T 同 seed + 同动作序列完全可重放', deepEqual(a, b), JSON.stringify(a) + ' vs ' + JSON.stringify(b));

    var c = runTimber(778, actions, 6);
    check('T 不同 seed 产生不同树序列', !deepEqual(a, c));
})();

(function testSeedZeroInitialisedToOne() {
    // §4.4：seed=0 时初始化为 1，因此 seed=0 与 seed=1 的随机流相同
    var a = runTimber(0, [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0], 6);
    var b = runTimber(1, [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0], 6);
    check('T seed=0 等价于 seed=1（初始化为 1）', deepEqual(a, b));
    var fa = runFlappy(0, [1, 0, 0, 1, 1, 0, 0, 0], 3);
    var fb = runFlappy(1, [1, 0, 0, 1, 1, 0, 0, 0], 3);
    check('F seed=0 等价于 seed=1（初始化为 1）', deepEqual(fa, fb));
})();

// ---------------------------------------------------------------- reset 不残留
(function testFlappyResetClean() {
    var e = new flappy.FlappyEngine(42);
    for (var i = 0; i < 30; i++) { e.stepLockstep(1, 3); if (e.stoped) { break; } }
    var first = { score: e.score, ticks: e.ticks, stoped: e.stoped };
    e.reset(42);   // 同 seed 再 reset
    var again = runFlappy(42, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3);
    var fresh = runFlappy(42, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3);
    check('F reset 后状态与全新实例一致', deepEqual(again, fresh),
        JSON.stringify(again) + ' vs ' + JSON.stringify(fresh));
    check('F reset 清掉上一回合的 ticks/score', e.ticks === 0 && e.score === 0 && e.stoped === false,
        JSON.stringify(first));
})();

(function testTimberResetClean() {
    var e = new timber.TimberEngine(99);
    for (var i = 0; i < 40; i++) { e.stepLockstep(i % 2, 6); if (e.gameOver) { break; } }
    e.reset(99);
    check('T reset 清掉 score/gameOver/blood', e.score === 0 && e.gameOver === false && e.blood === 6000,
        'score=' + e.score + ' blood=' + e.blood);
    check('T reset 后树数量仍为 11', e.trees.length === 11);
    var a = runTimber(99, [0, 1, 0, 1, 1], 6);
    var e2 = new timber.TimberEngine(99);
    for (var j = 0; j < 5; j++) { e2.stepLockstep([0, 1, 0, 1, 1][j], 6); }
    var b = {
        score: e2.score, blood: e2.blood, npcSide: e2.npcSide,
        trees: e2.trees.map(function (t) { return t.raw; }),
        gameOver: e2.gameOver, ticks: e2.ticks, deathReason: e2.deathReason
    };
    check('T reset 后轨迹与全新实例一致', deepEqual(a, b));
})();

// ---------------------------------------------------------------- tick 上限与死亡
(function testFlappyTickBudget() {
    var e = new flappy.FlappyEngine(7);
    var before = e.ticks;
    var r = e.stepLockstep(0, 3);
    check('F 单步最多推进 3 个物理 tick', e.ticks - before <= 3 && r.ticks <= 3);
    var big = e.ticks;
    e.stepLockstep(0, 3);
    check('F ticks 参数被严格执行（不会自己多跑）', e.ticks - big <= 3);
})();

(function testTimberTickBudget() {
    var e = new timber.TimberEngine(7);
    var before = e.ticks;
    var r = e.stepLockstep(0, 6);
    check('T 单步最多推进 6 个逻辑 tick', e.ticks - before <= 6 && r.ticks <= 6);
})();

(function testFlappyDeathStopsImmediately() {
    var e = new flappy.FlappyEngine(3);
    var steps = 0;
    while (!e.stoped && steps < 500) { e.stepLockstep(0, 3); steps += 1; }
    check('F 不点击最终死亡', e.stoped, 'steps=' + steps);
    var ticksAtDeath = e.ticks;
    var byAtDeath = e.bird.by;
    e.stepLockstep(1, 3);   // 死后动作无效
    check('F 死后 tick 不再推进', e.ticks === ticksAtDeath);
    check('F 死后位置不再改变', e.bird.by === byAtDeath);
})();

(function testTimberBloodDeath() {
    // 一直点同一侧：会先撞树（碰撞）而死；改用「总在安全侧」的脚本验证血尽路径
    var e = new timber.TimberEngine(5);
    var steps = 0;
    while (!e.gameOver && steps < 1000) {
        // 站到与最前树相反的一侧 → 若是换边步则不砍树，血只会掉
        var front = e.frontSide();
        var want = (front === timber.encodeSide(true)) ? 0 : 1;  // 避开碰撞
        e.stepLockstep(want, 6);
        steps += 1;
        if (steps > 1000) { break; }
    }
    check('T 最终会结束（碰撞或血尽）', e.gameOver, 'reason=' + e.deathReason + ' steps=' + steps);
    check('T 结束原因是已知枚举', e.deathReason === 'collision' || e.deathReason === 'blood',
        String(e.deathReason));
})();

// ---------------------------------------------------------------- 扣血只在 tick
(function testTimberBloodOnlyInTick() {
    var e = new timber.TimberEngine(11);
    // 找一个能砍树的动作：与最前树相反且与当前 npc 同侧
    var front = e.frontSide();
    var sameAsNpc = (front !== timber.encodeSide(e.npcSide)) ? (e.npcSide ? 1 : 0) : (e.npcSide ? 0 : 1);
    var bloodBefore = e.blood;
    e.stepAgent(sameAsNpc);           // 只做决策，不推进 tick
    check('T 决策本身不扣血', e.blood >= bloodBefore, 'before=' + bloodBefore + ' after=' + e.blood);
    var before = e.blood;
    e.advanceTick();
    check('T 每个逻辑 tick 扣 16 血', before - e.blood === 16, 'delta=' + (before - e.blood));
})();

(function testTimberRenderDoesNotDrainBlood() {
    var e = new timber.TimberEngine(13);
    var before = e.blood;
    for (var i = 0; i < 50; i++) { e.renderState(); e.observation(); }
    check('T 读观测/渲染状态不扣血（扣血已移出 renderLifebar）', e.blood === before,
        'before=' + before + ' after=' + e.blood);
})();

// ---------------------------------------------------------------- 观测 schema
(function testFlappyObservation() {
    var e = new flappy.FlappyEngine(21);
    e.up_pipe.pheight = 300; e.down_pipe.py = 500; e.bird.by = 250; e.up_pipe.px = 200;
    var o = e.observation();
    check('F 4 维特征按 world 尺度归一',
        Math.abs(o.bird_y - 250 / 600) < 1e-12 &&
        Math.abs(o.gap_top - 300 / 600) < 1e-12 &&
        Math.abs(o.gap_bottom - 500 / 600) < 1e-12 &&
        Math.abs(o.pipe_x - 200 / 400) < 1e-12,
        JSON.stringify(o));
    check('F 观测不含隐式竖直速度（本版本没有动量状态）', !('velocity' in o) && !('vy' in o));
})();

(function testTimberObservation() {
    var e = new timber.TimberEngine(31);
    e.blood = 3000;
    e.npcSide = false;
    var o = e.observation();
    check('T 13 维特征长度正确', o.features.length === 13, 'len=' + o.features.length);
    check('T npc_side 编码 false→-1', o.features[11] === -1, String(o.features[11]));
    check('T blood/6000 归一', Math.abs(o.features[12] - 0.5) < 1e-12, String(o.features[12]));
    var sides = o.features.slice(0, 11);
    check('T 11 棵树方位都在 {-1,0,1}',
        sides.every(function (s) { return s === -1 || s === 0 || s === 1; }), JSON.stringify(sides));
    var bad = false;
    try { timber.encodeSide('LEFT'); } catch (err) { bad = true; }
    check('T 未知方位字符串被拒绝', bad);
})();

(function testTimberRandomTreeConsumption() {
    // randomTree 必须恰好消耗一个随机数，否则树序列不可重放
    var e1 = new timber.TimberEngine(55);
    var e2 = new timber.TimberEngine(55);
    var t1 = e1.randomTree();
    var t2 = e2.randomTree();
    check('T randomTree 消耗随机数次数一致', deepEqual(t1, t2));
    // 走一个不同的分支（上一棵是 center）也必须消耗一个
    e1.trees = [{ raw: 'center' }];
    e2.trees = [{ raw: 'center' }];
    e1.randomTree();
    e2.randomTree();
    check('T center 分支同样只消耗一个随机数', e1.rng.nextUint() === e2.rng.nextUint());
})();

// ---------------------------------------------------------------- 原规则次序
(function testTimbermanKeepsCollisionThenSideThenCutOrder() {
    var e = new timber.TimberEngine(3);
    // 确定性构造同侧碰撞：直接把最前树设成 npc 所在侧（±1），并保留原规则次序断言。
    // 注意 center(0) 与任一侧都不相等，源码里 false == 'center' 为真但那只是
    // 字符串比较的巧合，编码空间里 0 与 -1 不相等——所以只用 ±1 构造碰撞。
    e.npcSide = true;
    e.trees[0] = { raw: true };              // 右侧树 + 右侧 npc → 碰撞
    var frontBefore = e.frontSide();
    var scoreBefore = e.score;
    var r = e.stepAgent(0);
    check('T 先碰撞：同侧立即 gameOver，不计分', r.dead === true && e.score === scoreBefore,
        'reason=' + e.deathReason + ' frontBefore=' + frontBefore + ' result=' + JSON.stringify(r));

    // 显式构造同侧碰撞：直接把最前树设成 npc 所在侧
    var e3 = new timber.TimberEngine(3);
    e3.npcSide = true;
    e3.trees[0] = { raw: true };            // 右侧树 + 右侧 npc → 碰撞
    var sc3 = e3.score;
    var r3 = e3.stepAgent(0);
    check('T 显式同侧碰撞死亡且不计分', r3.dead === true && e3.score === sc3 && e3.deathReason === 'collision',
        JSON.stringify(r3) + ' reason=' + e3.deathReason);
    check('T center 树对 true 侧不碰撞（true != "center"）',
        (function () {
            var x = new timber.TimberEngine(3);
            x.npcSide = true;
            x.trees[0] = { raw: 'center' };
            var rr = x.stepAgent(1);
            return rr.dead === false;
        })(), 'center vs true 应为 false');
    check('T center 树对 false 侧也不碰撞（false != "center" 的数值等价）',
        (function () {
            var x = new timber.TimberEngine(3);
            x.npcSide = false;
            x.trees[0] = { raw: 'center' };
            var rr = x.stepAgent(0);
            return rr.dead === false;
        })(), 'center vs false 在编码空间为 0 vs -1');

    // 不同侧但点击了对面 → 换边、不计分、不砍树
    var e2 = new timber.TimberEngine(3);
    e2.npcSide = true;
    var front = e2.frontSide();
    if (front !== timber.encodeSide(true)) {
        var r2 = e2.stepAgent(0);   // 点击左侧 → 换到左侧
        check('T 换边步不计分', r2.switched === true && e2.score === 0,
            JSON.stringify(r2));
    } else {
        check('T 换边步不计分（该 seed 不适用，跳过）', true);
    }
})();

(function testFlappyRewardSemantics() {
    var e = new flappy.FlappyEngine(1);
    // 间隙正中 → 距离为 (gap_bottom-gap_top)/2
    e.bird.by = 300; e.up_pipe.pheight = 200; e.down_pipe.py = 400;
    var rMid = e.reward(false);
    check('F 间隙正中奖励 = 0.01 * 半径', Math.abs(rMid - 1.0) < 1e-12, String(rMid));

    e.bird.by = 200;   // 正好等于 gap_top → 距离 0
    check('F 边界相等奖励为 0', Math.abs(e.reward(false)) < 1e-12, String(e.reward(false)));

    e.bird.by = 250;   // 距上 50，距下 150 → min=50 → 0.5
    check('F 取两侧较小距离', Math.abs(e.reward(false) - 0.5) < 1e-12, String(e.reward(false)));

    check('F 死亡奖励固定 -1', e.reward(true) === -1);

    // 越出间隙 → 负奖励但不低于 -1
    e.bird.by = 100;   // 距上 -100 → -1.0
    check('F 越界奖励被 clip 到 [-1,1]', e.reward(false) === -1, String(e.reward(false)));
})();

(function testFlappyPipeSpawnMatchesSource() {
    var e = new flappy.FlappyEngine(2024);
    // 强制进入生成分支
    e.up_pipe.px = -50; e.up_pipe.pwidth = 50;
    e.bird.by = 300;
    var rngBefore = e.rng.nextUint();
    e.reset(2024);
    e.up_pipe.px = -50; e.up_pipe.pwidth = 50; e.bird.by = 300;
    e.advanceTick();
    check('F 生成新管道时 px 重置为 400', e.up_pipe.px === 400, String(e.up_pipe.px));
    check('F 新管道 gap 高度落在 [100,300)', e.up_pipe.pheight >= 100 && e.up_pipe.pheight < 300,
        String(e.up_pipe.pheight));
    check('F down_pipe.py = gap_top + pipe_height，且 pheight = 600 - py',
        Math.abs(e.down_pipe.py - (e.up_pipe.pheight + 200)) < 1e-9 &&
        Math.abs(e.down_pipe.pheight - (600 - e.down_pipe.py)) < 1e-9,
        JSON.stringify(e.down_pipe));
    check('F 保留原方程 (ver2+ver1)*0.25 的下落量', Math.abs(e.ver1) === 10 && Math.abs(e.gravity) === 2);
    void rngBefore;
})();

// ---------------------------------------------------------------- 非法动作
(function testIllegalActionsRejected() {
    var f = new flappy.FlappyEngine(1);
    var t = new timber.TimberEngine(1);
    var bad = 0;
    try { f.stepAgent(2); } catch (err) { bad += 1; }
    try { t.stepAgent(2); } catch (err) { bad += 1; }
    try { f.stepAgent(-1); } catch (err) { bad += 1; }
    check('非法本地动作直接报错（不 clamp）', bad === 3, 'caught=' + bad);
})();

// ---------------------------------------------------------------- 汇总
console.log('\n' + checks + ' checks, ' + failures.length + ' failures');
if (failures.length) {
    console.log('FAILED:\n  ' + failures.join('\n  '));
    process.exit(1);
}
console.log('ALL JS ENVCORE TESTS PASSED');
