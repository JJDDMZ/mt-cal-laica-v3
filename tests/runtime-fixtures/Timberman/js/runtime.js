/**
 * Timberman lockstep 运行时（技术方案 §4.2/§4.3/§5.2）。
 *
 * 与原 main.js 的差别：
 *  - `run()`（碰撞→换边→砍树）只在收到 `op=step` 时执行一次；
 *  - 扣血从 renderLifebar 移入逻辑 tick，每个 agent step 扣 6×16=96；
 *  - 观测直接读树方位/npc 方位/blood（不再依赖 render 初始化坐标）；
 *  - 渲染是独立 RAF 循环，只读状态。
 */

'use strict';

var envcore = require('./envcore.js');
// 与 Flappybird 项目里的是同一份实现，各自留一份副本，
// 避免跨项目相对路径在小游戏打包器里解析失败（appLaunch timeout 的常见成因）。
var wemical = require('./wemical.js');

var TICKS_PER_STEP = 6;              // §4.3：一次 agent step 最多 6 个逻辑 tick
var BLOOD_MAX = envcore.BLOOD_MAX;

function TimberRuntime(opts) {
    opts = opts || {};
    this.ctx = opts.ctx || canvas.getContext('2d');
    this.canvas = opts.canvas || canvas;
    this.serverUrl = opts.serverUrl || 'ws://localhost:17912';
    this.screenHeight = opts.screenHeight || (typeof window !== 'undefined' ? window.innerHeight : 667);
    this.screenWidth = opts.screenWidth || (typeof window !== 'undefined' ? window.innerWidth : 375);

    this.engine = new envcore.TimberEngine(1);
    // 小游戏的 ctx.drawImage 只接受 Image/Canvas 对象，**不能传路径字符串**。
    // 原版 main.js 之所以能跑，就是因为它先 new Image() 再画。这里补齐这一步。
    this.imgs = {
        bg: this._loadImage('images/bg.jpg'),
        npc: this._loadImage('images/npc.png'),
        npcMove: this._loadImage('images/npcMove.png'),
        npcDie: this._loadImage('images/npcDie.png'),
        treeLeft: this._loadImage('images/treeLeft.png'),
        treeCenter: this._loadImage('images/tree.png'),
        treeRight: this._loadImage('images/treeRight.png')
    };
    this.sprite = this.imgs.npc;

    var self = this;
    this.client = new wemical.WemiCalClient({
        gameId: 'Timberman',
        url: this.serverUrl,
        envHash: opts.envHash || 'timberman-lockstep-v1',
        onRequest: function (msg) { return self._onRequest(msg); }
    });
    this.client.connect();
    // render:false 时不起渲染循环（Node 测试用），逻辑不受影响
    if (opts.render !== false) { this._startRenderLoop(); }
}

TimberRuntime.prototype._loadImage = function (src) {
    var img = new Image();
    img.src = src;
    return img;
};

// ---------------------------------------------------------------- 请求处理
TimberRuntime.prototype._onRequest = function (msg) {
    if (msg.op === 'reset') {
        var seed = (msg.seed === undefined || msg.seed === null) ? 1 : msg.seed;
        this.engine.reset(seed);
        return this._snapshot(false, 'none');
    }
    if (msg.op === 'step') {
        var action = msg.action;
        if (action !== 0 && action !== 1) {
            throw new Error('Timberman 非法动作: ' + action);
        }
        var r = this.engine.stepLockstep(action, TICKS_PER_STEP);
        var reason = 'none';
        if (r.dead) {
            reason = this.engine.deathReason === 'blood' ? 'blood' : 'collision';
        }
        return this._snapshot(r.dead, reason);
    }
    throw new Error('未知 op: ' + msg.op);
};

/** 13 项特征（§5.2）：11 棵树方位 + npc 方位 + blood/6000。 */
TimberRuntime.prototype._snapshot = function (terminated, reason) {
    var e = this.engine;
    var o = e.observation();
    return {
        obs: { features: o.features },
        score: e.score,
        terminated: !!terminated,
        terminationReason: reason,
        simTicks: e.ticks
    };
};

// ---------------------------------------------------------------- 渲染（只读）
// 渲染节流到 25fps：真机上 step 间隔极小，不节流会狂刷（画面像加速播放）。
// 逻辑仍只由 op=step 驱动，渲染频率与物理无关。
var RENDER_INTERVAL_MS = 40;

TimberRuntime.prototype._startRenderLoop = function () {
    var self = this;
    var lastDraw = 0;
    function loop(ts) {
        var now = (typeof ts === 'number') ? ts : Date.now();
        if (now - lastDraw >= RENDER_INTERVAL_MS) {
            lastDraw = now;
            try {
                self._render();
            } catch (err) {
                console.error('[timber] render error: ' + (err && err.stack ? err.stack : err));
            }
        }
        if (typeof window !== 'undefined' && window.requestAnimationFrame) {
            window.requestAnimationFrame(loop, self.canvas);
        } else {
            setTimeout(function () { loop(); }, RENDER_INTERVAL_MS);
        }
    }
    if (typeof window !== 'undefined' && window.requestAnimationFrame) {
        window.requestAnimationFrame(loop, this.canvas);
    } else {
        setTimeout(function () { loop(); }, RENDER_INTERVAL_MS);
    }
};

TimberRuntime.prototype._treeSprite = function (raw) {
    if (raw === 'center') { return this.imgs.treeCenter; }
    return raw ? this.imgs.treeRight : this.imgs.treeLeft;
};

TimberRuntime.prototype._render = function () {
    var e = this.engine;
    var ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width || this.screenWidth, this.canvas.height || this.screenHeight);
    ctx.drawImage(this.imgs.bg, 0, 0, this.screenWidth, this.screenHeight);

    for (var i = 0; i < e.trees.length; i++) {
        var raw = e.trees[i].raw;
        var y = this.screenHeight - 60 - (i + 1) * 60;
        var x = (raw && raw !== 'center') ? this.screenWidth / 2 - 54 : this.screenWidth / 2 - 156;
        ctx.drawImage(this._treeSprite(raw), x, y, 207, 60);
    }

    // npc（只读 posi 与 blood）
    var npcX = e.npcSide ? this.screenWidth / 2 + 40 : this.screenWidth / 2 - 25;
    ctx.drawImage(this.sprite, npcX, this.screenHeight - 160, 100, 100);

    // 血条：只画，不扣血
    ctx.fillStyle = '#222';
    ctx.fillRect(this.screenWidth / 2 - 95, 30, 190, 30);
    ctx.fillStyle = 'red';
    var ratio = Math.max(0, Math.min(1, e.blood / BLOOD_MAX));
    ctx.fillRect(this.screenWidth / 2 - 90, 35, 180 * ratio, 20);

    ctx.font = '30px Microsoft YaHei';
    ctx.fillStyle = '#ffffff';
    ctx.fillText('score: ' + e.score, this.screenWidth / 2 - 50, 100);
    if (e.gameOver) {
        ctx.fillText('Game Over', this.screenWidth / 2 - 50, 140);
    }
};

module.exports = { TimberRuntime: TimberRuntime, TICKS_PER_STEP: TICKS_PER_STEP };
