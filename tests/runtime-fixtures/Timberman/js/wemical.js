/**
 * wemi-cal-v1 WebSocket 客户端（技术方案 §4.2）。
 *
 * 与原 WemiEnv 客户端的区别：
 *  - 协议是消息级的：服务端发 {op, ...}，客户端**每条都必须回一条响应**；
 *  - 不再用 "超时 sleep 100ms 再回包" 制造时序，改为同步执行后立即回；
 *  - 响应带 protocol/game_id/session_id/episode_id/request_id 五元组配对，
 *    以及 obs/score/terminated/termination_reason/sim_ticks/info；
 *  - 不合法消息一律记 `bad_message`，绝不用默认值继续跑。
 *
 * 用法：
 *   const client = new WemiCalClient({ gameId, url, envHash, ... });
 *   client.onRequest = (msg) => ({ obs, score, terminated, terminationReason, simTicks });
 */

'use strict';

var PROTOCOL = 'wemi-cal-v1';

function WemiCalClient(options) {
    options = options || {};
    this.protocol = PROTOCOL;
    this.gameId = options.gameId;
    this.url = options.url;
    this.envHash = options.envHash || null;
    this.obsSchema = options.obsSchema || 'compact29-v1';
    this.rewardSchema = options.rewardSchema || 'bounded-v1';
    this.clockMode = options.clockMode || 'lockstep-v1';
    this.onRequest = options.onRequest || null;
    this.onOpen = options.onOpen || null;
    this.onClose = options.onClose || null;
    this.onError = options.onError || null;
    this.log = options.log || function () {};

    this.sessionId = 'unset';
    this.episodeId = -1;
    this.simTicks = 0;
    this.connected = false;
    this.closed = false;
    this.reconnectDelayMs = (options.reconnectDelayMs === undefined) ? 1500 : options.reconnectDelayMs;
    this.maxReconnectAttempts = (options.maxReconnectAttempts === undefined) ? 1000 : options.maxReconnectAttempts;
    this.reconnectAttempts = 0;
    this.stats = { requests: 0, responses: 0, bad: 0, lastOp: null, reconnects: 0 };
    this._bind();
    this._startHeartbeat();
}

WemiCalClient.prototype._bind = function () {
    var self = this;
    wx.onSocketOpen(function () {
        self.connected = true;
        self.reconnectAttempts = 0;
        self.log('[wemi-cal] connected ' + self.url);
        if (self.onOpen) { self.onOpen(); }
    });
    wx.onSocketClose(function () {
        self.connected = false;
        self.log('[wemi-cal] closed');
        if (self.onClose) { self.onClose(); }
        // 掉线必须自己重连：wx.connectSocket 只在启动时调过一次，
        // 不重连就会永远卡在"没有客户端"，训练端只能一直等。
        if (!self.closed) { self._scheduleReconnect(); }
    });
    wx.onSocketError(function (err) {
        self.log('[wemi-cal] error ' + JSON.stringify(err));
        if (self.onError) { self.onError(err); }
        if (!self.closed && !self.connected) { self._scheduleReconnect(); }
    });
    wx.onSocketMessage(function (message) {
        self._onMessage(message);
    });
};

WemiCalClient.prototype.connect = function () {
    if (this.connected) { return; }
    var self = this;
    try {
        // 带 fail 回调：真机上连接失败**不一定**触发 onSocketError，
        // 没有 fail 回调就会永远卡在"没连上也不重试"的状态。
        wx.connectSocket({
            url: this.url,
            fail: function (res) {
                console.error('[wemi-cal] connectSocket fail: ' + JSON.stringify(res));
                self._scheduleReconnect();
            }
        });
    } catch (e) {
        console.error('[wemi-cal] connectSocket threw: ' + e);
        this._scheduleReconnect();
    }
};

/** 心跳：只要没连上就周期性重试，作为回调缺失时的兜底。 */
WemiCalClient.prototype._startHeartbeat = function () {
    var self = this;
    if (this._heartbeatTimer) { return; }
    this._heartbeatTimer = setInterval(function () {
        if (self.closed) { return; }
        if (!self.connected) {
            if (!self._reconnectTimer) { self._scheduleReconnect(); }
        }
    }, 3000);
    // Node 测试环境：不能让这个 interval 阻止进程退出
    if (typeof this._heartbeatTimer.unref === 'function') { this._heartbeatTimer.unref(); }
};

/** 停止心跳与重连（关闭会话/测试收尾时调用）。 */
WemiCalClient.prototype.stopTimers = function () {
    this.closed = true;
    if (this._heartbeatTimer) {
        if (typeof clearInterval === 'function') { clearInterval(this._heartbeatTimer); }
        this._heartbeatTimer = null;
    }
    if (this._reconnectTimer) {
        if (typeof clearTimeout === 'function') { clearTimeout(this._reconnectTimer); }
        this._reconnectTimer = null;
    }
};

/** 延迟重连（指数退避到上限，避免断线时疯狂重试）。 */
WemiCalClient.prototype._scheduleReconnect = function () {
    if (this._reconnectTimer) { return; }
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.error('[wemi-cal] 重连次数已达上限 ' + this.maxReconnectAttempts);
        return;
    }
    var self = this;
    var delay = Math.min(this.reconnectDelayMs * Math.pow(1.5, this.reconnectAttempts), 10000);
    this.reconnectAttempts += 1;
    this.log('[wemi-cal] ' + Math.round(delay) + 'ms 后第 ' + this.reconnectAttempts + ' 次重连');
    this._reconnectTimer = setTimeout(function () {
        self._reconnectTimer = null;
        if (self.closed) { return; }
        self.stats.reconnects += 1;
        self.connect();
    }, delay);
};

WemiCalClient.prototype._send = function (obj) {
    if (!this.connected) {
        this.stats.bad += 1;
        console.error('[wemi-cal] 未连接，丢弃一条响应（训练端会因超时而重连）');
        return;
    }
    wx.sendSocketMessage({
        data: JSON.stringify(obj),
        fail: function (res) {
            // 发送失败不能静默：服务端会超时，这里留下证据
            try { console.error('[wemi-cal] send failed', res); } catch (e) {}
        }
    });
};

WemiCalClient.prototype._onMessage = function (message) {
    var msg;
    try {
        msg = JSON.parse(message.data);
    } catch (e) {
        this.stats.bad += 1;
        console.error('[wemi-cal] bad json from server', message && message.data);
        return;
    }
    if (!msg || typeof msg !== 'object' || typeof msg.op !== 'string') {
        this.stats.bad += 1;
        console.error('[wemi-cal] bad message shape', message && message.data);
        return;
    }
    this.stats.requests += 1;
    this.stats.lastOp = msg.op;

    if (msg.protocol !== PROTOCOL) {
        this.stats.bad += 1;
        console.error('[wemi-cal] protocol mismatch: ' + msg.protocol);
        return;
    }
    if (msg.game_id !== this.gameId) {
        this.stats.bad += 1;
        console.error('[wemi-cal] game_id mismatch: ' + msg.game_id + ' expected ' + this.gameId);
        return;
    }

    // 服务端每次请求都会携带 session/episode；客户端照抄回去以便配对
    if (msg.session_id !== undefined) { this.sessionId = msg.session_id; }
    if (msg.episode_id !== undefined) { this.episodeId = msg.episode_id; }
    if (msg.op === 'hello' || msg.op === 'reset') { this.simTicks = 0; }

    var reply = {
        protocol: PROTOCOL,
        game_id: this.gameId,
        session_id: this.sessionId,
        episode_id: this.episodeId,
        request_id: msg.request_id,
        obs: {},
        score: 0,
        terminated: false,
        termination_reason: 'none',
        sim_ticks: this.simTicks,
        info: this._info()
    };

    if (msg.op === 'hello') {
        // hello 不改变游戏状态，只汇报身份
        this._send(reply);
        this.stats.responses += 1;
        this.log('[wemi-cal] hello ok, env_hash=' + this.envHash);
        return;
    }

    if (msg.op === 'close') {
        this.stopTimers();
        this._send(reply);
        this.stats.responses += 1;
        try { wx.closeSocket(); } catch (e) {}
        return;
    }

    if (typeof this.onRequest !== 'function') {
        this.stats.bad += 1;
        console.error('[wemi-cal] no onRequest handler');
        return;
    }

    var out;
    try {
        out = this.onRequest(msg);
    } catch (e) {
        // 逻辑异常必须暴露：既打印堆栈，也回一条带 error 的响应。
        // 不能直接 throw —— 那会让异常冒泡进 socket 回调，Node 下直接终止进程、
        // 真机上则表现为 Python 侧静默超时，两种情况都会掩盖根因。
        this.stats.bad += 1;
        var trace = (e && e.stack) ? e.stack : String(e);
        console.error('[wemi-cal] onRequest threw: ' + trace);
        reply.error = (e && e.message) ? e.message : 'onRequest_failed';
        reply.error_kind = 'illegal_request';
        reply.error_stack = trace;
        this._send(reply);
        this.stats.responses += 1;
        return;
    }
    if (!out || typeof out !== 'object') {
        this.stats.bad += 1;
        console.error('[wemi-cal] onRequest returned nothing for op=' + msg.op);
        return;
    }

    reply.obs = out.obs || {};
    reply.score = (out.score === undefined) ? 0 : out.score;
    reply.terminated = !!out.terminated;
    reply.termination_reason = out.terminationReason || 'none';
    reply.sim_ticks = (out.simTicks === undefined) ? this.simTicks : out.simTicks;
    this.simTicks = reply.sim_ticks;

    this._send(reply);
    this.stats.responses += 1;
};

WemiCalClient.prototype._info = function () {
    return {
        clock_mode: this.clockMode,
        env_hash: this.envHash,
        obs_schema: this.obsSchema,
        reward_schema: this.rewardSchema
    };
};

module.exports = { WemiCalClient: WemiCalClient, PROTOCOL: PROTOCOL };
