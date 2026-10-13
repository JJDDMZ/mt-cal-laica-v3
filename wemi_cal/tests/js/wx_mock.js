/**
 * 受控 wx 环境模拟（无开发者工具）。
 *
 * 把 wx 的 socket / canvas API 换成 Node 实现，让 Flappybird 与 Timberman 的运行时
 * 能连到**真实的 Python 桥接**上跑。这样可以在打开模拟器之前就验证 wemi-cal-v1
 * 协议与 lockstep 行为，是 T2 真机验证之前的必要一步。
 */

'use strict';

var net = require('net');
var crypto = require('crypto');

// ---------------------------------------------------------------- WebSocket 客户端（RFC6455 子集）
function WsClient(url) {
    var m = /^ws:\/\/([^:/]+):(\d+)(\/.*)?$/.exec(url);
    if (!m) { throw new Error('只支持 ws://host:port 形式，得到 ' + url); }
    this.host = m[1];
    this.port = parseInt(m[2], 10);
    this.sock = null;
    this.buf = Buffer.alloc(0);
    this.handlers = { open: null, message: null, close: null, error: null };
}

WsClient.prototype.connect = function () {
    var self = this;
    this.sock = net.createConnection({ host: this.host, port: this.port }, function () {
        var key = crypto.randomBytes(16).toString('base64');
        self.sock.write(
            'GET / HTTP/1.1\r\n' +
            'Host: ' + self.host + ':' + self.port + '\r\n' +
            'Upgrade: websocket\r\nConnection: Upgrade\r\n' +
            'Sec-WebSocket-Key: ' + key + '\r\n' +
            'Sec-WebSocket-Version: 13\r\n\r\n'
        );
    });
    this.sock.on('data', function (chunk) { self._onData(chunk); });
    this.sock.on('close', function () { if (self.handlers.close) { self.handlers.close(); } });
    this.sock.on('error', function (e) { if (self.handlers.error) { self.handlers.error(e); } });
};

WsClient.prototype._onData = function (chunk) {
    this.buf = Buffer.concat([this.buf, chunk]);
    if (this._handshaked !== true) {
        var idx = this.buf.indexOf('\r\n\r\n');
        if (idx < 0) { return; }
        var head = this.buf.slice(0, idx).toString();
        this.buf = this.buf.slice(idx + 4);
        if (head.indexOf('101') < 0) {
            if (this.handlers.error) { this.handlers.error(new Error('握手失败: ' + head)); }
            return;
        }
        this._handshaked = true;
        if (this.handlers.open) { this.handlers.open(); }
    }
    while (true) {
        var frame = this._readFrame();
        if (!frame) { return; }
        if (frame.opcode === 0x8) {
            if (this.handlers.close) { this.handlers.close(); }
            return;
        }
        if (frame.opcode === 0x1 && this.handlers.message) {
            this.handlers.message({ data: frame.payload.toString('utf8') });
        }
    }
};

WsClient.prototype._readFrame = function () {
    if (this.buf.length < 2) { return null; }
    var b0 = this.buf[0], b1 = this.buf[1];
    var opcode = b0 & 0x0f;
    var masked = (b1 & 0x80) !== 0;
    var len = b1 & 0x7f;
    var off = 2;
    if (len === 126) {
        if (this.buf.length < off + 2) { return null; }
        len = this.buf.readUInt16BE(off); off += 2;
    } else if (len === 127) {
        if (this.buf.length < off + 8) { return null; }
        len = Number(this.buf.readBigUInt64BE(off)); off += 8;
    }
    var mask = null;
    if (masked) {
        if (this.buf.length < off + 4) { return null; }
        mask = this.buf.slice(off, off + 4); off += 4;
    }
    if (this.buf.length < off + len) { return null; }
    var payload = this.buf.slice(off, off + len);
    this.buf = this.buf.slice(off + len);
    if (mask) {
        payload = Buffer.from(payload);
        for (var i = 0; i < payload.length; i++) { payload[i] ^= mask[i % 4]; }
    }
    return { opcode: opcode, payload: payload };
};

WsClient.prototype.send = function (text) {
    var payload = Buffer.from(text, 'utf8');
    var header;
    var n = payload.length;
    if (n < 126) {
        header = Buffer.from([0x81, 0x80 | n]);
    } else if (n < 65536) {
        header = Buffer.alloc(4); header[0] = 0x81; header[1] = 0x80 | 126;
        header.writeUInt16BE(n, 2);
    } else {
        header = Buffer.alloc(10); header[0] = 0x81; header[1] = 0x80 | 127;
        header.writeBigUInt64BE(BigInt(n), 2);
    }
    var mask = crypto.randomBytes(4);
    var masked = Buffer.alloc(n);
    for (var i = 0; i < n; i++) { masked[i] = payload[i] ^ mask[i % 4]; }
    this.sock.write(Buffer.concat([header, mask, masked]));
};

WsClient.prototype.close = function () {
    try { this.sock.end(); } catch (e) {}
};

// ---------------------------------------------------------------- canvas 桩
function makeCtx() {
    var ops = [];
    var ctx = {
        _ops: ops,
        clearRect: function () { ops.push(['clearRect']); },
        drawImage: function () { ops.push(['drawImage'].concat([].slice.call(arguments))); },
        fillRect: function () { ops.push(['fillRect']); },
        fillText: function (t) { ops.push(['fillText', t]); },
        setFillStyle: function () {}, setFontSize: function () {},
        set fillStyle(v) {}, set font(v) {},
        getActions: function () { return ops.slice(); }
    };
    return ctx;
}

// ---------------------------------------------------------------- wx 环境
function installWxMock(options) {
    options = options || {};
    var state = {
        socket: null,
        handlers: {},
        draws: 0,
        ctx: makeCtx(),
        sent: [],
        canvasId: options.canvasId || 'birdId'
    };

    global.wx = {
        connectSocket: function (o) {
            var ws = new WsClient(o.url);
            state.socket = ws;
            ws.handlers.open = function () { if (state.handlers.open) { state.handlers.open(); } };
            ws.handlers.message = function (m) { if (state.handlers.message) { state.handlers.message(m); } };
            ws.handlers.close = function () { if (state.handlers.close) { state.handlers.close(); } };
            ws.handlers.error = function (e) { if (state.handlers.error) { state.handlers.error(e); } };
            ws.connect();
        },
        onSocketOpen: function (cb) { state.handlers.open = cb; },
        onSocketMessage: function (cb) { state.handlers.message = cb; },
        onSocketClose: function (cb) { state.handlers.close = cb; },
        onSocketError: function (cb) { state.handlers.error = cb; },
        sendSocketMessage: function (o) { state.sent.push(o.data); state.socket.send(o.data); },
        closeSocket: function () { if (state.socket) { state.socket.close(); } },
        createContext: function () { return state.ctx; },
        drawCanvas: function () { state.draws += 1; }
    };

    global.window = {
        innerWidth: options.width || 375,
        innerHeight: options.height || 667,
        requestAnimationFrame: function () { return 0; }   // 渲染循环在测试里不需要真跑
    };
    global.canvas = {
        width: options.width || 375,
        height: options.height || 667,
        getContext: function () { return state.ctx; },
        addEventListener: function () {}, removeEventListener: function () {}
    };
    global.Image = function () { this.src = ''; };
    global.console = console;

    return state;
}

module.exports = { installWxMock: installWxMock, WsClient: WsClient };
