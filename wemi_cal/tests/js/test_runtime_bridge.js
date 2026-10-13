/**
 * T2 加强验证：**真实 Python 协议驱动** ↔ **真实 JS 游戏运行时**（不需要开发者工具）。
 *
 * Python 侧 (wemi_cal/tests/bridge_driver.py) 扮演"训练客户端"，用真正的 Session
 * 发 hello/reset/step 并校验；JS 侧用 wechat_games_cal 的运行时执行真实物理并回包。
 * 分工与真机一致，因此在打开开发者工具之前就能确认整条链路。
 *
 * 用法：node wemi_cal/tests/js/test_runtime_bridge.js
 */

'use strict';

var path = require('path');
var cp = require('child_process');
var fs = require('fs');
var ROOT = path.resolve(__dirname, '..', '..', '..');

var mock = require('./wx_mock.js');

/** 把 ES module 语法的 Timberman runtime 变成可被 Node require 的 CommonJS。 */
function loadTimberRuntime() {
    var src = fs.readFileSync(path.join(ROOT, 'wechat_games_cal/Timberman/js/runtime.js'), 'utf8');
    src = src.replace("var envcore = require('./envcore.js');",
        'var envcore = require(' + JSON.stringify(
            path.join(ROOT, 'wechat_games_cal/Timberman/js/envcore.js')) + ');');
    src = src.replace("require('./wemical.js')",
        'require(' + JSON.stringify(
            path.join(ROOT, 'wechat_games_cal/Timberman/js/wemical.js')) + ')');
    var Module = require('module');
    var m = new Module('timber-runtime', null);
    m.filename = path.join(ROOT, 'wechat_games_cal/Timberman/js/runtime.cjs.js');
    m.paths = Module._nodeModulePaths(path.dirname(m.filename));
    m._compile(src, m.filename);
    return m.exports;
}

function startDriver(game, envHash) {
    var py = process.env.WEMI_CAL_PYTHON || '/opt/miniconda3/envs/WemiEnv/bin/python';
    var proc = cp.spawn(py, ['-m', 'wemi_cal.tests.bridge_driver',
        '--game', game, '--env-hash', envHash],
        { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
    var out = '', err = '';
    var portPromise = new Promise(function (resolve, reject) {
        var timer = setTimeout(function () {
            reject(new Error('驱动启动超时\nstdout:\n' + out + '\nstderr:\n' + err));
        }, 60000);
        proc.stdout.on('data', function (d) {
            out += d.toString();
            process.stdout.write('[py] ' + d.toString());
            var m = /PORT=(\d+)/.exec(out);
            if (m) { clearTimeout(timer); resolve(parseInt(m[1], 10)); }
        });
        proc.stderr.on('data', function (d) {
            err += d.toString();
            process.stderr.write('[py:err] ' + d.toString());
        });
    });
    var exitPromise = new Promise(function (resolve) {
        proc.on('exit', function (code) { resolve(code === null ? -1 : code); });
    });
    return { proc: proc, port: portPromise, exit: exitPromise };
}

function startFlappyRuntime(port, state) {
    // 每次都重新 require，保证拿到干净的模块状态
    delete require.cache[require.resolve(path.join(ROOT, 'wechat_games_cal/Flappybird/lib/wxbird.js'))];
    delete require.cache[require.resolve(path.join(ROOT, 'wechat_games_cal/Flappybird/lib/wemical.js'))];
    delete require.cache[require.resolve(path.join(ROOT, 'wechat_games_cal/Flappybird/lib/envcore.js'))];
    var wxBird = require(path.join(ROOT, 'wechat_games_cal/Flappybird/lib/wxbird.js'));
    return new wxBird.WxBirdRuntime({
        ctx: state.ctx, id: 'birdId', width: 400, height: 600,
        serverUrl: 'ws://127.0.0.1:' + port, envHash: 'flappybird-lockstep-v1',
        render: false
    });
}

function startTimberRuntime(port, state) {
    var exportsObj = loadTimberRuntime();
    return new exportsObj.TimberRuntime({
        ctx: state.ctx, canvas: global.canvas,
        serverUrl: 'ws://127.0.0.1:' + port, envHash: 'timberman-lockstep-v1',
        width: 375, height: 667, render: false
    });
}

async function runGame(game) {
    var envHash = game === 'Flappybird' ? 'flappybird-lockstep-v1' : 'timberman-lockstep-v1';
    var state = mock.installWxMock({ canvasId: game === 'Flappybird' ? 'birdId' : 'canvas' });
    var drv = startDriver(game, envHash);
    var port;
    try {
        port = await drv.port;
    } catch (e) {
        console.log('FAIL  ' + game + ' 驱动未就绪: ' + e.message);
        drv.proc.kill();
        return false;
    }
    console.log('\n--- ' + game + ' 驱动已就绪，端口 ' + port + '，启动 JS 运行时 ---');

    var rt = game === 'Flappybird'
        ? startFlappyRuntime(port, state)
        : startTimberRuntime(port, state);

    var code = await drv.exit;
    try {
        if (rt && rt.client) {
            if (typeof rt.client.stopTimers === 'function') { rt.client.stopTimers(); }
            rt.client.closed = true;
        }
    } catch (e) {}
    console.log('--- ' + game + ' 驱动退出码 = ' + code + ' ---');
    return code === 0;
}

(async function main() {
    var okAll = true;
    try {
        okAll = (await runGame('Flappybird')) && okAll;
        okAll = (await runGame('Timberman')) && okAll;
    } catch (err) {
        console.log('FAIL  异常: ' + (err && err.stack ? err.stack : err));
        okAll = false;
    }
    if (!okAll) {
        console.log('\nJS RUNTIME <-> PYTHON BRIDGE: FAILED');
        process.exit(1);
    }
    console.log('\nALL JS RUNTIME <-> PYTHON BRIDGE TESTS PASSED');
})();
