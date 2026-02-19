// Based on https://github.com/antimatter15/splat (MIT, Kevin Kwok 2023)

import {
    getProjectionMatrix,
    multiply4,
    invert4,
    rotate4,
    translate4,
    vertexShaderSource,
    fragmentShaderSource
} from "./math.js";

let FX = 300;
let FY = 300;

function createWorker(self) {
    let buffer;
    let vertexCount = 0;
    let viewProj;
    // 6*4 + 4 + 4 = 8*4
    // XYZ - Position (Float32)
    // XYZ - Scale (Float32)
    // RGBA - colors (uint8)
    // IJKL - quaternion/rot (uint8)
    let lastProj = [];
    let lastVertexCount = 0;

    var _floatView = new Float32Array(1);
    var _int32View = new Int32Array(_floatView.buffer);
    function floatToHalf(float) {
        _floatView[0] = float;
        var f = _int32View[0];
        var sign = (f >> 31) & 0x0001;
        var exp = (f >> 23) & 0x00ff;
        var frac = f & 0x007fffff;
        var newExp;
        if (exp == 0) { newExp = 0; }
        else if (exp < 113) { newExp = 0; frac |= 0x00800000; frac = frac >> (113 - exp); if (frac & 0x01000000) { newExp = 1; frac = 0; } }
        else if (exp < 142) { newExp = exp - 112; }
        else { newExp = 31; frac = 0; }
        return (sign << 15) | (newExp << 10) | (frac >> 13);
    }

    function packHalf2x16(x, y) {
        return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
    }

    function generateTexture() {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        const u_buffer = new Uint8Array(buffer);

        var texwidth = 1024 * 2;
        var texheight = Math.ceil((2 * vertexCount) / texwidth);
        var texdata = new Uint32Array(texwidth * texheight * 4);
        var texdata_c = new Uint8Array(texdata.buffer);
        var texdata_f = new Float32Array(texdata.buffer);

        const stride_f32 = 8;
        const stride_u8 = 32;

        for (let i = 0; i < vertexCount; i++) {
            // means
            texdata_f[stride_f32 * i + 0] = f_buffer[stride_f32 * i + 0];
            texdata_f[stride_f32 * i + 1] = f_buffer[stride_f32 * i + 1];
            texdata_f[stride_f32 * i + 2] = f_buffer[stride_f32 * i + 2];

            // scale
            let scale = [
                f_buffer[stride_f32 * i + 3],
                f_buffer[stride_f32 * i + 4],
                f_buffer[stride_f32 * i + 5],
            ];

            // rgba
            texdata_c[4 * (8 * i + 7) + 0] = u_buffer[stride_u8 * i + 24];
            texdata_c[4 * (8 * i + 7) + 1] = u_buffer[stride_u8 * i + 25];
            texdata_c[4 * (8 * i + 7) + 2] = u_buffer[stride_u8 * i + 26];
            texdata_c[4 * (8 * i + 7) + 3] = u_buffer[stride_u8 * i + 27];

            // quat
            let rot = [
                (u_buffer[stride_u8 * i + 28] - 128) / 128,
                (u_buffer[stride_u8 * i + 29] - 128) / 128,
                (u_buffer[stride_u8 * i + 30] - 128) / 128,
                (u_buffer[stride_u8 * i + 31] - 128) / 128,
            ];

            // covariance: M = S * R
            const M = [
                1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
                2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
                2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),

                2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
                2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),

                2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
                2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
            ].map((k, i) => k * scale[Math.floor(i / 3)]);

            const sigma = [
                M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
                M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
                M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
                M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
                M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
                M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
            ];

            texdata[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
            texdata[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
            texdata[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);
        }

        self.postMessage({ texdata, texwidth, texheight }, [texdata.buffer]);
    }

    function runSort(viewProj) {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        if (lastVertexCount == vertexCount) {
            let dot =
                lastProj[2] * viewProj[2] +
                lastProj[6] * viewProj[6] +
                lastProj[10] * viewProj[10];
            if (Math.abs(dot - 1) < 0.01) {
                return;
            }
        } else {
            generateTexture();
            lastVertexCount = vertexCount;
        }

        console.time("sort");
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++) {
            let depth =
                ((viewProj[2] * f_buffer[8 * i + 0] +
                    viewProj[6] * f_buffer[8 * i + 1] +
                    viewProj[10] * f_buffer[8 * i + 2]) *
                    4096) |
                0;
            sizeList[i] = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (depth < minDepth) minDepth = depth;
        }

        let depthInv = (256 * 256 - 1) / (maxDepth - minDepth);
        let counts0 = new Uint32Array(256 * 256);
        for (let i = 0; i < vertexCount; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }
        let starts0 = new Uint32Array(256 * 256);
        for (let i = 1; i < 256 * 256; i++)
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        let depthIndex = new Uint32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++)
            depthIndex[starts0[sizeList[i]]++] = i;

        console.timeEnd("sort");

        lastProj = viewProj;
        self.postMessage({ depthIndex, viewProj, vertexCount }, [
            depthIndex.buffer,
        ]);
    }

    const throttledSort = () => {
        if (!sortRunning) {
            sortRunning = true;
            let lastView = viewProj;
            runSort(lastView);
            setTimeout(() => {
                sortRunning = false;
                if (lastView !== viewProj) {
                    throttledSort();
                }
            }, 0);
        }
    };

    let sortRunning;
    self.onmessage = (e) => {
        if (e.data.buffer) {
            buffer = e.data.buffer;
            vertexCount = e.data.vertexCount;
            lastVertexCount = 0; // force texture regeneration
        } else if (e.data.vertexCount) {
            vertexCount = e.data.vertexCount;
        } else if (e.data.view) {
            viewProj = e.data.view;
            throttledSort();
        }
    };
}

async function main() {
    let viewMatrix = [
        0.47, 0.04, 0.88, 0, -0.11, 0.99, 0.02, 0, -0.88, -0.11, 0.47, 0, 0.07,
        0.03, 6.55, 1,
    ];

    try {
        viewMatrix = JSON.parse(decodeURIComponent(location.hash.slice(1)));
    } catch (err) {}

    const worker = new Worker(
        URL.createObjectURL(
            new Blob(["(", createWorker.toString(), ")(self)"], {
                type: "application/javascript",
            }),
        ),
    );

    const ws = new WebSocket("ws://localhost:9090");
    ws.binaryType = "arraybuffer";

    let gaussianBuffer = null;
    let originSet = false;

    ws.onmessage = (e) => {
        const view = new DataView(e.data);
        const type = view.getUint8(0);

        if (type === 0x03) {
            // intrinsics: [type:u8][fx:f32][fy:f32]
            const buf = new ArrayBuffer(8);
            new Uint8Array(buf).set(new Uint8Array(e.data, 1, 8));
            const fbuf = new Float32Array(buf);
            FX = fbuf[0];
            FY = fbuf[1];
            console.log(`Intrinsics received: fx=${FX}, fy=${FY}`);
            resize();
            return;
        }

        if (type === 0x02) {
            // camera pose: use first one as origin
            if (!originSet) {
                const buf = new ArrayBuffer(64);
                new Uint8Array(buf).set(new Uint8Array(e.data, 1, 64));
                viewMatrix = Array.from(new Float32Array(buf));
                originSet = true;
            }
            return;
        }

        if (type === 0x00) {
            // full replace: [type:u8][n:u32][v:u32][data]
            const n = view.getUint32(1, true);
            const version = view.getUint32(5, true);
            console.log(`Buffer received: version=${version}`);
            gaussianBuffer = new ArrayBuffer(n * 32);
            new Uint8Array(gaussianBuffer).set(new Uint8Array(e.data, 9));
            worker.postMessage({ buffer: gaussianBuffer, vertexCount: n });
        }
    };

    const canvas = document.getElementById("canvas");
    const fps = document.getElementById("fps");

    let projectionMatrix;

    const gl = canvas.getContext("webgl2", { antialias: false });

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(vertexShader));

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(fragmentShader));

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        console.error(gl.getProgramInfoLog(program));

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(
        gl.ONE_MINUS_DST_ALPHA, gl.ONE,
        gl.ONE_MINUS_DST_ALPHA, gl.ONE,
    );
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    const u_projection = gl.getUniformLocation(program, "projection");
    const u_viewport = gl.getUniformLocation(program, "viewport");
    const u_focal = gl.getUniformLocation(program, "focal");
    const u_view = gl.getUniformLocation(program, "view");

    const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
    const a_position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(a_position);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    var u_textureLocation = gl.getUniformLocation(program, "u_texture");
    gl.uniform1i(u_textureLocation, 0);

    const indexBuffer = gl.createBuffer();
    const a_index = gl.getAttribLocation(program, "index");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
    gl.vertexAttribDivisor(a_index, 1);

    const resize = () => {
        gl.uniform2fv(u_focal, new Float32Array([FX, FY]));
        projectionMatrix = getProjectionMatrix(FX, FY, innerWidth, innerHeight);
        gl.uniform2fv(u_viewport, new Float32Array([innerWidth, innerHeight]));
        gl.canvas.width = innerWidth;
        gl.canvas.height = innerHeight;
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.uniformMatrix4fv(u_projection, false, projectionMatrix);
    };

    window.addEventListener("resize", resize);
    resize();


    worker.onmessage = (e) => {
        if (e.data.texdata) {
            const { texdata, texwidth, texheight } = e.data;
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texImage2D(
                gl.TEXTURE_2D, 0, gl.RGBA32UI,
                texwidth, texheight, 0,
                gl.RGBA_INTEGER, gl.UNSIGNED_INT, texdata,
            );
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture);
        } else if (e.data.depthIndex) {
            const { depthIndex } = e.data;
            gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
            vertexCount = e.data.vertexCount;
        }
    };

    let activeKeys = new Set();
    let vertexCount = 0;
    let lastFrame = 0;
    let avgFps = 0;

    // --- Keyboard ---
    window.addEventListener("keydown", (e) => activeKeys.add(e.code));
    window.addEventListener("keyup", (e) => activeKeys.delete(e.code));
    window.addEventListener("blur", () => activeKeys.clear());

    // --- Scroll (dolly) ---
    window.addEventListener("wheel", (e) => {
        e.preventDefault();
        const scale = e.deltaMode == 1 ? 10 : e.deltaMode == 2 ? innerHeight : 1;
        let inv = invert4(viewMatrix);
        inv = translate4(inv, 0, 0, (-e.deltaY * scale * 5) / innerHeight);
        viewMatrix = invert4(inv);
    }, { passive: false });

    // --- Mouse (left=orbit, right=pan) ---
    let startX, startY, down;
    canvas.addEventListener("mousedown", (e) => {
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = e.button === 2 || e.ctrlKey || e.metaKey ? 2 : 1;
    });
    canvas.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = 2;
    });
    canvas.addEventListener("mousemove", (e) => {
        e.preventDefault();
        if (down == 1) {
            let inv = invert4(viewMatrix);
            let dx = (4 * (e.clientX - startX)) / innerWidth;
            let dy = (4 * (e.clientY - startY)) / innerHeight;
            let d = 4;
            inv = translate4(inv, 0, 0, d);
            inv = rotate4(inv, dx, 0, 1, 0);
            inv = rotate4(inv, -dy, 1, 0, 0);
            inv = translate4(inv, 0, 0, -d);
            viewMatrix = invert4(inv);
            startX = e.clientX;
            startY = e.clientY;
        } else if (down == 2) {
            let inv = invert4(viewMatrix);
            inv = translate4(
                inv,
                (-5 * (e.clientX - startX)) / innerWidth,
                (5 * (e.clientY - startY)) / innerHeight,
                0,
            );
            viewMatrix = invert4(inv);
            startX = e.clientX;
            startY = e.clientY;
        }
    });
    canvas.addEventListener("mouseup", (e) => {
        e.preventDefault();
        down = false;
    });

    // --- Touch (1-finger orbit, 2-finger pinch/pan) ---
    let altX = 0, altY = 0;
    canvas.addEventListener("touchstart", (e) => {
        e.preventDefault();
        if (e.touches.length === 1) {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
            down = 1;
        } else if (e.touches.length === 2) {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
            altX = e.touches[1].clientX;
            altY = e.touches[1].clientY;
            down = 1;
        }
    }, { passive: false });
    canvas.addEventListener("touchmove", (e) => {
        e.preventDefault();
        if (e.touches.length === 1 && down) {
            let inv = invert4(viewMatrix);
            let dx = (4 * (e.touches[0].clientX - startX)) / innerWidth;
            let dy = (4 * (e.touches[0].clientY - startY)) / innerHeight;
            let d = 4;
            inv = translate4(inv, 0, 0, d);
            inv = rotate4(inv, dx, 0, 1, 0);
            inv = rotate4(inv, -dy, 1, 0, 0);
            inv = translate4(inv, 0, 0, -d);
            viewMatrix = invert4(inv);
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
        } else if (e.touches.length === 2) {
            const dscale =
                Math.hypot(startX - altX, startY - altY) /
                Math.hypot(
                    e.touches[0].clientX - e.touches[1].clientX,
                    e.touches[0].clientY - e.touches[1].clientY,
                );
            const dx = (e.touches[0].clientX + e.touches[1].clientX - (startX + altX)) / 2;
            const dy = (e.touches[0].clientY + e.touches[1].clientY - (startY + altY)) / 2;
            let inv = invert4(viewMatrix);
            inv = translate4(inv, -dx / innerWidth, dy / innerHeight, 0);
            inv = translate4(inv, 0, 0, 3 * (1 - dscale));
            viewMatrix = invert4(inv);
            startX = e.touches[0].clientX;
            altX = e.touches[1].clientX;
            startY = e.touches[0].clientY;
            altY = e.touches[1].clientY;
        }
    }, { passive: false });
    canvas.addEventListener("touchend", (e) => {
        e.preventDefault();
        down = false;
    }, { passive: false });

    // --- Gamepad ---
    window.addEventListener("gamepadconnected", (e) => {
        console.log(`Gamepad connected: ${navigator.getGamepads()[e.gamepad.index].id}`);
    });

    // --- Render loop ---
    const frame = (now) => {
        const dt = Math.min((now - lastFrame) / 1000, 0.05) || 0.016;
        const moveSpeed = 3.0 * dt;
        const lookSpeed = 1.5 * dt;

        let inv = invert4(viewMatrix);

        // WASD move
        if (activeKeys.has("KeyW")) inv = translate4(inv, 0, 0, moveSpeed);
        if (activeKeys.has("KeyS")) inv = translate4(inv, 0, 0, -moveSpeed);
        if (activeKeys.has("KeyA")) inv = translate4(inv, -moveSpeed, 0, 0);
        if (activeKeys.has("KeyD")) inv = translate4(inv, moveSpeed, 0, 0);
        if (activeKeys.has("Space")) inv = translate4(inv, 0, -moveSpeed, 0);
        if (activeKeys.has("ShiftLeft") || activeKeys.has("ShiftRight"))
            inv = translate4(inv, 0, moveSpeed, 0);

        // Arrow keys look
        if (activeKeys.has("ArrowLeft"))  inv = rotate4(inv, -lookSpeed, 0, 1, 0);
        if (activeKeys.has("ArrowRight")) inv = rotate4(inv, lookSpeed, 0, 1, 0);
        if (activeKeys.has("ArrowUp"))    inv = rotate4(inv, lookSpeed, 1, 0, 0);
        if (activeKeys.has("ArrowDown"))  inv = rotate4(inv, -lookSpeed, 1, 0, 0);

        // Q/E roll
        if (activeKeys.has("KeyQ")) inv = rotate4(inv, lookSpeed, 0, 0, 1);
        if (activeKeys.has("KeyE")) inv = rotate4(inv, -lookSpeed, 0, 0, 1);

        // Gamepad
        const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
        for (let gamepad of gamepads) {
            if (!gamepad) continue;
            const dead = 0.15;
            const gpMove = 4.0 * dt;
            const gpLook = 2.0 * dt;

            const lx = Math.abs(gamepad.axes[0]) > dead ? gamepad.axes[0] : 0;
            const ly = Math.abs(gamepad.axes[1]) > dead ? gamepad.axes[1] : 0;
            const rx = Math.abs(gamepad.axes[2]) > dead ? gamepad.axes[2] : 0;
            const ry = Math.abs(gamepad.axes[3]) > dead ? gamepad.axes[3] : 0;

            inv = translate4(inv, gpMove * lx, 0, -gpMove * ly);
            inv = rotate4(inv, gpLook * rx, 0, 1, 0);
            inv = rotate4(inv, -gpLook * ry, 1, 0, 0);

            const upDown = gamepad.buttons[7].value - gamepad.buttons[6].value;
            if (Math.abs(upDown) > dead)
                inv = translate4(inv, 0, -gpMove * upDown, 0);

            const roll = (gamepad.buttons[5].value || gamepad.buttons[5].pressed ? 1 : 0)
                       - (gamepad.buttons[4].value || gamepad.buttons[4].pressed ? 1 : 0);
            if (roll) inv = rotate4(inv, gpLook * roll, 0, 0, 1);
        }

        viewMatrix = invert4(inv);

        const viewProj = multiply4(projectionMatrix, viewMatrix);
        worker.postMessage({ view: viewProj });

        const currentFps = 1000 / (now - lastFrame) || 0;
        avgFps = avgFps * 0.9 + currentFps * 0.1;

        if (vertexCount > 0) {
            document.getElementById("spinner").style.display = "none";
            gl.uniformMatrix4fv(u_view, false, viewMatrix);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
        } else {
            gl.clear(gl.COLOR_BUFFER_BIT);
            document.getElementById("spinner").style.display = "";
        }
        fps.innerText = Math.round(avgFps) + " fps";
        lastFrame = now;
        requestAnimationFrame(frame);
    };

    frame();
}

main().catch((err) => {
    document.getElementById("spinner").style.display = "none";
    document.getElementById("message").innerText = err.toString();
});
