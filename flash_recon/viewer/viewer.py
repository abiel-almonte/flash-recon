import sys
import os
import atexit
import struct

import subprocess
import asyncio
import websockets

import torch
import numpy as np

from threading import Thread

from ..splat import SplatSnapshot

BYTES_PER_GAUSSIAN = 32


class _HttpCtx:
    def __init__(self, port):
        self.port = port

    def __enter__(self):
        self._http = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "http.server",
                "-d",
                f"{os.path.dirname(__file__)}",
                f"{self.port}",
            ]
        )
        atexit.register(self._http.terminate)

        return self

    def __exit__(self, *args):
        self._http.terminate()


class Viewer:
    def __init__(self, cfg) -> None:
        sc = cfg.get("viewer", {}).get("server", {})
        self._server_port = int(sc.get("port", 9090))

        cc = cfg.get("viewer", {}).get("client", {})
        self._client_port = int(cc.get("port", 9876))

        cam = cfg.get("cam", {})
        self._fx = float(cam.get("fx", 300.0))
        self._fy = float(cam.get("fy", 300.0))

        self._served = False
        self._package = None
        self._camera_package = None

        self._version = 0
        self._last_version = 0

        self._serve_thread = Thread(target=self._serve, daemon=True)

    def _serve(self):
        async def _serve():
            with _HttpCtx(self._client_port):
                async with websockets.serve(
                    self.send, "0.0.0.0", self._server_port
                ) as server:
                    await server.serve_forever()

        asyncio.run(_serve())

    async def send(self, websocket: websockets.ServerConnection):
        try:

            await websocket.send(
                bytes([0x03]) + struct.pack("<ff", self._fx, self._fy)
            )  # send intrinsics

            if self._camera_package is not None:
                await websocket.send(self._camera_package)
            while True:
                if self._last_version < self._version and self._package is not None:
                    await websocket.send(self._package)
                    self._last_version = self._version
                else:
                    await asyncio.sleep(1 / 90)
        except (websockets.ConnectionClosed, AttributeError):
            pass

    def _prepare(self, snapshot):
        n = snapshot.n
        if n == 0:
            return None

        try:
            means = snapshot.means[:n]
            colors = snapshot.colors[:n].clamp(0, 1) * 255
            n = min(n, means.shape[0], colors.shape[0])
            scales = snapshot.scales[:n].exp()
            alpha = snapshot.alphas[:n].sigmoid() * 255
            quats = snapshot.quats[:n]

            if any(t.shape[0] != n for t in [means, scales, colors, alpha, quats]):
                return None

            quats = ((quats / quats.norm(dim=-1, keepdim=True)) * 128 + 128).clamp(
                0, 255
            )
            return means, scales, colors, alpha, quats, n
        except (IndexError, RuntimeError):
            return None

    def _pack_gaussians(self, means, scales, colors, alpha, quats, n):
        """[mean:u32][scale:u32][rgba:u8][quat:u8]"""

        mean = means.to(torch.float32).cpu().numpy()
        scale = scales.to(torch.float32).cpu().numpy()
        rgb = colors.to(torch.uint8).cpu().numpy()
        a = alpha.to(torch.uint8).cpu().numpy()
        quat = quats.to(torch.uint8).cpu().numpy()

        rgba = np.concatenate([rgb, a.reshape(-1, 1)], 1)

        buf = np.empty((n, BYTES_PER_GAUSSIAN), dtype=np.uint8)
        buf[:, 0:12] = mean.view(np.uint8).reshape(n, 12)
        buf[:, 12:24] = scale.view(np.uint8).reshape(n, 12)
        buf[:, 24:28] = rgba
        buf[:, 28:32] = quat
        return buf

    def _create_package(self, snapshot, version):
        """Type 0x00: full replace. [type:u8][n:u32][v:32][data]"""

        prep = self._prepare(snapshot)
        if prep is None:
            return None
        means, scales, colors, alpha, quats, n = prep
        buf = self._pack_gaussians(means, scales, colors, alpha, quats, n)

        header = (
            bytes([0x00])
            + n.to_bytes(4, byteorder="little")
            + version.to_bytes(4, byteorder="little")
        )
        return header + buf.tobytes()

    def set_camera(self, w2c: torch.Tensor):
        """Type 0x02: [type:u8][16 x float32] = 65 bytes."""

        mat = w2c.to(torch.float32).cpu().numpy().T
        self._camera_package = bytes([0x02]) + mat.tobytes()

    def is_new_version(self):
        return self._last_version < self._version

    def __call__(self, snapshot: SplatSnapshot):
        if not self._served:
            self._serve_thread.start()
            self._served = True

        version = snapshot.version
        if not self._last_version < version:
            return False

        package = self._create_package(snapshot, version)
        if package is None:
            return False

        self._package = package
        self._version = version
        return True
