import sys
import os
import atexit

import subprocess
import asyncio
import websockets

import torch
import numpy as np

from threading import Thread

from ..splat import SplatSnapshot

BYTES_PER_GAUSSIAN = 32
CHUNK_SIZE = 8192  # gaussians per chunk update


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


def _pack_gaussians(means, scales, colors, alpha, quats, n):
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


class Viewer:
    def __init__(self, cfg) -> None:
        sc = cfg.get("viewer", {}).get("server", {})
        self._server_port = int(sc.get("port", 9090))

        cc = cfg.get("viewer", {}).get("client", {})
        self._client_port = int(cc.get("port", 9876))

        self._served = False
        self._package = None
        self._camera_package = None
        self._last_n = 0
        self._chunk_offset = 0
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
            while True:
                if self._camera_package is not None:
                    await websocket.send(self._camera_package)
                    self._camera_package = None
                if self._package is not None:
                    await websocket.send(self._package)
                    self._package = None
                else:
                    await asyncio.sleep(1 / 30)
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

    def _package_full(self, snapshot):
        """Type 0x00: full replace. [type:u8][n:u32][data]"""
        prep = self._prepare(snapshot)
        if prep is None:
            return None
        means, scales, colors, alpha, quats, n = prep
        buf = _pack_gaussians(means, scales, colors, alpha, quats, n)

        header = bytes([0x00]) + n.to_bytes(4, byteorder="little")
        return header + buf.tobytes()

    def _package_chunk(self, snapshot):
        """Type 0x01: chunk update. [type:u8][offset:u32][count:u32][data]"""
        prep = self._prepare(snapshot)
        if prep is None:
            return None
        means, scales, colors, alpha, quats, n = prep

        offset = self._chunk_offset
        if offset >= n:
            offset = 0

        end = min(offset + CHUNK_SIZE, n)
        count = end - offset

        buf = _pack_gaussians(
            means[offset:end],
            scales[offset:end],
            colors[offset:end],
            alpha[offset:end],
            quats[offset:end],
            count,
        )

        self._chunk_offset = end if end < n else 0

        header = bytes([0x01])
        header += offset.to_bytes(4, byteorder="little")
        header += count.to_bytes(4, byteorder="little")
        return header + buf.tobytes()

    def set_camera(self, w2c: torch.Tensor):
        """Type 0x02: [type:u8][16 x float32] = 65 bytes."""

        mat = w2c.to(torch.float32).cpu().numpy().T
        self._camera_package = bytes([0x02]) + mat.tobytes()

    def __call__(self, snapshot: SplatSnapshot):
        if not self._served:
            self._serve_thread.start()
            self._served = True

        if snapshot.n != self._last_n:
            package = self._package_full(snapshot)
            if package is None:
                return False

            self._last_n = snapshot.n
            self._chunk_offset = 0
        else:
            package = self._package_chunk(snapshot)
            if package is None:
                return False

        self._package = package
        return True


if __name__ == "__main__":
    v = Viewer({})

    v._serve()
