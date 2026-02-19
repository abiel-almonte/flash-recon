import time
from threading import Thread
from typing import Iterator
from collections import defaultdict
from queue import Queue, Empty

import torch
from neural import MonoDepth

from .slam import SLAM
from .splat import Splat
from .viewer import Viewer
from geometry import pose_to_matrix

FrameGenerator = Iterator[torch.Tensor]


class _Logger:
    sessions = defaultdict(int)

    def log(self, caller: str = "", msg: str = ""):
        print(f"[flash-recon {caller}]: {msg}")

    def _periodic_log(self, period: int, name: str, msg: str):
        count = self.sessions[name]
        if count % period == 0:
            self.log(name, msg.format(i=count))

        self.sessions[name] += 1

    def log10(self, name: str, msg: str):
        self._periodic_log(10, name, msg)

    def log50(self, name: str, msg: str):
        self._periodic_log(50, name, msg)

    def log100(self, name: str, msg: str):
        self._periodic_log(100, name, msg)

    def log1000(self, name: str, msg: str):
        self._periodic_log(1000, name, msg)


_logger = _Logger()


class System:
    def __init__(self, cfg, persist: bool = False) -> None:
        self._persist = persist

        self.slam = SLAM(cfg)
        self.splat = Splat(cfg)
        self.viewer = Viewer(cfg)
        self.depth_model = MonoDepth(cfg).to("cuda")

        self._prev_n_keyframes = 0
        self._keyframe_q = Queue()

        self._running = False
        self._slam_done = False
        self._frame_generator: FrameGenerator | None = None

        self._splatter_thread = Thread(target=self._splatter, daemon=True)
        self._viewer_thread = Thread(target=self._viewer, daemon=True)

    @property
    def running(self):
        return self._running

    @property
    def frame_generator(self) -> FrameGenerator:
        if self._frame_generator is None:
            raise RuntimeError(
                "Frame generator is not set. "
                "Use system.set_frame_generator before calling run()."
            )
        return self._frame_generator

    def set_frame_generator(self, generator_fn: FrameGenerator) -> None:
        self._frame_generator = generator_fn

    def _slammer(self, frame: torch.Tensor, raw) -> None:
        self._prev_n_keyframes = self.slam.n_keyframes

        with torch.inference_mode():
            depth = self.depth_model(frame)
            self.slam(frame, depth)

            _logger.log50(
                "slammer.track",
                "localization step {i} completed"
                + f" | {self._prev_n_keyframes} keyframes",
            )

        if self.slam.n_keyframes > self._prev_n_keyframes:
            first_keyframe = self._prev_n_keyframes == 0
            self._prev_n_keyframes = self.slam.n_keyframes
            keyframe = self.slam.latest_keyframe
            keyframe.add_frame(raw)

            if first_keyframe:
                w2c = pose_to_matrix(keyframe.poses).squeeze(0)
                self.viewer.set_camera(w2c)

            self._keyframe_q.put(keyframe)

            _logger.log10("slammer.add", "new keyframe added | {i} keyframes")

    def _splatter(self):
        while self._running:
            try:
                keyframe = self._keyframe_q.get_nowait()
                if self.splat.add(keyframe):
                    _logger.log10(
                        "splatter.add", "new gaussians added | {i} gaussian updates"
                    )
            except Empty:
                pass

            snapshot = self.slam.snapshot
            loss = self.splat(snapshot)

            if loss != 0.0:
                _logger.log100(
                    "splatter.step",
                    "optimization step {i} completed" + f" | {loss:.2e} loss",
                )
                if not self._slam_done:
                    time.sleep(1 / 30)  # yield GPU to SLAM
            else:
                time.sleep(1 / 30)

    def _viewer(self):
        while self._running:
            splat_state = self.splat.snapshot

            if self.viewer(splat_state):
                _logger.log1000("viewer", "rendering step {i} completed")

            time.sleep(1 / 30)

    def run(self):
        self._running = True

        self._splatter_thread.start()
        _logger.log("splatter thread started")

        self._viewer_thread.start()
        _logger.log("viewer thread started")

        _logger.log("system started")
        try:
            for frame, raw in self.frame_generator:
                self._slammer(frame, raw)

            self._slam_done = True
            torch.cuda.empty_cache()

            if self._persist:
                while True:
                    self._splatter_thread.join()
        finally:
            self._running = False
            self._splatter_thread.join()
            self._viewer_thread.join()
            _logger.log("terminated")
