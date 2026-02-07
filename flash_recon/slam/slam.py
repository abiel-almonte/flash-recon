import torch
import geometry
from geometry import Intrinsics, identity_pose

from .factor_graph import FactorGraph
from .keyframe_buffer import KeyFrameBuffer
from .video_odemetry import VideoOdometry

from .structs import EdgeRequest, EdgeStrategy


class SLAM:
    def __init__(self, cfg) -> None:
        self.buffer = KeyFrameBuffer(cfg)
        self.graph = FactorGraph(cfg)
        self.vo = VideoOdometry(cfg)

        H_out = int(cfg.get("cam", {}).get("H_out", 480))
        W_out = int(cfg.get("cam", {}).get("W_out", 640))
        fx = float(cfg.get("cam", {}).get("fx", 300.0))
        fy = float(cfg.get("cam", {}).get("fy", 300.0))
        cx = float(cfg.get("cam", {}).get("cx", W_out / 2.0))
        cy = float(cfg.get("cam", {}).get("cy", H_out / 2.0))

        self.beta = cfg.get("tracking", {}).get("beta")
        self.local_nms = cfg.get("tracking", {}).get("local", {}).get("nms")
        self.local_window = cfg.get("tracking", {}).get("local", {}).get("window")
        self.local_thresh = cfg.get("tracking", {}).get("local", {}).get("thresh")
        self.local_radius = cfg.get("tracking", {}).get("local", {}).get("radius")
        self.local_max_factors = (
            cfg.get("tracking", {}).get("local", {}).get("max_factors")
        )

        self.global_nms = cfg.get("tracking", {}).get("global", {}).get("nms")
        self.global_window = cfg.get("tracking", {}).get("global", {}).get("window")
        self.global_thresh = cfg.get("tracking", {}).get("global", {}).get("thresh")
        self.global_radius = cfg.get("tracking", {}).get("global", {}).get("radius")

        self.loop_nms = cfg.get("tracking", {}).get("loop", {}).get("nms")
        self.loop_window = cfg.get("tracking", {}).get("loop", {}).get("window")
        self.loop_thresh = cfg.get("tracking", {}).get("loop", {}).get("thresh")
        self.loop_radius = cfg.get("tracking", {}).get("loop", {}).get("radius")

        self.intrinsics = Intrinsics(fx, fy, cx, cy, device=cfg.get("device", "cuda"))

        self.warmup_keyframes = int(cfg.get("tracking", {}).get("warmup", 8))
        self.max_age = int(cfg.get("tracking", {}).get("max_age", 25))
        self.global_ba_freq = int(cfg.get("global_ba", {}).get("freq", 20))
        self.publish_freq = int(cfg.get("publish_freq", {}))
        self.motion_threshold = float(cfg.get("motion_threshold", {}))
        self.dist_threshold = float(cfg.get("dist_threshold", {}))

        self.prev_fmap, self.prev_net, self.prev_inp = None, None, None
        self._initialized = False

    def _initialize(self, frame, mono_depth):
        fmap, net, inp = self.vo.extract(frame)
        dummy_pose = identity_pose(1)

        self.buffer.append(
            pose=dummy_pose,
            disp=1 / mono_depth,
            mono_depth=mono_depth,
            fmap=fmap,
            net=net,
            inp=inp,
        )

        self.prev_fmap, self.prev_net, self.prev_inp = fmap, net, inp
        self._initialized = True

    def _bootstrap(self):
        count = len(self.buffer)
        snapshot = self.buffer.snapshot

        self.graph.add_neighborhood_factors(t0=0, t1=count, rad=3, buffer=snapshot)
        self.vo(self.buffer, self.graph, steps=8, alternate=False)

        self.graph.add_proximity_factors(
            EdgeRequest(
                strategy=EdgeStrategy.LOCAL,
                buffer=snapshot,
                beta=self.beta,
                t0=0,
                t1=0,
                rad=2,
                nms=2,
                remove=False,
            )
        )
        self.vo(self.buffer, self.graph, steps=8, alternate=False)

        self.buffer.propagate()
        self.graph.remove_factors(
            self.graph.ii < self.warmup_keyframes - 4, store_inac=True
        )

    def _is_keyframe(self, fmap):
        return (
            self.vo.compute_motion(self.prev_fmap, fmap, self.prev_net, self.prev_inp)
            > self.motion_threshold
        )

    def _reject_keyframe(self) -> bool:
        count = len(self.buffer)
        dist = geometry.compute_distance(
            *self.buffer.get_geometric_attrs(),
            ii=[count - 2],
            jj=[count - 1],
            beta=self.beta,
            bidirectional=True,
        ).item()

        if dist < self.dist_threshold:
            self.graph.remove_edge(count - 1)
            self.buffer.remove(count - 1)
            return True

        return False

    def _add_local_edges(self, count, snapshot):
        self.graph.add_proximity_factors(
            EdgeRequest(
                strategy=EdgeStrategy.LOCAL,
                buffer=snapshot,
                beta=self.beta,
                t0=count - 5,
                t1=max(count - self.local_window, 0),
                rad=self.local_radius,
                nms=self.local_nms,
                thresh=self.local_thresh,
                remove=True,
            )
        )

    def _add_loop_edges(self, count, snapshot):
        t_start_loop = max(0, count - self.loop_window)
        max_factors = 8 * self.loop_window - len(self.graph)

        self.graph.add_proximity_factors(
            EdgeRequest(
                strategy=EdgeStrategy.LOOP,
                buffer=snapshot,
                beta=self.beta,
                t0_loop=t_start_loop,
                t0=0,
                t1=count,
                rad=self.loop_radius,
                nms=self.loop_nms,
                thresh=self.loop_thresh,
                max_factors=max_factors,
                loop=True,
            )
        )

    def _add_global_edges(self, count, snapshot):
        max_factors = ((self.global_radius + 2) * 2) * count

        self.graph.add_proximity_factors(
            EdgeRequest(
                strategy=EdgeStrategy.GLOBAL,
                buffer=snapshot,
                beta=self.beta,
                t0=0,
                t1=count,
                rad=self.global_radius,
                nms=self.global_nms,
                thresh=self.global_thresh,
                max_factors=max_factors,
                loop=False,
            )
        )

    def _remove_nonlocal_edges(self, n_local_edges):
        mask = torch.arange(len(self.graph), device=self.graph.device) >= n_local_edges
        self.graph.remove_factors(mask, store_inac=False)

    def _track(self, frame, mono_depth):
        fmap, net, inp = self.vo.extract(frame)

        if not self._is_keyframe(fmap):
            return

        self.prev_fmap, self.prev_net, self.prev_inp = fmap, net, inp

        self.buffer.append(
            mono_depth=mono_depth,
            fmap=fmap,
            net=net,
            inp=inp,
        )

        if len(self.buffer) < self.warmup_keyframes:
            return

        if len(self.buffer) == self.warmup_keyframes:
            self._bootstrap()
            return

        # steady-state tracking
        count = len(self.buffer)
        snapshot = self.buffer.snapshot

        self.graph.remove_factors(self.graph.age > self.max_age, store_inac=True)
        self._add_local_edges(count, snapshot)

        self.vo(self.buffer, self.graph, steps=8, alternate=True)

        if not self._reject_keyframe():
            n_local_edges = len(self.graph)

            if count > self.local_window:
                self._add_loop_edges(count, snapshot)

            self.vo(self.buffer, self.graph, steps=4, alternate=True)
            self._remove_nonlocal_edges(n_local_edges)

        self.buffer.propagate()

        count = len(self.buffer)
        if count % self.global_ba_freq == 0:

            self.buffer.normalize()
            snapshot = self.buffer.snapshot

            n_local_edges = len(self.graph)
            self._add_global_edges(count, snapshot)

            if len(self.graph) > n_local_edges:
                self.vo(self.buffer, self.graph, steps=2, alternate=True)
                self._remove_nonlocal_edges(n_local_edges)

            self.buffer.set_needs_update(slice(0, count))
            self.buffer.update_vmask(up=True)

    def __call__(self, frame, mono_depth):
        if not self._initialized:
            self._initialize(frame, mono_depth)
            return
        self._track(frame, mono_depth)
