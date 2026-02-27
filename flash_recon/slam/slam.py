import torch

from geometry import identity_pose, compute_distance

from .utils import KeyFrameBuffer, LocalGraph, GlobalGraph, SLAMSnapshot
from .video_odometry import VideoOdometry


class SLAM:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = cfg.get("device", "cuda")

        down = int(cfg["cam"].get("down_scale", 8))
        self.down_scale = down
        self.ht = int(cfg["cam"]["H_out"]) // down
        self.wd = int(cfg["cam"]["W_out"]) // down

        self.buffer = KeyFrameBuffer(cfg)
        self.vo = VideoOdometry(cfg)
        self.graph = LocalGraph(cfg)

        # local
        fc = cfg["tracking"]["local"]
        self.warmup = int(cfg["tracking"]["warmup"])
        self.max_age = int(cfg["tracking"]["max_age"])
        self.max_factors = int(fc["max_factors"])
        self.beta = float(cfg["tracking"]["beta"])
        self.keyframe_thresh = float(fc["keyframe_thresh"])
        self.local_window = int(fc["window"])
        self.local_thresh = float(fc["thresh"])
        self.local_radius = int(fc["radius"])
        self.local_nms = int(fc["nms"])

        # loop closure
        lc = cfg["tracking"].get("loop_closure", {})
        self.enable_loop = bool(fc.get("enable_loop", False))
        self.loop_window = int(lc.get("window", 25))
        self.loop_thresh = float(lc.get("thresh", 24.0))
        self.loop_radius = int(lc.get("radius", 2))
        self.loop_nms = int(lc.get("nms", 2))

        # periodic global BA
        bc = cfg["tracking"].get("global", {})
        self.enable_global = bool(bc.get("enabled", False))
        self.global_freq = int(bc.get("freq", 20))
        self.global_thresh = float(bc.get("thresh", 24.0))
        self.global_radius = int(bc.get("radius", 2))
        self.global_nms = int(bc.get("nms", 2))
        self.global_normalize_enabled = bool(bc.get("normalize", False))

        self.motion_thresh = float(cfg["tracking"]["motion_filter"]["thresh"])

        self.prev_fmap = None
        self.prev_net = None
        self.prev_inp = None

    @property
    def n_keyframes(self):
        return len(self.buffer)

    @property
    def snapshot(self):
        count = len(self.buffer)
        poses = self.buffer._poses
        disps = self.buffer._disps_up
        vmask = self.buffer.get_vmask()
        intrinsics = self.buffer._intrinsics
        return SLAMSnapshot(
            poses=poses[:count],
            disps=disps[:count],
            vmask=vmask[:count],
            intrinsics=intrinsics,
            version=self.buffer.version,
        )

    @property
    def latest_keyframe(self):
        idx = len(self.buffer) - 1
        return self.buffer.get_keyframe(idx)

    def _try_add_keyframe(self, frame, mono_depth=None, tstamp=0.0) -> bool:
        fmap, net, inp = self.vo.extract(frame)
        if len(self.buffer) == 0:
            self.prev_fmap, self.prev_net, self.prev_inp = fmap, net, inp

            net_buf = net[0, 0].expand_as(net).contiguous()
            inp_buf = inp[0, 0].expand_as(inp).contiguous()

            self.buffer.append(
                pose=identity_pose(1),
                disp=1.0,
                mono_depth=mono_depth,
                fmap=fmap,
                net=net_buf,
                inp=inp_buf,
                tstamp=tstamp,
            )
            return True

        motion = self.vo.compute_motion(
            self.prev_fmap, fmap, self.prev_net, self.prev_inp
        )
        if motion <= self.motion_thresh:
            return False

        self.prev_fmap, self.prev_net, self.prev_inp = fmap, net, inp
        self.buffer.append(
            mono_depth=mono_depth, fmap=fmap, net=net, inp=inp, tstamp=tstamp
        )
        return True

    def _try_reject_keyframe(self) -> bool:
        count = len(self.buffer)
        if count < 5:
            return False

        poses, disps, intrinsics = self.buffer.get_geometric_attrs()
        ii = torch.tensor([count - 4], device=self.device)
        jj = torch.tensor([count - 2], device=self.device)
        dist = compute_distance(
            poses, disps, intrinsics, ii=ii, jj=jj, beta=self.beta, bidirectional=True
        )
        if dist.item() < 2.0 * self.keyframe_thresh:
            self.graph.remove_keyframe(count - 3, count)
            self.buffer.remove(count - 3)
            return True
        return False

    def _create_loop_graph(self):
        if len(self.buffer) <= self.local_window:
            return None, 0

        t_end = len(self.buffer)
        window = self.loop_window
        max_factors = 8 * window
        t_start_loop = max(0, t_end - window)

        loop_graph = GlobalGraph.from_local(self.graph)
        loop_graph.max_factors = max_factors

        n_edges = loop_graph.add_proximity_edges(
            buffer=self.buffer,
            t_start=0,
            t_end=t_end,
            t_start_loop=t_start_loop,
            nms=self.loop_nms,
            radius=self.loop_radius,
            thresh=self.loop_thresh,
            max_factors=max_factors - len(loop_graph),
            beta=self.beta,
            loop=True,
        )

        if n_edges == 0:
            del loop_graph
            return None, 0

        return loop_graph, t_start_loop + 1

    def _create_global_graph(self):
        if self.global_normalize_enabled:
            self.buffer.normalize()

        t_end = len(self.buffer)
        max_factors = 16 * t_end

        global_graph = GlobalGraph.from_local(self.graph)
        global_graph.max_factors = max_factors

        n_edges = global_graph.add_proximity_edges(
            buffer=self.buffer,
            t_start=0,
            t_end=t_end,
            t_start_loop=0,
            nms=self.global_nms,
            radius=self.global_radius,
            thresh=self.global_thresh,
            max_factors=max_factors,
            beta=self.beta,
            loop=False,
        )

        if n_edges == 0:
            del global_graph
            return None, 0

        return global_graph, 0

    def _bootstrap(self) -> None:
        count = len(self.buffer)

        self.graph.add_from_buffer(
            self.buffer, *self.graph.neighborhood_pairs(0, count, r=3)
        )
        self.vo(self.buffer, self.graph, steps=8, alternate=False, t0=1)

        self.graph.add_proximity_edges(
            buffer=self.buffer,
            t0=0,
            t1=0,
            count=count,
            rad=2,
            nms=2,
            thresh=self.local_thresh,
            max_factors=self.max_factors,
        )
        self.vo(self.buffer, self.graph, steps=8, alternate=False, t0=1)

        self.buffer.propagate(use_init_mean=True)
        self.buffer.set_needs_update(torch.arange(0, count, device=self.device))
        self.buffer.update_vmask(up=True)
        self.graph.remove(self.graph.ii < self.warmup - 4, store=True)

    def _track(self) -> None:
        count = len(self.buffer)

        if self.graph.has_edges():
            self.graph.remove(self.graph.age > self.max_age, store=True)

        # add local_edegs
        self.graph.add_proximity_edges(
            buffer=self.buffer,
            t0=count - 5,
            t1=max(count - self.local_window, 0),
            count=count,
            rad=self.local_radius,
            nms=self.local_nms,
            beta=self.beta,
            thresh=self.local_thresh,
            max_factors=self.max_factors,
            remove=True,
        )

        # local update
        self.vo(buffer=self.buffer, graph=self.graph, steps=3, alternate=False)

        if not self._try_reject_keyframe():
            if self.enable_loop:
                loop_graph, t0 = self._create_loop_graph()
                if loop_graph is not None:
                    self.vo(
                        buffer=self.buffer,
                        graph=loop_graph,
                        t0=t0,
                        steps=4,
                        alternate=True,
                    )
                    del loop_graph
                else:
                    self.vo(
                        buffer=self.buffer, graph=self.graph, steps=2, alternate=True
                    )
            else:
                self.vo(buffer=self.buffer, graph=self.graph, steps=2, alternate=False)

        count = len(self.buffer)
        if self.graph.ii.numel() > 0:
            t_min = int(self.graph.ii.min())
        else:
            t_min = max(count - 1, 0)
        self.buffer.set_needs_update(torch.arange(t_min, count, device=self.device))
        self.buffer.update_vmask(up=True)
        self.buffer.propagate(use_init_mean=False)

        if self.enable_global and count % self.global_freq == 0:
            global_graph, t0 = self._create_global_graph()
            if global_graph is not None:
                self.vo(
                    buffer=self.buffer,
                    graph=global_graph,
                    t0=t0,
                    steps=2,
                    alternate=True,
                )
                del global_graph

    def __call__(self, frame, mono_depth=None, tstamp=0.0) -> None:
        if not self._try_add_keyframe(frame, mono_depth, tstamp=tstamp):
            return

        count = len(self.buffer)
        if count < self.warmup:
            return

        if count == self.warmup:
            self._bootstrap()
            return

        self._track()

    def finalize(self, steps_per_pass=(7, 12)):
        for steps in steps_per_pass:
            torch.cuda.empty_cache()

            global_graph, t0 = self._create_global_graph()
            if global_graph is None:
                break

            self.vo(
                buffer=self.buffer,
                graph=global_graph,
                t0=t0,
                steps=steps,
                alternate=False,
            )
            del global_graph

    def __len__(self) -> int:
        return len(self.graph)
