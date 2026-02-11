import torch

from geometry import identity_pose, compute_distance

from .utils import KeyFrameBuffer, LocalGraph, GlobalGraph
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
        lc = cfg["tracking"]["loop_closure"]
        self.loop_window = int(lc["window"])
        self.loop_thresh = float(lc["thresh"])
        self.loop_radius = int(lc["radius"])
        self.loop_nms = int(lc["nms"])

        # global
        bc = cfg["tracking"]["global"]
        self.global_freq = int(bc["freq"])
        self.global_thresh = float(bc["thresh"])
        self.global_radius = int(bc["radius"])
        self.global_nms = int(bc["nms"])

        self.motion_thresh = float(cfg["tracking"]["motion_filter"]["thresh"])

        self.prev_fmap = None
        self.prev_net = None
        self.prev_inp = None

    def _try_add_keyframe(self, frame, mono_depth) -> bool:
        fmap, net, inp = self.vo.extract(frame)
        if len(self.buffer) == 0:
            self.prev_fmap, self.prev_net, self.prev_inp = fmap, net, inp
            self.buffer.append(
                pose=identity_pose(1),
                disp=1.0,
                mono_depth=mono_depth,
                fmap=fmap,
                net=net,
                inp=inp,
            )
            return True

        motion = self.vo.compute_motion(
            self.prev_fmap, fmap, self.prev_net, self.prev_inp
        )
        if motion <= self.motion_thresh:
            return False

        self.prev_fmap, self.prev_net, self.prev_inp = fmap, net, inp
        self.buffer.append(mono_depth=mono_depth, fmap=fmap, net=net, inp=inp)
        return True

    def _try_reject_keyframe(self) -> bool:
        count = len(self.buffer)
        poses, disps, intrinsics = self.buffer.get_geometric_attrs()
        ii = torch.tensor([count - 2], device=self.device)
        jj = torch.tensor([count - 1], device=self.device)
        dist = compute_distance(
            poses, disps, intrinsics, ii=ii, jj=jj, beta=self.beta, bidirectional=True
        )
        if dist.item() < self.keyframe_thresh:
            self.graph.remove_keyframe(count - 1)
            self.buffer.remove(count - 1)
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
        self.buffer.normalize()

        t_end = len(self.buffer)
        max_factors = ((self.global_radius + 2) * 2) * (t_end)

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
        # evict old edges
        if self.graph.corr_initialized():
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
        self.vo(buffer=self.buffer, graph=self.graph, steps=8, alternate=True)

        # reject or loop-close
        if not self._try_reject_keyframe():
            loop_graph, t0 = self._create_loop_graph()

            if loop_graph is not None:
                self.vo(
                    buffer=self.buffer, graph=loop_graph, t0=t0, steps=4, alternate=True
                )
                del loop_graph
            else:
                self.vo(buffer=self.buffer, graph=self.graph, steps=4, alternate=True)

        self.buffer.set_needs_update(
            torch.arange(int(self.graph.ii.min()), count, device=self.device)
        )
        self.buffer.update_vmask(up=True)
        self.buffer.propagate(use_init_mean=False)

        if len(self.buffer) % self.global_freq == 0:
            global_graph, t0 = self._create_global_graph()

            if global_graph is not None:
                self.vo(buffer=self.buffer, graph=self.graph, steps=2, alternate=True)
                del global_graph

    def __call__(self, frame, mono_depth) -> None:
        if not self._try_add_keyframe(frame, mono_depth):
            return

        count = len(self.buffer)
        if count < self.warmup:
            return

        if count == self.warmup:
            self._bootstrap()
            return

        self._track()

    def __len__(self) -> int:
        return len(self.graph)
