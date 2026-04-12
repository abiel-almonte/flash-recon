`flash-recon` **- Real-time 3D reconstruction from monocular video.**

Estimate poses, depth, and gaussians all at once.

```python
from flash_recon import System

system = System(cfg, persist=True)

def frames():
    for image in camera_stream():
        yield normalize(image), image

system.set_frame_generator(frames())
system.run()  # open http://localhost:9876
```

Run SLAM standalone for evaluation:
```python
from flash_recon.slam import SLAM

slam = SLAM(cfg)
for frame in video:
    slam(frame, tstamp=t)

slam.finalize()
trajectory = slam.buffer.get_cam2world(kf_idx)  # SE(3)
```

Evaluate on EuRoC:
```bash
bash evals/evaluate_euroc.sh
```

---

**flash-recon**, SLAM that doesn't make you wait.
