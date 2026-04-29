`flash-recon` **- Real-time 3D reconstruction from monocular video.**

<p align="center">
  <img src="assets/demo.gif" alt="VOOM demo" width="800"/>
</p>

Estimate poses, depth, and gaussians all at once — visualize it live on your browser.

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

Run on TUM fr1_desk:

```bash
python -m examples.run_system  # forward ports 9090 & 9876
```

---

**flash-recon**, SLAM that doesn't make you wait.