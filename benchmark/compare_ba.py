
import sys
import argparse
import glob
import torch
import lietorch

sys.path.append("/workspace")
sys.path.append("/workspace/Splat-SLAM")
sys.path.append("/workspace/benchmark")

from pose_utils import Pose, Intrinsics
from ba_utils import full_ba, ba_scale_shift
from projective_utils import projective_transform

def compute_reprojection_error(target, weight, poses, disps, intrinsics, ii, jj, n = 0):
    T = poses.t.shape[0]
    
    
    if n == 0:
        n = T
    
    use_window = n < T
    if use_window:
        poses_window = poses[:n]
        disps_window = disps[:n]

        valid = (ii < n) & (jj < n)
        ii_window = ii[valid].contiguous()
        jj_window = jj[valid].contiguous()
        target_window = target[valid].contiguous()
        weight_window = weight[valid].contiguous()
    else:
        poses_window = poses
        disps_window = disps
        ii_window = ii
        jj_window = jj
        target_window = target
        weight_window = weight
    
    coords, _ = projective_transform(poses_window, disps_window, intrinsics, ii_window, jj_window, jacobian=False)
    residual = (target_window - coords) * weight_window
    return (residual ** 2).sum().item()


def compute_depth_alignment_cost(disps, mono_disps, scales, shifts, valid_depth_mask, ii, alpha: float, n=0):
    """Depth alignment cost used in depth_scale (approx. ref objective weighting).

    cost = alpha * sum( (s * (d - (scale*mono + shift)))^2 ),
    where s=10 for valid_depth_mask, else s=1, evaluated over source keyframes.
    """
    T = disps.shape[0]
    if n == 0:
        n = T
    
    kf_indices = torch.unique(ii)
    kf_indices = kf_indices[kf_indices < n]
    
    if kf_indices.numel() == 0:
        return 0.0
    
    d = disps[kf_indices]
    m = mono_disps[kf_indices]
    v = valid_depth_mask[kf_indices]
    s = scales[kf_indices].view(-1, 1, 1)
    q = shifts[kf_indices].view(-1, 1, 1)
    residual = d - (s * m + q)
    scale = torch.where(v > 0, torch.tensor(10.0, device=d.device, dtype=d.dtype), torch.tensor(1.0, device=d.device, dtype=d.dtype))
    cost = float(alpha) * (scale * residual).pow(2).sum().item()
    return cost

def run_ba(f_name, data, ref):
    is_ba_ss = data["ba_type"] == "depth_scale"
    
    if is_ba_ss:
        poses_state = data["poses_vec"].squeeze(0).to("cuda")
        disps_state = data["disps"].squeeze(0).to("cuda")
        intr = data["intrinsics"].squeeze(0).to("cuda")
        ii = data["ii"].to("cuda")
        jj = data["jj"].to("cuda")
        target = data["target"].squeeze(0).to("cuda")
        weight = data["weight"].squeeze(0).to("cuda")
        eta_m = data["eta"].to("cuda")
        mono = data["mono_disps"].squeeze(0).to("cuda")
        scales_state = data["scales"].squeeze(0).to("cuda")
        shifts_state = data["shifts"].squeeze(0).to("cuda")
        vmask = data["valid_depth_mask_small"].squeeze(0).to("cuda")
        ignore_frames = data["ignore_frames"]
        lm = data["lm"]
        ep = data["ep"]
        alpha = data["alpha"]

        poses_state = Pose(poses_state[:, :3].contiguous(), poses_state[:, 3:].contiguous())
        intr = Intrinsics(float(intr[0, 0]), float(intr[0, 1]), float(intr[0, 2]), float(intr[0, 3]), device="cuda")

        init_error = compute_depth_alignment_cost(disps_state, mono, scales_state, shifts_state, vmask, ii, alpha)

        # eta_m is [M, H, W] for M source keyframes - scaling applied internally
        disps_state, scale_shift_params_out = ba_scale_shift(target, weight, eta_m, poses_state, disps_state, intr, ii, jj, mono, scales_state, shifts_state, vmask, ignore_frames, lm, ep, alpha)
        scales_state, shifts_state = scale_shift_params_out.split([1, 1], dim= -1)

        new_final_error = compute_depth_alignment_cost(disps_state, mono, scales_state, shifts_state, vmask, ii, alpha)
        
        ref_final_disps = ref["disps"].squeeze(0).to("cuda")
        ref_final_scale_shift= ref["wqs"].squeeze(0).to("cuda")
        ref_final_scale, ref_final_shift = ref_final_scale_shift.split([1, 1], dim= -1)

        ref_final_error = compute_depth_alignment_cost(ref_final_disps, mono, ref_final_scale, ref_final_shift, vmask, ii, alpha)

        new_error_reduction = (100 * (init_error - new_final_error) / init_error) if init_error else 0.0
        ref_error_reduction = (100 * (init_error - ref_final_error) / init_error) if init_error else 0.0
        is_good = new_error_reduction > 0 and (new_error_reduction - ref_error_reduction) > 5.0
        is_bad = ref_error_reduction > 0 and (ref_error_reduction - new_error_reduction) > 5.0

        print(f"{f_name} - [Depth & Scale]")
        print(f"  new:  {new_error_reduction:.2f}%")
        print(f"  ref:  {ref_error_reduction:.2f}%")

        return {"f_name": f_name, "new": new_error_reduction, "ref": ref_error_reduction, "bad" : is_bad, "good": is_good}

    else:
        t1 = data["t1"]
        poses_state = data["poses"].squeeze(0).to("cuda")
        disps_state = data["disps"].squeeze(0).to("cuda")
        intr = data["intrinsics"].squeeze(0).to("cuda")
        ii = data["ii"].to("cuda")
        jj = data["jj"].to("cuda")
        target = data["target"].squeeze(0).permute(0,2,3,1).to("cuda")
        weight = data["weight"].squeeze(0).permute(0,2,3,1).to("cuda")
        eta_m = data["eta"].to("cuda")
        lm = data["lm"]
        ep = data["ep"]
        iters = data["iterations"]

        poses_state = Pose(poses_state[:, :3].contiguous(), poses_state[:, 3:].contiguous())
        intr = Intrinsics(float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3]), device="cuda")

        init_error = compute_reprojection_error(target, weight, poses_state, disps_state, intr, ii, jj, t1)

        t0 = data["t0"]
        for _ in range(iters):
            poses_state, disps_state = full_ba(target, weight, eta_m, poses_state, disps_state, intr, ii, jj, t1, lm, ep, num_fixed_poses=t0)
        
        new_final_error = compute_reprojection_error(target, weight, poses_state, disps_state, intr, ii, jj, t1)

        ref_final_poses = ref["poses"].squeeze(0).to("cuda")
        ref_final_disps = ref["disps"].squeeze(0).to("cuda")

        ref_final_poses = Pose(ref_final_poses[:, :3].contiguous(), ref_final_poses[:, 3:].contiguous())
        ref_final_error = compute_reprojection_error(target, weight, ref_final_poses, ref_final_disps, intr, ii, jj, t1)

        new_error_reduction = (100 * (init_error - new_final_error) / init_error) if init_error else 0.0
        ref_error_reduction = (100 * (init_error - ref_final_error) / init_error) if init_error else 0.0
        is_good = new_error_reduction > 0 and (new_error_reduction - ref_error_reduction) > 5.0
        is_bad = ref_error_reduction > 0 and (ref_error_reduction - new_error_reduction) > 5.0

        print(f"{f_name} - [Pose & Depth]")
        print(f"  opt:  {t1}")
        print(f"  new:  {new_error_reduction:.2f}%")
        print(f"  ref:  {ref_error_reduction:.2f}%\n")

        return {"f_name": f_name, "new": new_error_reduction, "ref": ref_error_reduction, "bad" : is_bad, "good": is_good}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    print("=" * 80)
    print("BA COMPARISON")
    print("=" * 80)
    files_all = sorted(glob.glob("/workspace/ba_inputs/dspo_*_out.pt"))
    if args.limit > 0:
        files_all = files_all[:args.limit]
    print(f"\nFound {len(files_all)} files \n")
    if not files_all:
        return

    results = []
    n_bad_results = []
    n_good_results = []

    for f_out in files_all:
        try:
            f_in = f_out.replace("_out", "")
            f_name = f_in.split('/')[-1]

            ref = torch.load(f_out)
            data = torch.load(f_in)

            res = run_ba(f_name, data, ref)
            results.append(res)

            if res["bad"]:
                n_bad_results.append(res)
            elif res["good"]:
                n_good_results.append(res)
                

        except Exception as e:
            print(f"❌ {f_in.split('/')[-1]}: ERROR - {e}")
            import traceback
            traceback.print_exc()

    if results:
        avg_new = sum(x["new"] for x in results) / len(results)
        avg_ref = sum(x["ref"] for x in results) / len(results)
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Steps: {len(results)}")
        print(f"Bad Steps: {len(n_bad_results)}")
        print(f"Avg ours: {avg_new:.2f}% | Avg ref: {avg_ref:.2f}%")
        if avg_new >= avg_ref - 0.5:
            print("\nOur BA matches or improves on the reference on average")
        else:
            print("\nOur BA underperforms the reference on average")
        print("Good Summary")
        print(80*"=")
        for good_res in n_good_results:
            print(f"{good_res["f_name"]}:")
            print(f"  new:  {good_res["new"]:.2f}%")
            print(f"  ref:  {good_res["ref"]:.2f}%")
        print(80*"=")
        print("Bad Summary")
        print(80*"=")
        for bad_res in n_bad_results:
            print(f"{bad_res["f_name"]}:")
            print(f"  new:  {bad_res["new"]:.2f}%")
            print(f"  ref:  {bad_res["ref"]:.2f}%")
        print(80*"=")
        

if __name__ == "__main__":
    main()
