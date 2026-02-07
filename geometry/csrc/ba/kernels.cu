#include "kernels.h"
#include "kernels/scale_shift.cuh"
#include "kernels/pose_depth.cuh"

std::tuple<Tensor, Tensor> fused_projective_transform_with_reduction_cuda(
    Tensor t, // [E, 3]
    Tensor q, // [E, 4]
    Tensor disps, // [E, ht, wd]
    Tensor intrinsics,
    Tensor ii, // [E]
    Tensor jj, // [E]
    Tensor target, // [E, ht, wd, 2]
    Tensor weight // [E, ht, wd, 2]
) {
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(disps);
    CHECK_INPUT(intrinsics);
    CHECK_INPUTL(ii);
    CHECK_INPUTL(jj);

    const int E = ii.size(0);
    const int H = disps.size(1);
    const int W = disps.size(2);

    auto opts = disps.options();

    Tensor curv = torch::zeros({E, H*W}, opts);
    Tensor rhs = torch::zeros({E, H*W}, opts);

    fused_projective_transform_with_reduction_kernel<<<E, THREADS>>>(
        t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        intrinsics.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
        target.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        weight.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        curv.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    return {curv, rhs};
}

std::tuple<Tensor, Tensor, Tensor> fused_depth_jacobians_cuda(
    Tensor disps,// [U, ht, wd]
    Tensor mono_depths, // [U, ht, wd]
    Tensor valid_depth, // [U, ht, wd]
    Tensor scales, // [U]
    Tensor shifts, // [U]
    Tensor ignore, // [U]
    const float alpha
) {

    CHECK_INPUT(disps);
    CHECK_INPUT(mono_depths);
    CHECK_INPUTB(valid_depth);
    CHECK_INPUT(scales);
    CHECK_INPUT(shifts);
    CHECK_INPUTB(ignore);

    const int E = disps.size(0);
    const int ht = disps.size(1);
    const int wd = disps.size(2);

    const float sqrt_alpha = std::sqrt(alpha);
    const float sqrt_alpha10 = sqrt_alpha*10;

    auto opts = disps.options();

    Tensor Jd = torch::zeros({E, ht*wd}, opts);
    Tensor Rd = torch::zeros({E, ht*wd}, opts);
    Tensor Jwq = torch::zeros({E, ht*wd, 2}, opts);

    fused_depth_jacobians_kernel<<<E, THREADS>>>(
        disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        mono_depths.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        valid_depth.packed_accessor32<bool, 3, torch::RestrictPtrTraits>(),
        scales.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        shifts.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        ignore.packed_accessor32<bool, 1, torch::RestrictPtrTraits>(),
        sqrt_alpha,
        sqrt_alpha10,
        Jwq.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        Jd.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        Rd.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    return {Jwq, Jd, Rd};
}


std::vector<Tensor> fused_project_and_accumulate_cuda(
    Tensor t, // [P, 3]
    Tensor q, // [P, 4]
    Tensor disps, // [E, ht, wd]
    Tensor intrn, // [4] 
    Tensor ii, // [E]
    Tensor jj, // [E]
    Tensor target, // [E, ht, wd, 2]
    Tensor weight, // [E, ht, wd, 2]
    bool ret_cross12
){

    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(disps);
    CHECK_INPUT(intrn);
    CHECK_INPUT(target);
    CHECK_INPUT(weight);
    CHECK_INPUTL(ii);
    CHECK_INPUTL(jj);

    const int E = ii.size(0);
    const int H = disps.size(1);
    const int W = disps.size(2);

    auto opts = disps.options();

    Tensor Hii = torch::zeros({E, 6, 6}, opts);
    Tensor Hij = torch::zeros({E, 6, 6}, opts);
    Tensor Hji = torch::zeros({E, 6, 6}, opts);
    Tensor Hjj = torch::zeros({E, 6, 6}, opts);

    Tensor vi = torch::zeros({E, 6}, opts);
    Tensor vj = torch::zeros({E, 6}, opts);

    Tensor depth_diag = torch::zeros({E, H*W}, opts);
    Tensor depth_residual = torch::zeros({E, H*W}, opts);

    if (ret_cross12){
        Tensor Ei = torch::zeros({E, 6, H*W}, opts);
        Tensor Ej = torch::zeros({E, 6, H*W}, opts);
        fused_project_and_accumulate_kernel<true><<<E, THREADS>>>(
            t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            intrn.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            target.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            weight.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            Hii.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hij.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hji.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hjj.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            vi.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            vj.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            Ei.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ej.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            depth_diag.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_residual.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
        );

        return {Hii, Hij, Hji, Hjj, vi, vj, Ei, Ej, depth_diag, depth_residual};

    } else {
        Tensor Ei = torch::empty({E, 6, H*W}, opts);
        Tensor Ej = torch::empty({E, 6, H*W}, opts);
        fused_project_and_accumulate_kernel<false><<<E, THREADS>>>(
            t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            q.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            disps.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            intrn.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            ii.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            jj.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            target.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            weight.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            Hii.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hij.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hji.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hjj.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            vi.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            vj.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            Ei.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ej.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            depth_diag.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_residual.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
        );

        return {Hii, Hij, Hji, Hjj, vi, vj, depth_diag, depth_residual};
    }
}

std::vector<Tensor> scatter_pose_system_cuda(
    Tensor Hii, // [E, 6, 6]
    Tensor Hij, // [E, 6, 6]
    Tensor Hji, // [E, 6, 6]
    Tensor Hjj, // [E, 6, 6]
    Tensor vi, // [E, 6]
    Tensor vj, // [E, 6]
    Tensor Ei, // [E, 6, hw]
    Tensor Ej, // [E, 6, hw]
    Tensor Ck, // [E, hw]
    Tensor wk, // [E, hw]
    Tensor source_indices, // [E]
    Tensor target_indices, // [E]
    Tensor edge_to_keyframe, // [E]
    Tensor keyframe_indices, // [M]
    Tensor damping, // [T, ht, wd]
    int num_opt_poses,
    int rig_size,
    int num_fixed_poses,
    bool ret_cross,
    bool ret_depth,
    int M, // num keyframes
    int ht, // height
    int wd, // width
    float ep, // epsilon damping for Hessian
    float lm // lambda (LM) damping for Hessian
) {
    const int E = Hii.size(0);
    const int manifold_dim = 6;
    const int hw = ht * wd;
    auto opts = Hii.options();

    CHECK_INPUT(Hii);
    CHECK_INPUT(Hij);
    CHECK_INPUT(Hji);
    CHECK_INPUT(Hjj);
    CHECK_INPUT(vi);
    CHECK_INPUT(vj);
    CHECK_INPUT(Ei);
    CHECK_INPUT(Ej);
    CHECK_INPUT(Ck);
    CHECK_INPUT(wk);
    CHECK_INPUTL(source_indices);
    CHECK_INPUTL(target_indices);
    CHECK_INPUTL(edge_to_keyframe);
    CHECK_INPUTL(keyframe_indices);
    CHECK_INPUT(damping);
    
    const int threads = 64;
    
    Tensor hessian = torch::zeros({num_opt_poses * num_opt_poses, manifold_dim, manifold_dim}, opts);
    Tensor gradient = torch::zeros({num_opt_poses, manifold_dim}, opts);
    
    Tensor cross_term = ret_cross ? torch::zeros({num_opt_poses * M, manifold_dim, hw}, opts)
                                  : torch::empty({1, 1, 1}, opts);
    
    Tensor depth_diag = ret_depth ? torch::zeros({M, hw}, opts) : torch::empty({1, 1}, opts);
    Tensor depth_gradient = ret_depth ? torch::zeros({M, hw}, opts) : torch::empty({1, 1}, opts);
    
    if (ret_cross && ret_depth) {
        scatter_pose_system_kernel<true, true><<<E, threads>>>(
            Hii.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hij.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hji.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hjj.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            vi.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            vj.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            Ei.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ej.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ck.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            wk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            source_indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            target_indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            edge_to_keyframe.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            keyframe_indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            damping.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            hessian.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cross_term.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            depth_diag.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            num_opt_poses,
            rig_size,
            num_fixed_poses,
            M,
            hw
        );
    } else if (ret_cross && !ret_depth) {
        scatter_pose_system_kernel<true, false><<<E, threads>>>(
            Hii.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hij.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hji.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hjj.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            vi.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            vj.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            Ei.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ej.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ck.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            wk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            source_indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            target_indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            edge_to_keyframe.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            keyframe_indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            damping.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            hessian.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cross_term.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            depth_diag.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            num_opt_poses,
            rig_size,
            num_fixed_poses,
            M,
            hw
        );
    } else if (!ret_cross && ret_depth) {
        scatter_pose_system_kernel<false, true><<<E, threads>>>(
            Hii.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hij.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hji.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hjj.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            vi.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            vj.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            Ei.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ej.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ck.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            wk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            source_indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            target_indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            edge_to_keyframe.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            keyframe_indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            damping.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            hessian.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cross_term.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            depth_diag.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            num_opt_poses,
            rig_size,
            num_fixed_poses,
            M,
            hw
        );
    } else {
        scatter_pose_system_kernel<false, false><<<E, threads>>>(
            Hii.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hij.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hji.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Hjj.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            vi.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            vj.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            Ei.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ej.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            Ck.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            wk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            source_indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            target_indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            edge_to_keyframe.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            keyframe_indices.packed_accessor32<long, 1, torch::RestrictPtrTraits>(),
            damping.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            hessian.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            cross_term.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            depth_diag.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            depth_gradient.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            num_opt_poses,
            rig_size,
            num_fixed_poses,
            M,
            hw
        );
    }
    
    hessian = hessian.view({num_opt_poses, num_opt_poses, manifold_dim, manifold_dim});
    
    if (ep > 0.0f || lm > 0.0f) {
        apply_hessian_damping_kernel<<<num_opt_poses, manifold_dim>>>(
            hessian.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            ep,
            lm
        );
    }
    
    if (ret_cross) {
        cross_term = cross_term.view({num_opt_poses, M, manifold_dim, hw});
    }
    
    if (ret_cross && ret_depth) {
        return {hessian, gradient, cross_term, depth_diag, depth_gradient};
    } else if (ret_cross) {
        return {hessian, gradient, cross_term};
    } else if (ret_depth) {
        return {hessian, gradient, depth_diag, depth_gradient};
    } else {
        return {hessian, gradient};
    }
}