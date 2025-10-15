#include "kernels.h"
#include "kernels/quat.cuh"
#include "kernels/se3.cuh"
#include "kernels/jacobian.cuh"
#include "kernels/transform.cuh"

Tensor quat_multiply_cuda(Tensor &q1, Tensor &q2) {
    CHECK_INPUT(q1);
    CHECK_INPUT(q2);

    const int batch_size = q1.size(0);
    auto out = torch::zeros_like(q1);

    const int blocks = (batch_size + THREADS - 1) / THREADS;

    quat_multiply_kernel<<<blocks, THREADS>>>(
        q1.data_ptr<float>(),
        q2.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size
    );

    return out;
}

Tensor quat_rotate_cuda(Tensor &q, Tensor &points) {
    CHECK_INPUT(q);
    CHECK_INPUT(points);

    const int batch_size = q.size(0);
    auto out = torch::zeros_like(points);

    const int blocks = (batch_size + THREADS - 1) / THREADS;

    quat_rotate_kernel<<<blocks, THREADS>>>(
        q.data_ptr<float>(),
        points.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size
    );

    return out;
}

Tensor quat_to_matrix_cuda(Tensor &q) {
    CHECK_INPUT(q);

    const int batch_size = q.size(0);
    auto result = torch::zeros({batch_size, 3, 3}, q.options());

    const int blocks = (batch_size + THREADS - 1) / THREADS;

    quat_to_matrix_kernel<<<blocks, THREADS>>>(
        q.data_ptr<float>(),
        result.data_ptr<float>(),
        batch_size
    );

    return result;
}

Tensor matrix_to_quat_cuda(Tensor &R) {
    CHECK_INPUT(R);

    const int batch_size = R.size(0);
    auto result = torch::zeros({batch_size, 4}, R.options());

    const int blocks = (batch_size + THREADS - 1) / THREADS;

    matrix_to_quat_kernel<<<blocks, THREADS>>>(
        R.data_ptr<float>(),
        result.data_ptr<float>(),
        batch_size
    );

    return result;
}

std::tuple<Tensor, Tensor> se3_exp_cuda(Tensor &rho, Tensor &phi) {
    CHECK_INPUT(rho);
    CHECK_INPUT(phi);

    const int batch_size = rho.size(0);

    auto q = torch::zeros({batch_size, 4}, rho.options());
    auto t = torch::zeros({batch_size, 3}, rho.options());

    const int blocks = (batch_size + THREADS - 1) / THREADS;

    se3_exp_kernel<<<blocks, THREADS>>>(
        rho.data_ptr<float>(),
        phi.data_ptr<float>(),
        q.data_ptr<float>(),
        t.data_ptr<float>(),
        batch_size
    );

    return {t, q};
}

std::tuple<Tensor, Tensor> se3_log_cuda(Tensor &t, Tensor &q) {
    CHECK_INPUT(q);
    CHECK_INPUT(t);

    const int batch_size = q.size(0);

    auto rho = torch::zeros({batch_size, 3}, q.options());
    auto phi = torch::zeros({batch_size, 3}, q.options());

    const int blocks = (batch_size + THREADS - 1) / THREADS;

    se3_log_kernel<<<blocks, THREADS>>>(
        q.data_ptr<float>(),
        t.data_ptr<float>(),
        rho.data_ptr<float>(),
        phi.data_ptr<float>(),
        batch_size
    );

    return {rho, phi};
}

Tensor se3_point_jac_cuda(Tensor &p) {
    CHECK_INPUT(p);

    const int n = p.size(0);
    auto jacobian = torch::zeros({n, 4, 6}, p.options());

    const int blocks = (n + THREADS - 1) / THREADS;

    se3_point_jac_kernel<<<blocks, THREADS>>>(
        p.data_ptr<float>(),
        jacobian.data_ptr<float>(),
        n
    );

    return jacobian;
}

Tensor se3_adjointT_cuda(Tensor &t, Tensor &q, Tensor &jac) {
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(jac);

    TORCH_CHECK(jac.size(-2) == 2 && jac.size(-1) == 6, "jac must be [...,2,6]");

    const int B = q.size(0);
    const int S = jac.numel() / (B * 2 * 6);

    auto out = torch::empty_like(jac);

    const int threads = 256;
    const int rows = B * S * 2;
    const int blocks = (rows + threads - 1) / threads;

    se3_adjointT_kernel<<<blocks, threads>>>(
        t.data_ptr<float>(),
        q.data_ptr<float>(),
        jac.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        S
    );

    return out;
}

Tensor se3_transform3d_cuda(Tensor &t, Tensor &q, Tensor &points) {
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(points);
    
    TORCH_CHECK(points.size(-1) == 3, "points must be [...,3]");

    const int B = t.size(0);
    const int S = points.numel() / (B * 3);

    auto out = torch::empty_like(points);

    const int blocks = (B * S + THREADS - 1) / THREADS;

    se3_transform_kernel<false><<<blocks, THREADS>>>(
        t.data_ptr<float>(),
        q.data_ptr<float>(),
        points.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        S
    );

    return out;
}

Tensor se3_transform4d_cuda(Tensor &t, Tensor &q, Tensor &points) {
    CHECK_INPUT(t);
    CHECK_INPUT(q);
    CHECK_INPUT(points);

    TORCH_CHECK(points.size(-1) == 4, "points must be [...,4]");

    const int B = t.size(0);
    const int S = points.numel() / (B * 4);

    auto out = torch::empty_like(points);

    const int blocks = (B * S + THREADS - 1) / THREADS;

    se3_transform_kernel<true><<<blocks, THREADS>>>(
        t.data_ptr<float>(),
        q.data_ptr<float>(),
        points.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        S
    );

    return out;
}
