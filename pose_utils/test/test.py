import torch
import lietorch
import time
from lie_ops_cuda import (
    quat_rotate_cuda,
    quat_multiply_cuda,
    quat_to_matrix_cuda,
    matrix_to_quat_cuda,
    se3_exp_cuda,
    se3_log_cuda,
    se3_point_jac_cuda,
)


def test_functions():
    device = "cuda"
    torch.manual_seed(42)

    print("Lie Group CUDA vs lietorch")

    B = 4096
    q1 = torch.randn(B, 4, device=device)
    q1 /= q1.norm(dim=-1, keepdim=True)
    q2 = torch.randn(B, 4, device=device)
    q2 /= q2.norm(dim=-1, keepdim=True)
    v = torch.randn(B, 3, device=device)
    xi = torch.randn(B, 6, device=device) * 0.1
    pts = torch.randn(B, 4, device=device)

    def check(name, ours, reference, atol=1e-5):
        err = (ours - reference).abs().max().item()
        status = "PASS" if err < atol else "FAIL"
        print(f"{status} {name:<20} max_err: {err:.2e}")
        return err < atol

    all_passed = True

    print("\nQuaternion Operations")

    def quat_mul_ref(q1, q2):
        x1, y1, z1, w1 = q1.unbind(-1)
        x2, y2, z2, w2 = q2.unbind(-1)
        return torch.stack(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dim=-1,
        )

    all_passed &= check(
        "quat_multiply", quat_multiply_cuda(q1, q2), quat_mul_ref(q1, q2)
    )

    def quat_rotate_ref(q, v):
        u, s = q[..., :3], q[..., 3:]
        return v + 2 * torch.cross(u, torch.cross(u, v, dim=-1) + s * v, dim=-1)

    all_passed &= check("quat_rotate", quat_rotate_cuda(q1, v), quat_rotate_ref(q1, v))

    R = quat_to_matrix_cuda(q1)
    q_rt = matrix_to_quat_cuda(R)
    R_rt = quat_to_matrix_cuda(q_rt)
    all_passed &= check("quat_matrix_rt", R, R_rt)

    print("\nSE3 Operations")

    def time_op(fn, *args, n=100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n):
            result = fn(*args)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / n * 1000, result

    g_lie = lietorch.SE3.exp(xi)
    T_lie = g_lie.matrix()
    t_lie, R_lie = T_lie[..., :3, 3], T_lie[..., :3, :3]

    t_ours, q_ours = se3_exp_cuda(xi[..., :3].contiguous(), xi[..., 3:].contiguous())
    R_ours = quat_to_matrix_cuda(q_ours)

    all_passed &= check("se3_exp_trans", t_ours, t_lie, atol=2e-6)
    all_passed &= check("se3_exp_rot", R_ours, R_lie, atol=1e-5)

    t_lie_time, _ = time_op(lambda: lietorch.SE3.exp(xi))
    t_ours_time, _ = time_op(
        lambda: se3_exp_cuda(xi[..., :3].contiguous(), xi[..., 3:].contiguous())
    )
    print(
        f"\nse3_exp timing: ours={t_ours_time:.2f}ms lietorch={t_lie_time:.2f}ms speedup={t_lie_time/t_ours_time:.1f}x\n"
    )

    xi_log_lie = g_lie.log()
    rho_lie, phi_lie = xi_log_lie[..., :3], xi_log_lie[..., 3:]

    rho_ours, phi_ours = se3_log_cuda(
        t_lie.contiguous(), matrix_to_quat_cuda(R_lie.contiguous())
    )

    all_passed &= check("se3_log_rho", rho_ours, rho_lie, atol=6e-6)
    all_passed &= check("se3_log_phi", phi_ours, phi_lie, atol=1e-6)

    log_lie_time, _ = time_op(lambda: g_lie.log())
    log_ours_time, _ = time_op(
        lambda: se3_log_cuda(
            t_lie.contiguous(), matrix_to_quat_cuda(R_lie.contiguous())
        )
    )
    print(
        f"\nse3_log timing: ours={log_ours_time:.2f}ms lietorch={log_lie_time:.2f}ms speedup={log_lie_time/log_ours_time:.1f}x\n"
    )

    rho_rt, phi_rt = se3_log_cuda(t_ours, q_ours)
    all_passed &= check("se3_rt_rho", rho_rt, xi[..., :3], atol=1e-5)
    all_passed &= check("se3_rt_phi", phi_rt, xi[..., 3:], atol=1e-5)

    print("\nSE3 Point Jacobian")
    J = se3_point_jac_cuda(pts)
    expected_shape = (B, 4, 6)
    shape_ok = J.shape == expected_shape
    finite_ok = torch.isfinite(J).all()

    print(
        f"{'PASS' if shape_ok else 'FAIL'} se3_point_jac_shape expected: {expected_shape}, got: {J.shape}"
    )
    print(
        f"{'PASS' if finite_ok else 'FAIL'} se3_point_jac_finite all finite: {finite_ok}"
    )
    all_passed &= shape_ok and finite_ok

    if all_passed:
        print("\nOK")
    else:
        print("\nFAIL")

    return all_passed


if __name__ == "__main__":
    test_functions()
