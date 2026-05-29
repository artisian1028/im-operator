#include "common.hpp"
#include "calibration/algorithms.hpp"
#include <vector>
#include <cmath>

namespace calibration {

// ============================================================
//  PnP Solver: Perspective-n-Point
//
//  Given N 3D world points, their 2D image projections, and
//  camera intrinsics K, solve for camera pose (R, t).
//
//  Algorithm:
//  1. DLT for initial pose (using all points)
//  2. Gauss-Newton refinement minimizing reprojection error
//
//  This enables incremental SfM: register new cameras by
//  matching their observations to existing 3D points.
// ============================================================

namespace {

// Analytical residual and Jacobian for one point
void project_residual(const CameraIntrinsics& K, const double R[9], const double t[3],
                      const double X[3], const Point2D& obs,
                      double& rx, double& ry,
                      double J_rt[2][6], double J_X[2][3]) {
    // Transform to camera coordinates: Pc = R*X + t
    double Pc[3];
    detail::mat3x3_vec3_mult(R, X, Pc);
    Pc[0] += t[0]; Pc[1] += t[1]; Pc[2] += t[2];

    double inv_z = 1.0 / Pc[2];
    double inv_z2 = inv_z * inv_z;

    // Projection
    double u = K.fx * Pc[0] * inv_z + K.cx;
    double v = K.fy * Pc[1] * inv_z + K.cy;

    // Residuals
    rx = u - obs.x;
    ry = v - obs.y;

    // Jacobian of (u,v) w.r.t. Pc
    // du/dPc = [fx/Z, 0, -fx*X/Z²]
    // dv/dPc = [0, fy/Z, -fy*Y/Z²]
    double du[3] = {K.fx * inv_z, 0, -K.fx * Pc[0] * inv_z2};
    double dv[3] = {0, K.fy * inv_z, -K.fy * Pc[1] * inv_z2};

    // Jacobian of Pc w.r.t. rotation (small-angle approximation):
    // dPc/dw = skew(-R*X) ≈ skew(-Pc + t)
    double mx[3] = {0, Pc[2]-t[2], -(Pc[1]-t[1])};
    double my[3] = {-(Pc[2]-t[2]), 0, Pc[0]-t[0]};
    double mz[3] = {Pc[1]-t[1], -(Pc[0]-t[0]), 0};

    // dPc/dt = I

    // J_rt: [2 x 6] derivatives of (u,v) w.r.t. (wx,wy,wz,tx,ty,tz)
    J_rt[0][0] = du[0]*mx[0] + du[1]*my[0] + du[2]*mz[0];
    J_rt[0][1] = du[0]*mx[1] + du[1]*my[1] + du[2]*mz[1];
    J_rt[0][2] = du[0]*mx[2] + du[1]*my[2] + du[2]*mz[2];
    J_rt[0][3] = du[0]; J_rt[0][4] = du[1]; J_rt[0][5] = du[2];

    J_rt[1][0] = dv[0]*mx[0] + dv[1]*my[0] + dv[2]*mz[0];
    J_rt[1][1] = dv[0]*mx[1] + dv[1]*my[1] + dv[2]*mz[1];
    J_rt[1][2] = dv[0]*mx[2] + dv[1]*my[2] + dv[2]*mz[2];
    J_rt[1][3] = dv[0]; J_rt[1][4] = dv[1]; J_rt[1][5] = dv[2];

    // J_X: [2 x 3]
    J_X[0][0] = du[0]*R[0] + du[1]*R[3] + du[2]*R[6];
    J_X[0][1] = du[0]*R[1] + du[1]*R[4] + du[2]*R[7];
    J_X[0][2] = du[0]*R[2] + du[1]*R[5] + du[2]*R[8];

    J_X[1][0] = dv[0]*R[0] + dv[1]*R[3] + dv[2]*R[6];
    J_X[1][1] = dv[0]*R[1] + dv[1]*R[4] + dv[2]*R[7];
    J_X[1][2] = dv[0]*R[2] + dv[1]*R[5] + dv[2]*R[8];
}

} // anonymous namespace

CalibrationError process_pnp_solver(const PnPParams* params, PnPResult* result) {
    if (!params || !result) return CalibrationError::NullInput;
    if (!params->image_pts || !params->world_pts || !params->intrinsics)
        return CalibrationError::NullInput;
    if (params->point_count < 4) return CalibrationError::InsufficientObservations;

    int n = params->point_count;
    const auto& K = *params->intrinsics;

    // Step 1: DLT initial pose
    DltCorrespondence* corrs = new DltCorrespondence[n];
    for (int i = 0; i < n; i++) {
        corrs[i].world_x = params->world_pts[i*3];
        corrs[i].world_y = params->world_pts[i*3+1];
        corrs[i].world_z = params->world_pts[i*3+2];
        corrs[i].image_x = params->image_pts[i].x;
        corrs[i].image_y = params->image_pts[i].y;
    }
    DltParams dp = {corrs, n};
    DltResult dr;
    CalibrationError dlt_err = process_dlt(&dp, &dr);
    delete[] corrs;
    if (dlt_err != CalibrationError::Ok) return dlt_err;

    // Use decomposed K from DLT as initial R, t
    double R[9], t[3];
    for (int i = 0; i < 9; i++) R[i] = dr.extrinsics.rotation[i];
    t[0] = dr.extrinsics.translation[0];
    t[1] = dr.extrinsics.translation[1];
    t[2] = dr.extrinsics.translation[2];

    // Step 2: Gauss-Newton refinement
    for (int iter = 0; iter < 20; iter++) {
        double JtJ[36] = {0}, Jtr[6] = {0};
        double total_err = 0;

        for (int i = 0; i < n; i++) {
            double rx, ry, J_rt[2][6], J_X[2][3];
            const double* X = params->world_pts + i*3;
            project_residual(K, R, t, X, params->image_pts[i], rx, ry, J_rt, J_X);

            total_err += rx*rx + ry*ry;

            // J^T J and J^T r accumulation
            for (int a = 0; a < 6; a++) {
                Jtr[a] += J_rt[0][a] * rx + J_rt[1][a] * ry;
                for (int b = 0; b < 6; b++) {
                    JtJ[a*6+b] += J_rt[0][a]*J_rt[0][b] + J_rt[1][a]*J_rt[1][b];
                }
            }
        }

        // Regularize
        for (int a = 0; a < 6; a++) JtJ[a*6+a] += 1e-8;

        // Solve delta
        std::vector<double> A(36), b(6);
        for (int i = 0; i < 36; i++) A[i] = JtJ[i];
        for (int i = 0; i < 6; i++) b[i] = -Jtr[i];

        if (!detail::solve_linear_system(6, A.data(), b.data())) break;

        // Update R, t
        double w[3] = {b[0], b[1], b[2]};
        double dR[9];
        detail::rodrigues_to_matrix(w, dR);

        double Rnew[9];
        detail::mat3x3_mult(dR, R, Rnew);
        for (int i = 0; i < 9; i++) R[i] = Rnew[i];
        t[0] += b[3]; t[1] += b[4]; t[2] += b[5];

        double step = b[0]*b[0] + b[1]*b[1] + b[2]*b[2] + b[3]*b[3] + b[4]*b[4] + b[5]*b[5];
        if (step < 1e-10) break;
    }

    // Output
    for (int i = 0; i < 9; i++) result->pose.rotation[i] = R[i];
    result->pose.translation[0] = t[0];
    result->pose.translation[1] = t[1];
    result->pose.translation[2] = t[2];
    result->inliers = n;

    // Final RMS
    double total_err = 0;
    for (int i = 0; i < n; i++) {
        double X[3] = {params->world_pts[i*3], params->world_pts[i*3+1], params->world_pts[i*3+2]};
        Point2D proj;
        detail::project_point_world(K, R, t, X, proj);
        double dx = proj.x - params->image_pts[i].x;
        double dy = proj.y - params->image_pts[i].y;
        total_err += dx*dx + dy*dy;
    }
    result->rms_error = std::sqrt(total_err / n);

    return CalibrationError::Ok;
}

} // namespace calibration
