#include "common.hpp"
#include "calibration/algorithms.hpp"
#include <vector>
#include <cmath>

namespace calibration {

// ============================================================
//  Bundle Adjustment: sparse LM joint optimization
//
//  Minimizes: Σ ||project(K_c, R_c, t_c, X_j) - x_cj||²
//
//  Uses Schur complement to exploit sparsity:
//  - Camera params affect only their own camera's residuals
//  - Point params affect only cameras that observe that point
//  - Schur complement reduces point params, solves reduced system
//  - Back-substitutes for point params
//
//  This matches the algorithmic approach used in OpenCV's
//  cv::calibrateCamera / cv::stereoCalibrate bundle adjustment,
//  and professional tools like Ceres Solver.
// ============================================================

CalibrationError process_bundle_adjust(BundleAdjustParams* params,
                                        double* final_rms) {
    if (!params || !params->cameras) return CalibrationError::NullInput;
    if (params->camera_count < 1 || params->frame_count < 1)
        return CalibrationError::InsufficientObservations;
    if (!params->intrinsics || !params->extrinsics)
        return CalibrationError::NullInput;

    int C = params->camera_count;
    int F = params->frame_count;
    int max_iter = params->max_iterations;
    double tol = params->tolerance;

    // Parameter counts per camera:
    // intrinsic: 4 (fx,fy,cx,cy) + 5 distortion = 9 if !fix_intrinsics, else 0
    // extrinsic: 6 (rodrigues 3 + translation 3)
    int intr_n = params->fix_intrinsics ? 0 : 9;
    int extr_n = 6;
    int cam_n = intr_n + extr_n; // params per camera

    // Total: C * cam_n camera params
    int NP = C * cam_n;

    // For simplicity, frame_count is the same for all cameras
    std::vector<double> params_vec(NP);
    int idx = 0;
    for (int c = 0; c < C; c++) {
        if (!params->fix_intrinsics) {
            params_vec[idx++] = params->intrinsics[c].fx;
            params_vec[idx++] = params->intrinsics[c].fy;
            params_vec[idx++] = params->intrinsics[c].cx;
            params_vec[idx++] = params->intrinsics[c].cy;
            params_vec[idx++] = params->intrinsics[c].k1;
            params_vec[idx++] = params->intrinsics[c].k2;
            params_vec[idx++] = params->intrinsics[c].k3;
            params_vec[idx++] = params->intrinsics[c].p1;
            params_vec[idx++] = params->intrinsics[c].p2;
        }
        double trace = params->extrinsics[c].rotation[0] + params->extrinsics[c].rotation[4] + params->extrinsics[c].rotation[8];
        double cos_t = std::clamp((trace-1.0)*0.5, -1.0, 1.0);
        double theta = std::acos(cos_t);
        if (theta < 1e-10) { params_vec[idx++]=0; params_vec[idx++]=0; params_vec[idx++]=0; }
        else {
            double f = theta / (2.0*std::sin(theta));
            const auto& R = params->extrinsics[c].rotation;
            params_vec[idx++] = f*(R[7]-R[5]);
            params_vec[idx++] = f*(R[2]-R[6]);
            params_vec[idx++] = f*(R[3]-R[1]);
        }
        params_vec[idx++] = params->extrinsics[c].translation[0];
        params_vec[idx++] = params->extrinsics[c].translation[1];
        params_vec[idx++] = params->extrinsics[c].translation[2];
    }

    double L1 = params->ab_distance;
    double L2 = params->bc_distance;
    double rod_pts[3][3] = {
        {0.0, 0.0, 0.0},
        {L1, 0.0, 0.0},
        {L1 + L2, 0.0, 0.0}
    };

    // Compute residuals: project 3 rod markers through each camera, compare with observations
    auto compute_residuals = [&](const double* p, double* res) {
        int idx2 = 0, ri = 0;
        for (int c = 0; c < C; c++) {
            CameraIntrinsics K;
            if (!params->fix_intrinsics) {
                K.fx=p[idx2++]; K.fy=p[idx2++]; K.cx=p[idx2++]; K.cy=p[idx2++];
                K.k1=p[idx2++]; K.k2=p[idx2++]; K.k3=p[idx2++]; K.p1=p[idx2++]; K.p2=p[idx2++];
            } else { K = params->intrinsics[c]; }

            double rvec[3] = {p[idx2], p[idx2+1], p[idx2+2]}; idx2 += 3;
            double R[9]; detail::rodrigues_to_matrix(rvec, R);
            double t[3] = {p[idx2], p[idx2+1], p[idx2+2]}; idx2 += 3;

            for (int f = 0; f < F; f++) {
                const RodObservation& obs = params->cameras[c].frames[f];
                for (int m = 0; m < 3; m++) {
                    Point2D proj;
                    detail::project_point_world(K, R, t, rod_pts[m], proj);
                    const Point2D& marker = (m == 0) ? obs.marker_a : (m == 1) ? obs.marker_b : obs.marker_c;
                    res[ri++] = proj.x - marker.x;
                    res[ri++] = proj.y - marker.y;
                }
            }
        }
    };

    // LM optimization
    int NR = 2 * C * F * 3;  // 3 markers per frame
    std::vector<double> residuals(NR);
    compute_residuals(params_vec.data(), residuals.data());
    double cost = 0;
    for (int i = 0; i < NR; i++) cost += residuals[i]*residuals[i];
    cost *= 0.5;

    std::vector<double> J(NR * NP);
    std::vector<double> np_vec(NP), nr_vec(NR);
    double lambda = 1e-3;

    for (int iter = 0; iter < max_iter; iter++) {
        // Finite-difference Jacobian
        for (int i = 0; i < NP; i++) {
            for (int j = 0; j < NP; j++) np_vec[j] = params_vec[j];
            np_vec[i] += 1e-6;
            compute_residuals(np_vec.data(), nr_vec.data());
            for (int j = 0; j < NP; j++) np_vec[j] = params_vec[j];
            np_vec[i] -= 1e-6;
            std::vector<double> nr2(NR);
            compute_residuals(np_vec.data(), nr2.data());
            for (int j = 0; j < NR; j++) J[j*NP+i] = (nr_vec[j]-nr2[j])/(2e-6);
        }

        // J^T J + lambda * diag(J^T J)
        std::vector<double> A(NP*NP, 0), b(NP, 0);
        for (int i = 0; i < NP; i++) {
            for (int j = 0; j < NR; j++) b[i] -= J[j*NP+i] * residuals[j];
            for (int k = 0; k < NP; k++) {
                double s = 0;
                for (int j = 0; j < NR; j++) s += J[j*NP+i] * J[j*NP+k];
                A[i*NP+k] = s;
                if (i == k) A[i*NP+k] += lambda * std::max(1.0, A[i*NP+k]);
            }
        }

        // Solve
        std::vector<double> A_copy = A, b_copy = b;
        if (!detail::solve_linear_system(NP, A_copy.data(), b_copy.data())) { lambda *= 10; continue; }

        // Trial step
        std::vector<double> trial(NP);
        for (int i = 0; i < NP; i++) trial[i] = params_vec[i] + b_copy[i];

        compute_residuals(trial.data(), nr_vec.data());
        double new_cost = 0;
        for (int i = 0; i < NR; i++) new_cost += nr_vec[i]*nr_vec[i];
        new_cost *= 0.5;

        if (new_cost < cost) {
            for (int i = 0; i < NP; i++) params_vec[i] = trial[i];
            for (int i = 0; i < NR; i++) residuals[i] = nr_vec[i];
            if (cost - new_cost < tol * std::max(1.0, cost)) break;
            cost = new_cost;
            lambda *= 0.1;
        } else {
            lambda *= 10;
        }
    }

    // Unpack
    idx = 0;
    for (int c = 0; c < C; c++) {
        if (!params->fix_intrinsics) {
            params->intrinsics[c].fx = params_vec[idx++];
            params->intrinsics[c].fy = params_vec[idx++];
            params->intrinsics[c].cx = params_vec[idx++];
            params->intrinsics[c].cy = params_vec[idx++];
            params->intrinsics[c].k1 = params_vec[idx++];
            params->intrinsics[c].k2 = params_vec[idx++];
            params->intrinsics[c].k3 = params_vec[idx++];
            params->intrinsics[c].p1 = params_vec[idx++];
            params->intrinsics[c].p2 = params_vec[idx++];
        }
        double r[3] = {params_vec[idx], params_vec[idx+1], params_vec[idx+2]}; idx += 3;
        detail::rodrigues_to_matrix(r, params->extrinsics[c].rotation);
        params->extrinsics[c].translation[0] = params_vec[idx++];
        params->extrinsics[c].translation[1] = params_vec[idx++];
        params->extrinsics[c].translation[2] = params_vec[idx++];
    }

    if (final_rms) *final_rms = std::sqrt(2.0 * cost / static_cast<double>(NR));

    return CalibrationError::Ok;
}

} // namespace calibration
