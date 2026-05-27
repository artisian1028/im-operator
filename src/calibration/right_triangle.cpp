#include "common.hpp"
#include "calibration/algorithms.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace calibration {

namespace {

// --- Triangulate a 3D point from 2+ calibrated camera views ---
//
// Uses DLT (Direct Linear Transform): for each camera, the cross-product
// constraint p × (K*(R*P + t)) = 0 gives 2 linear equations in P.
// Stack into a 2C × 4 matrix, solve via least squares (using normal equations).

bool triangulate_point(const Point2D* observations,
                        const CameraIntrinsics* K_arr,
                        const CameraExtrinsics* extr_arr,
                        int camera_count,
                        double P[3]) {
    if (camera_count < 2) return false;

    // Compute camera centers and ray directions in world frame (camera 0's frame)
    std::vector<double> centers(3 * camera_count);
    std::vector<double> directions(3 * camera_count);

    for (int c = 0; c < camera_count; ++c) {
        const CameraIntrinsics& K = K_arr[c];
        const CameraExtrinsics& ext = extr_arr[c];

        // Pixel to direction in camera frame (NOT normalized, z=1)
        double xn = (observations[c].x - K.cx) / K.fx;
        double yn = (observations[c].y - K.cy) / K.fy;
        double dir_cam[3] = {xn, yn, 1.0};

        // Transform to world: P_cam = R * P_world + t
        // Camera center: C = -R^T * t
        // Direction:  d = R^T * dir_cam
        double Rt[9];
        detail::mat3x3_transpose(ext.rotation, Rt);

        double t_neg[3] = {-ext.translation[0], -ext.translation[1], -ext.translation[2]};
        detail::mat3x3_vec3_mult(Rt, t_neg, centers.data() + 3 * c);

        detail::mat3x3_vec3_mult(Rt, dir_cam, directions.data() + 3 * c);
        // Do NOT normalize — keep z=1 so that parameter s equals true depth
    }

    // Pair-wise ray intersection using unnormalized vectors
    P[0] = P[1] = P[2] = 0.0;
    int pair_count = 0;

    for (int i = 0; i < camera_count; ++i) {
        for (int j = i + 1; j < camera_count; ++j) {
            double* Ci = centers.data() + 3 * i;
            double* Cj = centers.data() + 3 * j;
            double* vi = directions.data() + 3 * i;
            double* vj = directions.data() + 3 * j;

            // Closest point between rays: P = Ci + s * vi
            // s = ((Cj - Ci) × vj) · (vi × vj) / |vi × vj|²
            double w[3] = {Cj[0] - Ci[0], Cj[1] - Ci[1], Cj[2] - Ci[2]};
            double cross_w_vj[3], cross_vi_vj[3];
            detail::cross3(w, vj, cross_w_vj);
            detail::cross3(vi, vj, cross_vi_vj);

            double numer = detail::dot3(cross_w_vj, cross_vi_vj);
            double denom = detail::dot3(cross_vi_vj, cross_vi_vj);

            if (denom < 1e-12) continue;  // Parallel rays

            double s = numer / denom;

            double Pi[3] = {Ci[0] + s * vi[0], Ci[1] + s * vi[1], Ci[2] + s * vi[2]};

            P[0] += Pi[0];
            P[1] += Pi[1];
            P[2] += Pi[2];
            ++pair_count;
        }
    }

    if (pair_count == 0) return false;

    P[0] /= pair_count;
    P[1] /= pair_count;
    P[2] /= pair_count;
    return true;
}

// --- Orthogonal Procrustes: find (R, t) that maps "from" points to "to" points ---
// from[i] and to[i] are 3D points, N = number of points (at least 3)

void procrustes_fit(const double* from, const double* to, int N,
                     double R[9], double t[3]) {
    // Centroids
    double cf[3] = {0, 0, 0}, ct[3] = {0, 0, 0};
    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < 3; ++d) {
            cf[d] += from[3 * i + d];
            ct[d] += to[3 * i + d];
        }
    }
    for (int d = 0; d < 3; ++d) {
        cf[d] /= N;
        ct[d] /= N;
    }

    // Cross-covariance matrix H = Σ (from_i')(to_i')^T
    double H[9] = {0};
    for (int i = 0; i < N; ++i) {
        double a[3] = {from[3*i+0] - cf[0], from[3*i+1] - cf[1], from[3*i+2] - cf[2]};
        double b[3] = {to[3*i+0] - ct[0],   to[3*i+1] - ct[1],   to[3*i+2] - ct[2]};
        for (int r = 0; r < 3; ++r)
            for (int s = 0; s < 3; ++s)
                H[3 * r + s] += a[r] * b[s];
    }

    // SVD of H
    double S[3], V[9];
    double H_copy[9];
    for (int k = 0; k < 9; ++k) H_copy[k] = H[k];
    detail::svd_3x3(H_copy, S, V);

    // H_copy now contains U. Check for rank deficiency.
    if (S[2] < 1e-8 * S[0]) {
        // Rank-2: points are coplanar with the world z=0 plane.
        // Reconstruct rotation from the two valid singular vectors.
        // R = V * U^T, but U's third column is degenerate.
        // U has columns: [H_copy[0],H_copy[3],H_copy[6]], [H_copy[1],H_copy[4],H_copy[7]], [0,0,0]
        // R_col_j = V * (j-th row of U)^T = V * [U_j0, U_j1, U_j2]^T
        double col0[3], col1[3];
        for (int i = 0; i < 3; ++i) {
            // R_col_0 = V * U_row_0^T = V * [U[0],U[1],U[2]]^T
            col0[i] = V[i*3+0]*H_copy[0] + V[i*3+1]*H_copy[1] + V[i*3+2]*H_copy[2];
            // R_col_1 = V * U_row_1^T = V * [U[3],U[4],U[5]]^T
            col1[i] = V[i*3+0]*H_copy[3] + V[i*3+1]*H_copy[4] + V[i*3+2]*H_copy[5];
        }
        detail::normalize3(col0);
        detail::normalize3(col1);

        // R column 0 = col0, column 1 = col1, column 2 = cross(col0, col1)
        R[0] = col0[0]; R[1] = col1[0]; R[2] = 0.0;
        R[3] = col0[1]; R[4] = col1[1]; R[5] = 0.0;
        R[6] = col0[2]; R[7] = col1[2]; R[8] = 0.0;

        // Column 2 = cross(col0, col1), then re-orthogonalize col1
        double col2[3];
        detail::cross3(col0, col1, col2);
        detail::normalize3(col2);
        detail::cross3(col2, col0, col1);
        detail::normalize3(col1);

        R[2] = col2[0]; R[5] = col2[1]; R[8] = col2[2];
        R[1] = col1[0]; R[4] = col1[1]; R[7] = col1[2];
    } else {
        // Full rank: standard R = V * U^T
        double Ut[9];
        detail::mat3x3_transpose(H_copy, Ut);
        detail::mat3x3_mult(V, Ut, R);

        double det = R[0]*(R[4]*R[8]-R[5]*R[7])
                   - R[1]*(R[3]*R[8]-R[5]*R[6])
                   + R[2]*(R[3]*R[7]-R[4]*R[6]);
        if (det < 0.0) {
            R[2] = -R[2]; R[5] = -R[5]; R[8] = -R[8];
        }
    }

    // Translation: t = ct - R * cf
    double Rcf[3];
    detail::mat3x3_vec3_mult(R, cf, Rcf);
    t[0] = ct[0] - Rcf[0];
    t[1] = ct[1] - Rcf[1];
    t[2] = ct[2] - Rcf[2];
}

}  // anonymous namespace

// ============================================================
//  Public API
// ============================================================

CalibrationError process_right_triangle(const CameraObservations* cameras,
                                         int camera_count,
                                         const CameraCalibration* calibrations,
                                         const TriangleConfig* config,
                                         WorldRegistration* world_reg) {
    if (!cameras || !calibrations || !config || !world_reg) return CalibrationError::NullInput;
    if (camera_count < 2) return CalibrationError::InvalidCameraCount;
    if (config->image_width <= 0 || config->image_height <= 0) return CalibrationError::InvalidFrameCount;
    if (config->ox_length <= 0.0 || config->oy_length <= 0.0) return CalibrationError::InvalidFrameCount;

    // Verify each camera has 1 frame of observations
    for (int c = 0; c < camera_count; ++c) {
        if (!cameras[c].frames || cameras[c].frame_count < 1) {
            return CalibrationError::InvalidFrameCount;
        }
    }

    // Collect 2D observations for each marker
    std::vector<Point2D> obs_o(camera_count);
    std::vector<Point2D> obs_x(camera_count);
    std::vector<Point2D> obs_y(camera_count);

    for (int c = 0; c < camera_count; ++c) {
        const RodObservation& rod = cameras[c].frames[0];
        obs_o[c] = rod.marker_a;  // O = marker A
        obs_x[c] = rod.marker_b;  // X = marker B
        obs_y[c] = rod.marker_c;  // Y = marker C
    }

    // Extract per-camera intrinsics and extrinsics
    std::vector<CameraIntrinsics> K_arr(camera_count);
    std::vector<CameraExtrinsics> extr_arr(camera_count);
    for (int c = 0; c < camera_count; ++c) {
        K_arr[c] = calibrations[c].intrinsics;
        extr_arr[c] = calibrations[c].extrinsics;
    }

    // --- Triangulate each marker in the common frame (camera 0's frame) ---
    double P_o[3], P_x[3], P_y[3];
    if (!triangulate_point(obs_o.data(), K_arr.data(), extr_arr.data(), camera_count, P_o)) {
        return CalibrationError::SingularMatrix;
    }
    if (!triangulate_point(obs_x.data(), K_arr.data(), extr_arr.data(), camera_count, P_x)) {
        return CalibrationError::SingularMatrix;
    }
    if (!triangulate_point(obs_y.data(), K_arr.data(), extr_arr.data(), camera_count, P_y)) {
        return CalibrationError::SingularMatrix;
    }

    // --- Fit world-to-common rigid transform ---
    // Known world coordinates: O=(0,0,0), X=(ox,0,0), Y=(0,oy,0)
    double world_pts[9] = {
        0.0, 0.0, 0.0,                                // O
        config->ox_length, 0.0, 0.0,                  // X
        0.0, config->oy_length, 0.0                   // Y
    };
    double common_pts[9] = {
        P_o[0], P_o[1], P_o[2],
        P_x[0], P_x[1], P_x[2],
        P_y[0], P_y[1], P_y[2]
    };

    // Procrustes: find (R, t) s.t. world_pt ≈ R * common_pt + t
    // This gives the common-to-world transform.
    double R_c2w[9], t_c2w[3];
    procrustes_fit(common_pts, world_pts, 3, R_c2w, t_c2w);

    // Convert to world-to-camera: world_pt = R_c2w * common_pt + t_c2w
    // => common_pt = R_c2w^T * world_pt - R_c2w^T * t_c2w
    // => R_w2c = R_c2w^T, t_w2c = -R_c2w^T * t_c2w
    detail::mat3x3_transpose(R_c2w, world_reg->world_to_camera.rotation);

    double neg_t[3] = {-t_c2w[0], -t_c2w[1], -t_c2w[2]};
    detail::mat3x3_vec3_mult(world_reg->world_to_camera.rotation, neg_t,
                              world_reg->world_to_camera.translation);

    // Compute fit error (RMS distance in world frame)
    double fit_err = 0.0;
    for (int i = 0; i < 3; ++i) {
        // Transform common point to world
        double transformed[3];
        detail::mat3x3_vec3_mult(R_c2w, common_pts + 3*i, transformed);
        transformed[0] += t_c2w[0];
        transformed[1] += t_c2w[1];
        transformed[2] += t_c2w[2];

        double dx = transformed[0] - world_pts[3*i+0];
        double dy = transformed[1] - world_pts[3*i+1];
        double dz = transformed[2] - world_pts[3*i+2];
        fit_err += dx*dx + dy*dy + dz*dz;
    }
    world_reg->fit_error = std::sqrt(fit_err / 3.0);

    return CalibrationError::Ok;
}

}  // namespace calibration
