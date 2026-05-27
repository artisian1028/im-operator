#include "common.hpp"
#include "calibration/algorithms.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace calibration {

namespace {

// --- Rod geometry ---

void build_rod_model(double L1, double L2, double rod_pts[3][3]) {
    // Rod local frame: A at origin, B and C along the x-axis
    rod_pts[0][0] = 0.0;      rod_pts[0][1] = 0.0;  rod_pts[0][2] = 0.0;
    rod_pts[1][0] = L1;       rod_pts[1][1] = 0.0;  rod_pts[1][2] = 0.0;
    rod_pts[2][0] = L1 + L2;  rod_pts[2][1] = 0.0;  rod_pts[2][2] = 0.0;
}

// --- Pixel to normalized image coordinates ---

inline void pixel_to_normalized(const CameraIntrinsics& K, double u, double v, double n[3]) {
    n[0] = (u - K.cx) / K.fx;
    n[1] = (v - K.cy) / K.fy;
    n[2] = 1.0;
}

// --- Solve depths and fit rod pose for one frame ---
// Given normalized image coordinates of the 3 markers and known rod distances,
// compute the 3D positions in camera coordinates and fit (R, t).
//
// The 3D points satisfy:
//   P_A = λ_a * a_n, P_B = λ_b * b_n, P_C = λ_c * c_n
//   |P_B - P_A| = L1, |P_C - P_B| = L2
//   P_A, P_B, P_C are collinear → P_C - P_A = k * (P_B - P_A) where k = (L1+L2)/L1
//
// Strategy: search over λ_a, solve for λ_b from AB distance, compute λ_c from collinearity,
// and pick the λ_a that best satisfies the BC distance constraint.

bool solve_frame_pose(const CameraIntrinsics& K, const RodObservation& obs,
                       double L1, double L2, double R[9], double t[3]) {
    // Step 1: Convert pixel coords to normalized image coords
    double a_n[3], b_n[3], c_n[3];
    pixel_to_normalized(K, obs.marker_a.x, obs.marker_a.y, a_n);
    pixel_to_normalized(K, obs.marker_b.x, obs.marker_b.y, b_n);
    pixel_to_normalized(K, obs.marker_c.x, obs.marker_c.y, c_n);

    // Step 2: Compute direction vector magnitudes and dot products
    double d_a2 = a_n[0] * a_n[0] + a_n[1] * a_n[1] + 1.0;
    double d_b2 = b_n[0] * b_n[0] + b_n[1] * b_n[1] + 1.0;
    double d_c2 = c_n[0] * c_n[0] + c_n[1] * c_n[1] + 1.0;
    double s_ab = a_n[0] * b_n[0] + a_n[1] * b_n[1] + 1.0;
    double s_bc = b_n[0] * c_n[0] + b_n[1] * c_n[1] + 1.0;

    double k = (L1 + L2) / L1;  // AC/AB ratio

    // --- Step 3: Search for λ_a that satisfies both distance constraints ---
    //
    // Given λ_a, the AB distance constraint is a quadratic in λ_b:
    //   d_b2 * λ_b² - 2*λ_a*s_ab*λ_b + (λ_a²*d_a2 - L1²) = 0
    //
    // Discriminant: Δ = 4[λ_a²(s_ab² - d_b2*d_a2) + d_b2*L1²]
    // Require Δ ≥ 0 → λ_a² ≤ d_b2*L1² / (d_b2*d_a2 - s_ab²)
    //
    // Then λ_c = (1-k)λ_a + k*λ_b from collinearity.
    // Check: |λ_c*c_n - λ_b*b_n|² = L2²

    double max_la2 = 0.0;
    double denom_ab = d_b2 * d_a2 - s_ab * s_ab;  // > 0 by Cauchy-Schwarz (unless parallel)
    if (denom_ab > 1e-30) {
        max_la2 = d_b2 * L1 * L1 / denom_ab;
    } else {
        return false;
    }

    double max_la = std::sqrt(max_la2);
    // Start search from a small distance (rod must be in front of camera)
    double la_min = L1 * 0.1;

    double best_la = 0.0;
    double best_lb = 0.0;
    double best_lc = 0.0;
    double best_err = 1e30;

    // Sample 500 values of λ_a logarithmically for better accuracy
    for (int s = 0; s < 500; ++s) {
        double t = static_cast<double>(s) / 499.0;
        double la = la_min * std::pow(max_la / la_min, t);

        // Solve AB distance quadratic for λ_b
        double disc_ab = la * la * s_ab * s_ab - d_b2 * (la * la * d_a2 - L1 * L1);
        if (disc_ab < 0.0) continue;

        double sqrt_disc = std::sqrt(disc_ab);
        double lb_candidates[2] = {
            (la * s_ab + sqrt_disc) / d_b2,
            (la * s_ab - sqrt_disc) / d_b2
        };

        for (double lb : lb_candidates) {
            if (lb <= 0.0) continue;

            // Compute λ_c from collinearity
            double lc = (1.0 - k) * la + k * lb;
            if (lc <= 0.0) continue;

            // Check BC distance error
            double bc_diff[3] = {
                lc * c_n[0] - lb * b_n[0],
                lc * c_n[1] - lb * b_n[1],
                lc - lb
            };
            double bc_dist2 = bc_diff[0] * bc_diff[0] + bc_diff[1] * bc_diff[1] + bc_diff[2] * bc_diff[2];
            double err = std::abs(bc_dist2 - L2 * L2);

            if (err < best_err) {
                best_err = err;
                best_la = la;
                best_lb = lb;
                best_lc = lc;
            }
        }
    }

    // Also check the distance ratio cross check: |P_C - P_A| ≈ L1 + L2
    double ac_err = 0.0;
    double la = best_la;
    double lb = best_lb;
    double lc = best_lc;

    double ac_diff[3] = {
        lc * c_n[0] - la * a_n[0],
        lc * c_n[1] - la * a_n[1],
        lc - la
    };
    double ac_dist2 = ac_diff[0] * ac_diff[0] + ac_diff[1] * ac_diff[1] + ac_diff[2] * ac_diff[2];
    double ac_expected = L1 + L2;
    ac_err = std::abs(std::sqrt(ac_dist2) - ac_expected);

    // Require at least one constraint to be approximately satisfied
    // Use relaxed tolerance since initial intrinsics are approximate
    double rel_tol = 0.10;  // 10% relative tolerance for initial estimate
    if (best_err > (L2 * L2) * rel_tol * rel_tol && ac_err > ac_expected * rel_tol) {
        // Fallback: use a simple default pose (rod at ~2m, facing camera)
        double z_est = 2000.0;
        la = z_est;
        // Compute lb from AB constraint with la fixed
        double disc = la * la * s_ab * s_ab - d_b2 * (la * la * d_a2 - L1 * L1);
        if (disc < 0.0) {
            // Increase la until AB constraint is feasible
            la = L1 / std::sqrt(d_b2) * 2.0;
        }
        disc = la * la * s_ab * s_ab - d_b2 * (la * la * d_a2 - L1 * L1);
        if (disc < 0.0) return false;
        lb = (la * s_ab + std::sqrt(disc)) / d_b2;
        lc = (1.0 - k) * la + k * lb;
        if (lb <= 0.0 || lc <= 0.0) return false;
    }

    // Step 4: Compute 3D positions in camera coordinates (already computed above)

    double P_A[3] = {la * a_n[0], la * a_n[1], la * a_n[2]};
    double P_B[3] = {lb * b_n[0], lb * b_n[1], lb * b_n[2]};
    double P_C[3] = {lc * c_n[0], lc * c_n[1], lc * c_n[2]};

    // Step 5: Fit rigid transform from rod local frame to camera frame
    // Rod points: A=(0,0,0), B=(L1,0,0), C=(L1+L2,0,0)
    // We need R s.t. R * rod_pt + t ≈ P_camera

    // Translation: t ≈ P_A (camera coords of origin)
    t[0] = P_A[0];
    t[1] = P_A[1];
    t[2] = P_A[2];

    // Direction: rod x-axis = normalize(P_B - P_A)
    double dx[3] = {P_B[0] - P_A[0], P_B[1] - P_A[1], P_B[2] - P_A[2]};
    detail::normalize3(dx);

    // Build y-axis perpendicular to dx and camera line-of-sight to A
    double P_A_norm = detail::norm3(P_A);
    double los[3] = {P_A[0] / P_A_norm, P_A[1] / P_A_norm, P_A[2] / P_A_norm};

    double dy[3];
    detail::cross3(dx, los, dy);
    double dy_norm = detail::norm3(dy);
    if (dy_norm < 1e-10) {
        // dx parallel to los, pick arbitrary perpendicular
        if (std::abs(dx[0]) < 0.9) {
            dy[0] = 0.0;
            dy[1] = dx[2];
            dy[2] = -dx[1];
        } else {
            dy[0] = -dx[2];
            dy[1] = 0.0;
            dy[2] = dx[0];
        }
        detail::normalize3(dy);
    } else {
        dy[0] /= dy_norm;
        dy[1] /= dy_norm;
        dy[2] /= dy_norm;
    }

    // z-axis = dx × dy
    double dz[3];
    detail::cross3(dx, dy, dz);

    // Rotation matrix: columns are dx, dy, dz (maps rod frame axes to camera frame)
    R[0] = dx[0]; R[1] = dy[0]; R[2] = dz[0];
    R[3] = dx[1]; R[4] = dy[1]; R[5] = dz[1];
    R[6] = dx[2]; R[7] = dy[2]; R[8] = dz[2];

    return true;
}

// --- Joint LM optimization context with parameter scaling ---
//
// To avoid ill-conditioning from different parameter scales
// (fx~1000 vs r~0.01), all parameters are scaled to ~1:
//   fx_scaled = fx / fscale, cx_scaled = cx / fscale, etc.
//   t_scaled  = t  / tscale
//   r, k1, k2, k3, p1, p2 are already ~1 or smaller — no scaling needed

static const double kFocalScale = 1000.0;
static const double kTransScale = 1000.0;

struct JointLMContext {
    const CameraObservations* camera;
    double rod_pts[3][3];
    int num_frames;
    bool fix_principal_point;
    bool fix_aspect_ratio;
    bool estimate_distortion;
    int image_width;
    int image_height;

    int num_params() const {
        int np = 1;  // fx
        if (!fix_aspect_ratio) np += 1;   // fy
        if (!fix_principal_point) np += 2; // cx, cy
        if (estimate_distortion) np += 5;  // k1,k2,k3,p1,p2
        np += 6 * num_frames;  // per-frame (r[3], t[3])
        return np;
    }

    void pack(const CameraIntrinsics& K,
              const double* frame_R, const double* frame_t,
              double* params) const {
        int idx = 0;
        params[idx++] = K.fx / kFocalScale;
        if (!fix_aspect_ratio) params[idx++] = K.fy / kFocalScale;
        if (!fix_principal_point) {
            params[idx++] = K.cx / kFocalScale;
            params[idx++] = K.cy / kFocalScale;
        }
        if (estimate_distortion) {
            params[idx++] = K.k1;
            params[idx++] = K.k2;
            params[idx++] = K.k3;
            params[idx++] = K.p1;
            params[idx++] = K.p2;
        }
        for (int i = 0; i < num_frames; ++i) {
            const double* Ri = frame_R + 9 * i;
            double trace = Ri[0] + Ri[4] + Ri[8];
            double cos_theta = (trace - 1.0) * 0.5;
            if (cos_theta > 1.0) cos_theta = 1.0;
            if (cos_theta < -1.0) cos_theta = -1.0;
            double theta = std::acos(cos_theta);
            if (theta < 1e-10) {
                params[idx++] = 0.0;
                params[idx++] = 0.0;
                params[idx++] = 0.0;
            } else {
                double factor = theta / (2.0 * std::sin(theta));
                params[idx++] = factor * (Ri[7] - Ri[5]);
                params[idx++] = factor * (Ri[2] - Ri[6]);
                params[idx++] = factor * (Ri[3] - Ri[1]);
            }
            const double* ti = frame_t + 3 * i;
            params[idx++] = ti[0] / kTransScale;
            params[idx++] = ti[1] / kTransScale;
            params[idx++] = ti[2] / kTransScale;
        }
    }

    void unpack(const double* params, CameraIntrinsics& K,
                double* frame_R, double* frame_t) const {
        int idx = 0;
        K.fx = params[idx++] * kFocalScale;
        K.fy = fix_aspect_ratio ? K.fx : params[idx++] * kFocalScale;
        K.cx = fix_principal_point ? (image_width * 0.5) : params[idx++] * kFocalScale;
        K.cy = fix_principal_point ? (image_height * 0.5) : params[idx++] * kFocalScale;
        if (estimate_distortion) {
            K.k1 = params[idx++];
            K.k2 = params[idx++];
            K.k3 = params[idx++];
            K.p1 = params[idx++];
            K.p2 = params[idx++];
        } else {
            K.k1 = K.k2 = K.k3 = K.p1 = K.p2 = 0.0;
        }
        for (int i = 0; i < num_frames; ++i) {
            double r[3] = {params[idx], params[idx + 1], params[idx + 2]};
            idx += 3;
            double R[9];
            detail::rodrigues_to_matrix(r, R);
            for (int j = 0; j < 9; ++j) frame_R[9 * i + j] = R[j];
            frame_t[3 * i + 0] = params[idx++] * kTransScale;
            frame_t[3 * i + 1] = params[idx++] * kTransScale;
            frame_t[3 * i + 2] = params[idx++] * kTransScale;
        }
    }
};

void joint_residuals(const double* params, double* residuals, void* ctx) {
    auto* c = static_cast<JointLMContext*>(ctx);

    CameraIntrinsics K;
    std::vector<double> frame_R(9 * c->num_frames);
    std::vector<double> frame_t(3 * c->num_frames);
    c->unpack(params, K, frame_R.data(), frame_t.data());

    for (int i = 0; i < c->num_frames; ++i) {
        const RodObservation& obs = c->camera->frames[i];
        const double* Ri = frame_R.data() + 9 * i;
        const double* ti = frame_t.data() + 3 * i;

        Point2D proj[3];
        detail::project_point_world(K, Ri, ti, c->rod_pts[0], proj[0]);
        detail::project_point_world(K, Ri, ti, c->rod_pts[1], proj[1]);
        detail::project_point_world(K, Ri, ti, c->rod_pts[2], proj[2]);

        int base = 6 * i;
        residuals[base + 0] = proj[0].x - obs.marker_a.x;
        residuals[base + 1] = proj[0].y - obs.marker_a.y;
        residuals[base + 2] = proj[1].x - obs.marker_b.x;
        residuals[base + 3] = proj[1].y - obs.marker_b.y;
        residuals[base + 4] = proj[2].x - obs.marker_c.x;
        residuals[base + 5] = proj[2].y - obs.marker_c.y;
    }
}

void joint_jacobian(const double* params, double* J, void* ctx) {
    auto* c = static_cast<JointLMContext*>(ctx);
    int np = c->num_params();
    int nr = 6 * c->num_frames;

    std::vector<double> p_plus(np), p_minus(np), r_plus(nr), r_minus(nr);

    for (int i = 0; i < np; ++i) {
        for (int j = 0; j < np; ++j) {
            p_plus[j] = params[j];
            p_minus[j] = params[j];
        }
        // Use a constant eps since all parameters are scaled to ~1
        double eps = 1e-5;
        p_plus[i] += eps;
        p_minus[i] -= eps;

        joint_residuals(p_plus.data(), r_plus.data(), ctx);
        joint_residuals(p_minus.data(), r_minus.data(), ctx);

        for (int j = 0; j < nr; ++j) {
            J[j * np + i] = (r_plus[j] - r_minus[j]) / (2.0 * eps);
        }
    }
}

// --- Multi-camera calibration with shared rod poses ---
//
// At each time step, all cameras observe the SAME rod pose.
// This breaks the scale ambiguity present in single-camera calibration.
//
// Parameters (scaled to ~1):
//   Per camera: fx/1000, (cx/1000, cy/1000), (k1,k2,k3,p1,p2)
//   Camera 1..C-1 extrinsics: r_c[3], t_c/1000[3] (camera 0 defines world frame)
//   Per frame rod pose: r_i[3], t_i/1000[3]

struct MultiCamContext {
    const CameraObservations* cameras;
    int num_cameras;
    int num_frames;
    double rod_pts[3][3];
    int image_width;
    int image_height;
    bool fix_principal_point;
    bool fix_aspect_ratio;
    bool estimate_distortion;

    int num_intrinsic_params() const {
        int n = 1;  // fx
        if (!fix_aspect_ratio) n += 1;   // fy
        if (!fix_principal_point) n += 2; // cx, cy
        if (estimate_distortion) n += 5;  // k1,k2,k3,p1,p2
        return n;
    }

    int num_params() const {
        int n = num_cameras * num_intrinsic_params();  // per-camera intrinsics
        n += 6 * (num_cameras - 1);  // camera 1..C-1 extrinsics (camera 0 = world)
        n += 6 * num_frames;         // shared rod poses
        return n;
    }

    void pack(const CameraIntrinsics* K_arr,
              const double* cam_R, const double* cam_t,  // per-camera extrinsics
              const double* rod_R, const double* rod_t,  // per-frame rod poses
              double* params) const {
        int idx = 0;

        // Per-camera intrinsics
        for (int c = 0; c < num_cameras; ++c) {
            const CameraIntrinsics& K = K_arr[c];
            params[idx++] = K.fx / kFocalScale;
            if (!fix_aspect_ratio) params[idx++] = K.fy / kFocalScale;
            if (!fix_principal_point) {
                params[idx++] = K.cx / kFocalScale;
                params[idx++] = K.cy / kFocalScale;
            }
            if (estimate_distortion) {
                params[idx++] = K.k1;
                params[idx++] = K.k2;
                params[idx++] = K.k3;
                params[idx++] = K.p1;
                params[idx++] = K.p2;
            }
        }

        // Camera extrinsics (camera 1..C-1 relative to camera 0 = world)
        for (int c = 1; c < num_cameras; ++c) {
            const double* Rc = cam_R + 9 * (c - 1);
            double trace = Rc[0] + Rc[4] + Rc[8];
            double cos_theta = (trace - 1.0) * 0.5;
            if (cos_theta > 1.0) cos_theta = 1.0;
            if (cos_theta < -1.0) cos_theta = -1.0;
            double theta = std::acos(cos_theta);
            if (theta < 1e-10) {
                params[idx++] = 0.0; params[idx++] = 0.0; params[idx++] = 0.0;
            } else {
                double factor = theta / (2.0 * std::sin(theta));
                params[idx++] = factor * (Rc[7] - Rc[5]);
                params[idx++] = factor * (Rc[2] - Rc[6]);
                params[idx++] = factor * (Rc[3] - Rc[1]);
            }
            const double* tc = cam_t + 3 * (c - 1);
            params[idx++] = tc[0] / kTransScale;
            params[idx++] = tc[1] / kTransScale;
            params[idx++] = tc[2] / kTransScale;
        }

        // Shared rod poses
        for (int i = 0; i < num_frames; ++i) {
            const double* Ri = rod_R + 9 * i;
            double trace = Ri[0] + Ri[4] + Ri[8];
            double cos_theta = (trace - 1.0) * 0.5;
            if (cos_theta > 1.0) cos_theta = 1.0;
            if (cos_theta < -1.0) cos_theta = -1.0;
            double theta = std::acos(cos_theta);
            if (theta < 1e-10) {
                params[idx++] = 0.0; params[idx++] = 0.0; params[idx++] = 0.0;
            } else {
                double factor = theta / (2.0 * std::sin(theta));
                params[idx++] = factor * (Ri[7] - Ri[5]);
                params[idx++] = factor * (Ri[2] - Ri[6]);
                params[idx++] = factor * (Ri[3] - Ri[1]);
            }
            const double* ti = rod_t + 3 * i;
            params[idx++] = ti[0] / kTransScale;
            params[idx++] = ti[1] / kTransScale;
            params[idx++] = ti[2] / kTransScale;
        }
    }

    void unpack(const double* params, CameraIntrinsics* K_arr,
                double* cam_R, double* cam_t,
                double* rod_R, double* rod_t) const {
        int idx = 0;

        for (int c = 0; c < num_cameras; ++c) {
            CameraIntrinsics& K = K_arr[c];
            K.fx = params[idx++] * kFocalScale;
            K.fy = fix_aspect_ratio ? K.fx : params[idx++] * kFocalScale;
            K.cx = fix_principal_point ? (image_width * 0.5) : params[idx++] * kFocalScale;
            K.cy = fix_principal_point ? (image_height * 0.5) : params[idx++] * kFocalScale;
            if (estimate_distortion) {
                K.k1 = params[idx++];
                K.k2 = params[idx++];
                K.k3 = params[idx++];
                K.p1 = params[idx++];
                K.p2 = params[idx++];
            } else {
                K.k1 = K.k2 = K.k3 = K.p1 = K.p2 = 0.0;
            }
        }

        // Camera 0 extrinsics = identity (world frame)
        if (cam_R) {
            cam_R[0] = 1.0; cam_R[1] = 0.0; cam_R[2] = 0.0;
            cam_R[3] = 0.0; cam_R[4] = 1.0; cam_R[5] = 0.0;
            cam_R[6] = 0.0; cam_R[7] = 0.0; cam_R[8] = 1.0;
        }
        if (cam_t) {
            cam_t[0] = 0.0; cam_t[1] = 0.0; cam_t[2] = 0.0;
        }

        for (int c = 1; c < num_cameras; ++c) {
            double r[3] = {params[idx], params[idx + 1], params[idx + 2]};
            idx += 3;
            double R[9];
            detail::rodrigues_to_matrix(r, R);
            if (cam_R) {
                for (int j = 0; j < 9; ++j) cam_R[9 * (c - 1) + j] = R[j];
            }
            if (cam_t) {
                cam_t[3 * (c - 1) + 0] = params[idx++] * kTransScale;
                cam_t[3 * (c - 1) + 1] = params[idx++] * kTransScale;
                cam_t[3 * (c - 1) + 2] = params[idx++] * kTransScale;
            }
        }

        for (int i = 0; i < num_frames; ++i) {
            double r[3] = {params[idx], params[idx + 1], params[idx + 2]};
            idx += 3;
            detail::rodrigues_to_matrix(r, rod_R + 9 * i);
            rod_t[3 * i + 0] = params[idx++] * kTransScale;
            rod_t[3 * i + 1] = params[idx++] * kTransScale;
            rod_t[3 * i + 2] = params[idx++] * kTransScale;
        }
    }
};

void multicam_residuals(const double* params, double* residuals, void* ctx) {
    auto* c = static_cast<MultiCamContext*>(ctx);

    std::vector<CameraIntrinsics> K_arr(c->num_cameras);
    std::vector<double> cam_R(9 * c->num_cameras);
    std::vector<double> cam_t(3 * c->num_cameras);
    std::vector<double> rod_R(9 * c->num_frames);
    std::vector<double> rod_t(3 * c->num_frames);

    c->unpack(params, K_arr.data(), cam_R.data(), cam_t.data(), rod_R.data(), rod_t.data());

    // cam_R[0..8] = camera 0 extrinsics (identity)
    // cam_R[9..17], cam_R[18..26], ... = camera 1,2,... extrinsics (world-to-cam)

    int res_idx = 0;
    for (int cam = 0; cam < c->num_cameras; ++cam) {
        const double* R_w2c = cam_R.data() + 9 * cam;
        const double* t_w2c = cam_t.data() + 3 * cam;

        for (int i = 0; i < c->num_frames; ++i) {
            const RodObservation& obs = c->cameras[cam].frames[i];

            // Rod pose in this camera's frame:
            // P_cam = R_w2c * (R_rod_world * P_rod + t_rod_world) + t_w2c
            //       = (R_w2c * R_rod_world) * P_rod + (R_w2c * t_rod_world + t_w2c)

            double R_combined[9];
            detail::mat3x3_mult(R_w2c, rod_R.data() + 9 * i, R_combined);

            double t_combined[3];
            detail::mat3x3_vec3_mult(R_w2c, rod_t.data() + 3 * i, t_combined);
            t_combined[0] += t_w2c[0];
            t_combined[1] += t_w2c[1];
            t_combined[2] += t_w2c[2];

            Point2D proj[3];
            detail::project_point_world(K_arr[cam], R_combined, t_combined, c->rod_pts[0], proj[0]);
            detail::project_point_world(K_arr[cam], R_combined, t_combined, c->rod_pts[1], proj[1]);
            detail::project_point_world(K_arr[cam], R_combined, t_combined, c->rod_pts[2], proj[2]);

            residuals[res_idx++] = proj[0].x - obs.marker_a.x;
            residuals[res_idx++] = proj[0].y - obs.marker_a.y;
            residuals[res_idx++] = proj[1].x - obs.marker_b.x;
            residuals[res_idx++] = proj[1].y - obs.marker_b.y;
            residuals[res_idx++] = proj[2].x - obs.marker_c.x;
            residuals[res_idx++] = proj[2].y - obs.marker_c.y;
        }
    }
}

void multicam_jacobian(const double* params, double* J, void* ctx) {
    auto* c = static_cast<MultiCamContext*>(ctx);
    int np = c->num_params();
    int nr = 6 * c->num_cameras * c->num_frames;

    std::vector<double> p_plus(np), p_minus(np), r_plus(nr), r_minus(nr);

    for (int i = 0; i < np; ++i) {
        for (int j = 0; j < np; ++j) {
            p_plus[j] = params[j];
            p_minus[j] = params[j];
        }
        double eps = 1e-5;
        p_plus[i] += eps;
        p_minus[i] -= eps;

        multicam_residuals(p_plus.data(), r_plus.data(), ctx);
        multicam_residuals(p_minus.data(), r_minus.data(), ctx);

        for (int j = 0; j < nr; ++j) {
            J[j * np + i] = (r_plus[j] - r_minus[j]) / (2.0 * eps);
        }
    }
}

// --- Main per-camera calibration (single camera, approximate) ---

CalibrationError calibrate_single_camera(const CameraObservations& camera,
                                          const ThreePointConfig& config,
                                          CameraCalibration& result) {
    int M = camera.frame_count;
    double L1 = config.ab_distance;
    double L2 = config.bc_distance;

    double rod_pts[3][3];
    build_rod_model(L1, L2, rod_pts);

    // --- Initialize intrinsics ---
    CameraIntrinsics& K = result.intrinsics;
    K.fx = static_cast<double>(std::max(config.image_width, config.image_height));
    K.fy = K.fx;
    K.cx = config.image_width * 0.5;
    K.cy = config.image_height * 0.5;
    K.k1 = K.k2 = K.k3 = K.p1 = K.p2 = 0.0;

    // --- Initialize per-frame extrinsics ---
    std::vector<double> frame_R(9 * M);
    std::vector<double> frame_t(3 * M);

    int successful_init_frames = 0;
    for (int i = 0; i < M; ++i) {
        double R[9], t[3];
        if (solve_frame_pose(K, camera.frames[i], L1, L2, R, t)) {
            for (int j = 0; j < 9; ++j) frame_R[9 * i + j] = R[j];
            frame_t[3 * i + 0] = t[0];
            frame_t[3 * i + 1] = t[1];
            frame_t[3 * i + 2] = t[2];
            ++successful_init_frames;
        } else {
            return CalibrationError::SingularMatrix;
        }
    }

    if (successful_init_frames < 3) return CalibrationError::InsufficientObservations;

    // --- Joint LM refinement (intrinsics + all per-frame poses) ---
    // Parameters are scaled to ~1 for numerical stability:
    //   fx_scaled = fx / 1000, t_scaled = t / 1000

    JointLMContext ctx;
    ctx.camera = &camera;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) ctx.rod_pts[i][j] = rod_pts[i][j];
    ctx.num_frames = M;
    ctx.fix_principal_point = config.fix_principal_point;
    ctx.fix_aspect_ratio = config.fix_aspect_ratio;
    ctx.estimate_distortion = config.estimate_distortion;
    ctx.image_width = config.image_width;
    ctx.image_height = config.image_height;

    int np = ctx.num_params();
    int nr = 6 * M;

    std::vector<double> params(np);
    ctx.pack(K, frame_R.data(), frame_t.data(), params.data());

    bool converged = detail::levenberg_marquardt(
        nr, np, params.data(),
        joint_residuals, joint_jacobian,
        &ctx, config.max_iterations, config.tolerance);

    // --- Unpack final result ---
    ctx.unpack(params.data(), K, frame_R.data(), frame_t.data());

    if (!converged) return CalibrationError::OptimizationFailed;

    // Store the extrinsics from the first frame (or average pose)
    // For multi-camera systems, the extrinsics are per-frame.
    // We store the first frame's extrinsics as the "reference" extrinsics.
    result.extrinsics.rotation[0] = frame_R[0];
    result.extrinsics.rotation[1] = frame_R[1];
    result.extrinsics.rotation[2] = frame_R[2];
    result.extrinsics.rotation[3] = frame_R[3];
    result.extrinsics.rotation[4] = frame_R[4];
    result.extrinsics.rotation[5] = frame_R[5];
    result.extrinsics.rotation[6] = frame_R[6];
    result.extrinsics.rotation[7] = frame_R[7];
    result.extrinsics.rotation[8] = frame_R[8];
    result.extrinsics.translation[0] = frame_t[0];
    result.extrinsics.translation[1] = frame_t[1];
    result.extrinsics.translation[2] = frame_t[2];

    // --- Compute final reprojection error ---
    double total_error = 0.0;
    int total_points = 0;
    for (int i = 0; i < M; ++i) {
        const double* Ri = frame_R.data() + 9 * i;
        const double* ti = frame_t.data() + 3 * i;

        Point2D proj[3];
        detail::project_point_world(K, Ri, ti, rod_pts[0], proj[0]);
        detail::project_point_world(K, Ri, ti, rod_pts[1], proj[1]);
        detail::project_point_world(K, Ri, ti, rod_pts[2], proj[2]);

        double dx, dy;
        dx = proj[0].x - camera.frames[i].marker_a.x;
        dy = proj[0].y - camera.frames[i].marker_a.y;
        total_error += dx * dx + dy * dy;

        dx = proj[1].x - camera.frames[i].marker_b.x;
        dy = proj[1].y - camera.frames[i].marker_b.y;
        total_error += dx * dx + dy * dy;

        dx = proj[2].x - camera.frames[i].marker_c.x;
        dy = proj[2].y - camera.frames[i].marker_c.y;
        total_error += dx * dx + dy * dy;

        total_points += 3;
    }
    result.reprojection_error = std::sqrt(total_error / static_cast<double>(total_points));

    return CalibrationError::Ok;
}

}  // anonymous namespace

// ============================================================
//  Public API
// ============================================================

CalibrationError process_three_point(const CameraObservations* cameras,
                                      int camera_count,
                                      CameraCalibration* results,
                                      const ThreePointConfig* config) {
    CalibrationError err = validate_calibration_inputs(cameras, camera_count, config);
    if (err != CalibrationError::Ok) return err;
    if (!results) return CalibrationError::NullInput;

    int M = cameras[0].frame_count;
    for (int c = 1; c < camera_count; ++c) {
        if (cameras[c].frame_count != M) return CalibrationError::InvalidFrameCount;
    }

    if (camera_count == 1) {
        return calibrate_single_camera(cameras[0], *config, results[0]);
    }

    double L1 = config->ab_distance;
    double L2 = config->bc_distance;
    double rod_pts[3][3];
    build_rod_model(L1, L2, rod_pts);

    // Initialize intrinsics
    double f_init = static_cast<double>(std::max(config->image_width, config->image_height));
    std::vector<CameraIntrinsics> K_arr(camera_count);
    for (int c = 0; c < camera_count; ++c) {
        K_arr[c].fx = f_init;
        K_arr[c].fy = f_init;
        K_arr[c].cx = config->image_width * 0.5;
        K_arr[c].cy = config->image_height * 0.5;
        K_arr[c].k1 = K_arr[c].k2 = K_arr[c].k3 = K_arr[c].p1 = K_arr[c].p2 = 0.0;
    }

    // For multi-camera calibration, the absolute focal length scale is ambiguous
    // (same rod observations can be explained by different (fx, depth) pairs).
    // However, the RELATIVE focal lengths between cameras ARE determined.
    // Strategy: fix camera 0's fx to the initial guess, optimize scales for other cameras.

    struct PerFrameData {
        double t[3];
    };

    // Estimate per-frame 3D positions for camera 0 (using initial fx)
    std::vector<PerFrameData> pts0(M);
    for (int i = 0; i < M; ++i) {
        double R[9], t[3];
        if (!solve_frame_pose(K_arr[0], cameras[0].frames[i], L1, L2, R, t)) {
            return CalibrationError::SingularMatrix;
        }
        pts0[i].t[0] = t[0];
        pts0[i].t[1] = t[1];
        pts0[i].t[2] = t[2];
    }

    // For each other camera, find the scale factor relative to camera 0
    // that produces consistent 3D geometry (minimizes variance of inter-camera distances)
    for (int c = 1; c < camera_count; ++c) {
        double best_scale = 1.0;
        double best_cost = 1e30;

        for (int ss = -20; ss <= 30; ++ss) {
            double scale = 1.0 + ss * 0.04;
            if (scale <= 0.1 || scale > 10.0) continue;

            CameraIntrinsics Kc = K_arr[c];
            Kc.fx = K_arr[0].fx * scale;
            Kc.fy = config->fix_aspect_ratio ? Kc.fx : Kc.fy;

            std::vector<double> dists;
            for (int i = 0; i < M; ++i) {
                double Rc[9], tc[3];
                if (!solve_frame_pose(Kc, cameras[c].frames[i], L1, L2, Rc, tc)) {
                    continue;
                }
                double dx = pts0[i].t[0] - tc[0];
                double dy = pts0[i].t[1] - tc[1];
                double dz = pts0[i].t[2] - tc[2];
                dists.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
            }

            if (dists.size() < 3) continue;

            double sum = 0.0, sum2 = 0.0;
            for (double d : dists) {
                sum += d;
                sum2 += d * d;
            }
            double mean = sum / dists.size();
            double var = sum2 / dists.size() - mean * mean;
            if (var < 0.0) var = 0.0;
            double cost_c = std::sqrt(var) / std::max(1.0, mean);

            if (cost_c < best_cost) {
                best_cost = cost_c;
                best_scale = scale;
            }
        }

        K_arr[c].fx = K_arr[0].fx * best_scale;
        K_arr[c].fy = config->fix_aspect_ratio ? K_arr[c].fx : K_arr[c].fy;
    }

    // Compute final per-frame poses and camera extrinsics
    std::vector<double> rod_R(9 * M);
    std::vector<double> rod_t(3 * M);
    for (int i = 0; i < M; ++i) {
        double R[9], t[3];
        if (!solve_frame_pose(K_arr[0], cameras[0].frames[i], L1, L2, R, t)) {
            return CalibrationError::SingularMatrix;
        }
        for (int j = 0; j < 9; ++j) rod_R[9 * i + j] = R[j];
        rod_t[3 * i + 0] = t[0];
        rod_t[3 * i + 1] = t[1];
        rod_t[3 * i + 2] = t[2];
    }

    std::vector<double> cam_R(9 * camera_count);
    std::vector<double> cam_t(3 * camera_count);
    cam_R[0] = 1.0; cam_R[1] = 0.0; cam_R[2] = 0.0;
    cam_R[3] = 0.0; cam_R[4] = 1.0; cam_R[5] = 0.0;
    cam_R[6] = 0.0; cam_R[7] = 0.0; cam_R[8] = 1.0;
    cam_t[0] = cam_t[1] = cam_t[2] = 0.0;

    for (int c = 1; c < camera_count; ++c) {
        std::vector<double> pts0, ptsc;
        for (int i = 0; i < M; ++i) {
            double R_c[9], t_c[3];
            if (!solve_frame_pose(K_arr[c], cameras[c].frames[i], L1, L2, R_c, t_c)) {
                continue;
            }
            pts0.push_back(rod_t[3*i+0]); pts0.push_back(rod_t[3*i+1]); pts0.push_back(rod_t[3*i+2]);
            ptsc.push_back(t_c[0]); ptsc.push_back(t_c[1]); ptsc.push_back(t_c[2]);
        }

        int npts = static_cast<int>(pts0.size()) / 3;
        if (npts >= 3) {
            double c0[3] = {0,0,0}, cc[3] = {0,0,0};
            for (int k = 0; k < npts; ++k) {
                for (int d = 0; d < 3; ++d) {
                    c0[d] += pts0[3*k+d];
                    cc[d] += ptsc[3*k+d];
                }
            }
            for (int d = 0; d < 3; ++d) { c0[d] /= npts; cc[d] /= npts; }

            double H[9] = {0};
            for (int k = 0; k < npts; ++k) {
                double da[3] = {pts0[3*k+0]-c0[0], pts0[3*k+1]-c0[1], pts0[3*k+2]-c0[2]};
                double db[3] = {ptsc[3*k+0]-cc[0], ptsc[3*k+1]-cc[1], ptsc[3*k+2]-cc[2]};
                for (int r = 0; r < 3; ++r)
                    for (int s = 0; s < 3; ++s)
                        H[3*r+s] += da[r] * db[s];
            }

            double S[3], V[9];
            for (int k = 0; k < 9; ++k) cam_R[9*c+k] = H[k];
            detail::svd_3x3(cam_R.data() + 9*c, S, V);

            // After svd_3x3, cam_R+9*c contains U. Procrustes rotation: R = V * U^T
            double Ut[9], R_est[9];
            detail::mat3x3_transpose(cam_R.data() + 9*c, Ut);
            detail::mat3x3_mult(V, Ut, R_est);
            for (int k = 0; k < 9; ++k) cam_R[9*c+k] = R_est[k];

            cam_t[3*c+0] = cc[0] - (R_est[0]*c0[0]+R_est[1]*c0[1]+R_est[2]*c0[2]);
            cam_t[3*c+1] = cc[1] - (R_est[3]*c0[0]+R_est[4]*c0[1]+R_est[5]*c0[2]);
            cam_t[3*c+2] = cc[2] - (R_est[6]*c0[0]+R_est[7]*c0[1]+R_est[8]*c0[2]);
        } else {
            for (int j = 0; j < 9; ++j) cam_R[9*c+j] = (j%4==0) ? 1.0 : 0.0;
            cam_t[3*c+0] = 500.0; cam_t[3*c+1] = 0.0; cam_t[3*c+2] = 0.0;
        }
    }

    // Store results
    for (int c = 0; c < camera_count; ++c) {
        results[c].intrinsics = K_arr[c];
        for (int j = 0; j < 9; ++j) results[c].extrinsics.rotation[j] = cam_R[9*c+j];
        results[c].extrinsics.translation[0] = cam_t[3*c+0];
        results[c].extrinsics.translation[1] = cam_t[3*c+1];
        results[c].extrinsics.translation[2] = cam_t[3*c+2];

        double total_error = 0.0;
        int total_pts = 0;
        for (int i = 0; i < M; ++i) {
            double R_combined[9], t_combined[3];
            detail::mat3x3_mult(cam_R.data() + 9*c, rod_R.data() + 9*i, R_combined);
            detail::mat3x3_vec3_mult(cam_R.data() + 9*c, rod_t.data() + 3*i, t_combined);
            t_combined[0] += cam_t[3*c+0];
            t_combined[1] += cam_t[3*c+1];
            t_combined[2] += cam_t[3*c+2];

            Point2D proj[3];
            detail::project_point_world(K_arr[c], R_combined, t_combined, rod_pts[0], proj[0]);
            detail::project_point_world(K_arr[c], R_combined, t_combined, rod_pts[1], proj[1]);
            detail::project_point_world(K_arr[c], R_combined, t_combined, rod_pts[2], proj[2]);

            const RodObservation& obs = cameras[c].frames[i];
            double dx = proj[0].x-obs.marker_a.x, dy = proj[0].y-obs.marker_a.y;
            total_error += dx*dx+dy*dy;
            dx = proj[1].x-obs.marker_b.x; dy = proj[1].y-obs.marker_b.y;
            total_error += dx*dx+dy*dy;
            dx = proj[2].x-obs.marker_c.x; dy = proj[2].y-obs.marker_c.y;
            total_error += dx*dx+dy*dy;
            total_pts += 3;
        }
        results[c].reprojection_error = std::sqrt(total_error / static_cast<double>(total_pts));
    }

    return CalibrationError::Ok;
}

}  // namespace calibration
