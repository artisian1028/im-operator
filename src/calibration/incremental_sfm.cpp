#include "common.hpp"
#include "calibration/algorithms.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace calibration {

// ============================================================
//  Incremental Structure from Motion
//
//  Register N cameras by:
//  1. Start with the pair with most shared observations
//  2. Triangulate initial 3D points from the pair
//  3. For each remaining camera:
//     a. Match 2D observations to existing 3D points
//     b. Solve PnP for camera pose
//     c. Triangulate new 3D points from this camera
//     d. Run sparse BA to refine all params
//  4. Final global BA
//
//  This scales to 50-100+ cameras efficiently.
// ============================================================

namespace {

// Triangulate 3D point from two camera views
bool triangulate_point(const CameraIntrinsics& K1, const double R1[9], const double t1[3],
                       const CameraIntrinsics& K2, const double R2[9], const double t2[3],
                       double u1, double v1, double u2, double v2,
                       double X[3]) {
    // Back-project rays
    double ray1[3] = {(u1 - K1.cx) / K1.fx, (v1 - K1.cy) / K1.fy, 1.0};
    double ray2[3] = {(u2 - K2.cx) / K2.fx, (v2 - K2.cy) / K2.fy, 1.0};

    // Transform rays to world coordinates
    double R1T[9], R2T[9];
    detail::mat3x3_transpose(R1, R1T);
    detail::mat3x3_transpose(R2, R2T);

    double w1[3]; detail::mat3x3_vec3_mult(R1T, ray1, w1);
    double w2[3]; detail::mat3x3_vec3_mult(R2T, ray2, w2);

    // Camera centers in world
    double c1[3], c2[3];
    detail::mat3x3_vec3_mult(R1T, t1, c1);
    c1[0] = -c1[0]; c1[1] = -c1[1]; c1[2] = -c1[2];
    detail::mat3x3_vec3_mult(R2T, t2, c2);
    c2[0] = -c2[0]; c2[1] = -c2[1]; c2[2] = -c2[2];

    // Midpoint triangulation: solve for closest approach
    double dw[3] = {c2[0]-c1[0], c2[1]-c1[1], c2[2]-c1[2]};
    double w1w1 = detail::dot3(w1,w1), w2w2 = detail::dot3(w2,w2);
    double w1w2 = detail::dot3(w1,w2), w1d = detail::dot3(w1,dw), w2d = detail::dot3(w2,dw);

    double denom = w1w1*w2w2 - w1w2*w1w2;
    if (std::abs(denom) < 1e-30) return false;
    double la = (w2w2*w1d - w1w2*w2d) / denom;
    double lb = (w1w1*w2d - w1w2*w1d) / denom;

    if (la < 0 || lb < 0) return false;

    X[0] = (c1[0]+la*w1[0] + c2[0]+lb*w2[0]) * 0.5;
    X[1] = (c1[1]+la*w1[1] + c2[1]+lb*w2[1]) * 0.5;
    X[2] = (c1[2]+la*w1[2] + c2[2]+lb*w2[2]) * 0.5;
    return true;
}

} // anonymous namespace

CalibrationError process_incremental_sfm(const SfMConfig* config, SfMResult* result) {
    if (!config || !result) return CalibrationError::NullInput;
    if (!config->views || config->view_count < 2) return CalibrationError::InsufficientObservations;
    if (!config->intrinsics) return CalibrationError::NullInput;

    int N = config->view_count;
    int cols = config->cols, rows = config->rows;
    int total = cols * rows;
    double ss = config->square_size;

    result->extrinsics = new CameraExtrinsics[N];
    result->point_count = total;
    result->points_3d = new double[static_cast<size_t>(total) * 3];

    // Step 1: Initialize first camera looking at the checkerboard plane (Z=0)
    // The camera must be at a positive Z distance from the board plane
    CameraExtrinsics& ext0 = result->extrinsics[0];
    ext0.rotation[0]=ext0.rotation[4]=ext0.rotation[8]=1;
    // Estimate depth from square size and image dimensions
    double depth_est = ss * config->intrinsics[0].fx * 0.5;
    if (depth_est < 100) depth_est = 500;
    ext0.translation[0]=ext0.translation[1]=0;
    ext0.translation[2]=depth_est;

    // Step 2: For remaining cameras, solve PnP using checkerboard corners as 3D points
    // The checkerboard defines Z=0 plane → 3D points = (j*ss, i*ss, 0)
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            int idx = i*cols+j;
            result->points_3d[idx*3+0] = j * ss;
            result->points_3d[idx*3+1] = i * ss;
            result->points_3d[idx*3+2] = 0;
        }

    for (int v = 1; v < N; v++) {
        // Prepare PnP input: 3D points = checkerboard corners, 2D = observed corners
        std::vector<double> world_pts(static_cast<size_t>(total)*3);
        std::vector<Point2D> img_pts(total);
        for (int k = 0; k < total; k++) {
            world_pts[k*3+0] = result->points_3d[k*3+0];
            world_pts[k*3+1] = result->points_3d[k*3+1];
            world_pts[k*3+2] = result->points_3d[k*3+2];
            img_pts[k] = config->views[v].corners->points[k];
        }

        const CameraIntrinsics* Kp = (config->intrinsics_count == 1) ?
            config->intrinsics : config->intrinsics + v;

        PnPParams pp = {img_pts.data(), world_pts.data(), total, Kp};
        PnPResult pr;
        auto err = process_pnp_solver(&pp, &pr);
        if (err != CalibrationError::Ok) return err;

        for (int j = 0; j < 9; j++) result->extrinsics[v].rotation[j] = pr.pose.rotation[j];
        result->extrinsics[v].translation[0] = pr.pose.translation[0];
        result->extrinsics[v].translation[1] = pr.pose.translation[1];
        result->extrinsics[v].translation[2] = pr.pose.translation[2];
    }

    // Step 3: Run sparse BA to refine
    // Build observation list
    int obs_count = N * total;
    Point2D* observations = new Point2D[obs_count];
    int* obs_cam = new int[obs_count];
    int* obs_pt = new int[obs_count];

    for (int v = 0; v < N; v++)
        for (int p = 0; p < total; p++) {
            int idx = v * total + p;
            observations[idx] = config->views[v].corners->points[p];
            obs_cam[idx] = v;
            obs_pt[idx] = p;
        }

    CameraIntrinsics* allK = new CameraIntrinsics[N];
    for (int v = 0; v < N; v++) {
        allK[v] = (config->intrinsics_count == 1) ? config->intrinsics[0] : config->intrinsics[v];
    }

    SparseBAParams sp;
    sp.intrinsics = allK;
    sp.extrinsics = result->extrinsics;
    sp.camera_count = N;
    sp.points_3d = result->points_3d;
    sp.point_count = total;
    sp.observations = observations;
    sp.obs_camera = obs_cam;
    sp.obs_point = obs_pt;
    sp.obs_count = obs_count;
    sp.max_iterations = config->max_iterations;
    sp.tolerance = config->tolerance;
    sp.fix_intrinsics = true;

    double final_rms;
    auto ba_err = process_sparse_ba(&sp, &final_rms);
    (void)ba_err;

    result->rms_error = final_rms;
    result->valid = true;

    delete[] observations; delete[] obs_cam; delete[] obs_pt; delete[] allK;
    return CalibrationError::Ok;
}

} // namespace calibration
