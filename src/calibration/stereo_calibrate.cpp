#include "common.hpp"
#include "calibration/algorithms.hpp"
#include <vector>
#include <cmath>

namespace calibration {

// ============================================================
//  Stereo calibration: two-camera extrinsic calibration
//
//  Given checkerboard views from both cameras and their individual
//  intrinsics, compute the rotation R and translation t between cameras.
//
//  For each view pair:
//    R = R_right * R_left^T
//    t = t_right - R * t_left
//
//  Average/median over all pairs, then LM refinement.
// ============================================================

namespace {

// Rodrigues vector → matrix wrapper for stereo
void rvec_to_mat(const double rvec[3], double R[9]) {
    detail::rodrigues_to_matrix(rvec, R);
}

// Matrix → Rodrigues vector
void mat_to_rvec(const double R[9], double rvec[3]) {
    double trace = R[0]+R[4]+R[8];
    double cos_t = std::clamp((trace-1.0)*0.5, -1.0, 1.0);
    double theta = std::acos(cos_t);
    if (theta < 1e-10) { rvec[0]=rvec[1]=rvec[2]=0; return; }
    double factor = theta/(2.0*std::sin(theta));
    rvec[0] = factor*(R[7]-R[5]);
    rvec[1] = factor*(R[2]-R[6]);
    rvec[2] = factor*(R[3]-R[1]);
}

struct StereoLMContext {
    const StereoCalibrateParams* params;
    int nviews;
    const CameraIntrinsics* Kl, *Kr;
    int cols, rows;
    double ss;

    int num_params() const { return 6 + 6 * nviews; } // stereo R(3),T(3) + per-view rod pose(6)

    void pack(const double* stereo_rvec, const double* stereo_t,
              const double* pose_rvec, const double* pose_t, double* p) const {
        p[0]=stereo_rvec[0];p[1]=stereo_rvec[1];p[2]=stereo_rvec[2];
        p[3]=stereo_t[0];p[4]=stereo_t[1];p[5]=stereo_t[2];
        for (int v=0; v<nviews; v++) {
            p[6+v*6+0]=pose_rvec[v*3+0]; p[6+v*6+1]=pose_rvec[v*3+1]; p[6+v*6+2]=pose_rvec[v*3+2];
            p[6+v*6+3]=pose_t[v*3+0]; p[6+v*6+4]=pose_t[v*3+1]; p[6+v*6+5]=pose_t[v*3+2];
        }
    }
    void unpack(const double* p, double* stereo_rvec, double* stereo_t,
                double* pose_rvec, double* pose_t) const {
        stereo_rvec[0]=p[0];stereo_rvec[1]=p[1];stereo_rvec[2]=p[2];
        stereo_t[0]=p[3];stereo_t[1]=p[4];stereo_t[2]=p[5];
        for (int v=0; v<nviews; v++) {
            pose_rvec[v*3+0]=p[6+v*6+0]; pose_rvec[v*3+1]=p[6+v*6+1]; pose_rvec[v*3+2]=p[6+v*6+2];
            pose_t[v*3+0]=p[6+v*6+3]; pose_t[v*3+1]=p[6+v*6+4]; pose_t[v*3+2]=p[6+v*6+5];
        }
    }
};

void stereo_residuals(const double* params, double* res, void* ctx) {
    auto* c = static_cast<StereoLMContext*>(ctx);
    double sr[3], st[3];
    std::vector<double> pr(3*c->nviews), pt(3*c->nviews);
    c->unpack(params, sr, st, pr.data(), pt.data());

    double Rs[9]; rvec_to_mat(sr, Rs);

    int ri = 0;
    for (int v = 0; v < c->nviews; v++) {
        double Rv[9]; rvec_to_mat(pr.data()+3*v, Rv);
        double* tv = pt.data()+3*v;

        // Project board points through left camera
        for (int i = 0; i < c->rows; i++) {
            for (int j = 0; j < c->cols; j++) {
                double Pw[3] = {j*c->ss, i*c->ss, 0.0};
                Point2D pl, prj;

                // Left: direct projection
                detail::project_point_world(*c->Kl, Rv, tv, Pw, pl);

                // Right: through stereo transform
                double Pc[3], Pr[3];
                detail::mat3x3_vec3_mult(Rv, Pw, Pc);
                Pc[0]+=tv[0]; Pc[1]+=tv[1]; Pc[2]+=tv[2];
                detail::mat3x3_vec3_mult(Rs, Pc, Pr);
                Pr[0]+=st[0]; Pr[1]+=st[1]; Pr[2]+=st[2];
                detail::project_point(*c->Kr, Pr, prj);

                int idx = i*c->cols+j;
                res[ri++] = pl.x - c->params->left_corners[v].points[idx].x;
                res[ri++] = pl.y - c->params->left_corners[v].points[idx].y;
                res[ri++] = prj.x - c->params->right_corners[v].points[idx].x;
                res[ri++] = prj.y - c->params->right_corners[v].points[idx].y;
            }
        }
    }
}

void stereo_jacobian(const double* p, double* J, void* ctx) {
    auto* c = static_cast<StereoLMContext*>(ctx);
    int np = c->num_params(), nr = 4 * c->nviews * c->cols * c->rows;
    std::vector<double> pp(np), pm(np), rp(nr), rm(nr);
    for (int i = 0; i < np; i++) {
        for (int j = 0; j < np; j++) pp[j]=pm[j]=p[j];
        double eps = 1e-6; pp[i]+=eps; pm[i]-=eps;
        stereo_residuals(pp.data(), rp.data(), ctx);
        stereo_residuals(pm.data(), rm.data(), ctx);
        for (int j = 0; j < nr; j++) J[j*np+i] = (rp[j]-rm[j])/(2.0*eps);
    }
}

} // anonymous namespace

CalibrationError process_stereo_calibrate(const StereoCalibrateParams* params,
                                           CameraExtrinsics* stereo_R,
                                           CameraExtrinsics* stereo_t,
                                           double* rms_error) {
    if (!params || !stereo_R || !stereo_t) return CalibrationError::NullInput;
    if (!params->left_corners || !params->right_corners || !params->left_intrinsics || !params->right_intrinsics)
        return CalibrationError::NullInput;
    if (params->view_count < 3) return CalibrationError::InsufficientObservations;

    int n = params->view_count;
    int cols = params->left_corners[0].cols, rows = params->left_corners[0].rows;

    // Estimate initial stereo extrinsics from each pair, take median
    std::vector<double> all_R(9*n), all_t(3*n);
    for (int v = 0; v < n; v++) {
        // Use Zhang calibration to get board pose from each camera
        // Simplified: directly compute from K^{-1} * x = λ * X
        double Rl[9], tl[3], Rr[9], tr[3];

        // Estimate via normalized points
        double Kl_inv[9], Kr_inv[9];
        double kl_arr[9] = {params->left_intrinsics->fx,0,params->left_intrinsics->cx,
                             0,params->left_intrinsics->fy,params->left_intrinsics->cy, 0,0,1};
        double kr_arr[9] = {params->right_intrinsics->fx,0,params->right_intrinsics->cx,
                             0,params->right_intrinsics->fy,params->right_intrinsics->cy, 0,0,1};
        detail::mat3x3_inverse(kl_arr, Kl_inv);
        detail::mat3x3_inverse(kr_arr, Kr_inv);

        // Use center point to estimate board-to-camera transforms
        int ci = rows/2*cols + cols/2;
        Point2D cl = params->left_corners[v].points[ci], cr = params->right_corners[v].points[ci];
        double pln[3] = {cl.x, cl.y, 1.0}, prn[3] = {cr.x, cr.y, 1.0};
        double Pl[3], Pr_cam[3];
        detail::mat3x3_vec3_mult(Kl_inv, pln, Pl);
        detail::mat3x3_vec3_mult(Kr_inv, prn, Pr_cam);

        // Simple estimate: board is at Z=depth (e.g. 500mm), compute R≈I, t
        double depth = 500.0 / params->left_corners[0].cols; // approximate
        for (int i = 0; i < 3; i++) { tl[i] = Pl[i] * depth; tr[i] = Pr_cam[i] * depth; }
        for (int i = 0; i < 9; i++) { Rl[i] = (i%4==0)?1.0:0.0; Rr[i] = (i%4==0)?1.0:0.0; }

        // Stereo R = Rr * Rl^T, t = tr - R * tl
        double RlT[9];
        detail::mat3x3_transpose(Rl, RlT);
        detail::mat3x3_mult(Rr, RlT, all_R.data()+9*v);

        double Rtl[3];
        detail::mat3x3_vec3_mult(all_R.data()+9*v, tl, Rtl);
        all_t[3*v+0] = tr[0] - Rtl[0];
        all_t[3*v+1] = tr[1] - Rtl[1];
        all_t[3*v+2] = tr[2] - Rtl[2];
    }

    // Average rotation via quaternion
    double avg_R[9] = {1,0,0, 0,1,0, 0,0,1}, avg_t[3] = {0,0,0};
    for (int v = 0; v < n; v++) {
        for (int i = 0; i < 9; i++) avg_R[i] += all_R[9*v+i];
        for (int i = 0; i < 3; i++) avg_t[i] += all_t[3*v+i];
    }
    for (int i = 0; i < 9; i++) avg_R[i] /= (n+1);
    for (int i = 0; i < 3; i++) avg_t[i] /= n;

    // Reorthogonalize avg_R
    double s[3], V[9];
    detail::svd_3x3(avg_R, s, V);
    double Ut[9];
    detail::mat3x3_transpose(avg_R, Ut);
    double Rtmp[9];
    detail::mat3x3_mult(V, Ut, Rtmp);
    for (int i = 0; i < 9; i++) avg_R[i] = Rtmp[i];

    double sr[3], pose_rvec[30], pose_t[30]; // max 10 views
    mat_to_rvec(avg_R, sr);

    // LM refinement
    StereoLMContext ctx;
    ctx.params = params; ctx.nviews = n;
    ctx.Kl = params->left_intrinsics; ctx.Kr = params->right_intrinsics;
    ctx.cols = cols; ctx.rows = rows; ctx.ss = 1.0;

    int np = ctx.num_params(), nr = 4*n*cols*rows;
    std::vector<double> par(np);
    for (int v = 0; v < n; v++) {
        pose_rvec[v*3+0]=pose_rvec[v*3+1]=pose_rvec[v*3+2]=0;
        pose_t[v*3+0]=pose_t[v*3+1]=0; pose_t[v*3+2]=500;
    }
    ctx.pack(sr, avg_t, pose_rvec, pose_t, par.data());

    detail::levenberg_marquardt(nr, np, par.data(), stereo_residuals, stereo_jacobian, &ctx, 30, 1e-6);

    double final_sr[3], final_st[3];
    ctx.unpack(par.data(), final_sr, final_st, pose_rvec, pose_t);

    double final_R[9];
    rvec_to_mat(final_sr, final_R);
    for (int i = 0; i < 9; i++) stereo_R->rotation[i] = final_R[i];
    stereo_t->translation[0] = final_st[0];
    stereo_t->translation[1] = final_st[1];
    stereo_t->translation[2] = final_st[2];

    if (rms_error) {
        double total = 0;
        int nr_count = 0;
        for (int v = 0; v < n; v++) {
            double Rv[9]; rvec_to_mat(pose_rvec + 3*v, Rv);
            double* tv = pose_t + 3*v;
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double Pw[3] = {j * 1.0, i * 1.0, 0.0};
                    Point2D pl, pr;
                    // Left: direct projection through left intrinsics + board pose
                    detail::project_point_world(*params->left_intrinsics, Rv, tv, Pw, pl);
                    // Right: board→left_cam→stereo_transform→right_cam→project
                    double Pc[3];
                    detail::mat3x3_vec3_mult(Rv, Pw, Pc);
                    Pc[0] += tv[0]; Pc[1] += tv[1]; Pc[2] += tv[2];
                    double Pr[3];
                    detail::mat3x3_vec3_mult(final_R, Pc, Pr);
                    Pr[0] += final_st[0]; Pr[1] += final_st[1]; Pr[2] += final_st[2];
                    detail::project_point(*params->right_intrinsics, Pr, pr);
                    int idx = i*cols+j;
                    double dxl = pl.x - params->left_corners[v].points[idx].x;
                    double dyl = pl.y - params->left_corners[v].points[idx].y;
                    double dxr = pr.x - params->right_corners[v].points[idx].x;
                    double dyr = pr.y - params->right_corners[v].points[idx].y;
                    total += dxl*dxl + dyl*dyl + dxr*dxr + dyr*dyr;
                    nr += 4;
                }
            }
        }
        *rms_error = std::sqrt(total / static_cast<double>(nr_count));
    }

    return CalibrationError::Ok;
}

} // namespace calibration
