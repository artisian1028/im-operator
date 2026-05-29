#include "common.hpp"
#include "calibration/algorithms.hpp"
#include <vector>
#include <cmath>
#include <cstring>

namespace calibration {

// ============================================================
//  Zhang's camera calibration from multiple checkerboard views
//
//  Steps:
//  1. For each view, compute homography between board plane (Z=0) and image
//  2. From homographies, build linear constraints on B = K^{-T} K^{-1}
//  3. Solve for B (6-vector) → decompose to K via Cholesky
//  4. Compute per-view extrinsics from K and homography
//  5. LM optimization over K + distortion + all extrinsics
// ============================================================

namespace {

// Normalize points for better numerical conditioning
void normalize_points(const Point2D* pts, int n,
                      Point2D* norm, double T[9]) {
    double mx = 0, my = 0;
    for (int i = 0; i < n; i++) { mx += pts[i].x; my += pts[i].y; }
    mx /= n; my /= n;

    double scale = 0;
    for (int i = 0; i < n; i++) {
        double dx = pts[i].x - mx, dy = pts[i].y - my;
        scale += std::sqrt(dx*dx + dy*dy);
    }
    scale = std::sqrt(2.0) * n / scale;

    T[0] = scale; T[1] = 0;     T[2] = -scale * mx;
    T[3] = 0;     T[4] = scale; T[5] = -scale * my;
    T[6] = 0;     T[7] = 0;     T[8] = 1;

    for (int i = 0; i < n; i++) {
        norm[i].x = T[0] * pts[i].x + T[1] * pts[i].y + T[2];
        norm[i].y = T[3] * pts[i].x + T[4] * pts[i].y + T[5];
    }
}

// Compute normalized DLT homography
// Board points: (j*square_size, i*square_size, 0) → image corners (row i, col j)
bool compute_homography(const Point2D* img_pts, int cols, int rows,
                         double square_size, double H[9]) {
    int n = cols * rows;
    std::vector<Point2D> world(n), img_norm(n);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            world[i*cols+j].x = j * square_size;
            world[i*cols+j].y = i * square_size;
        }

    double T_img[9], T_world[9];
    std::vector<Point2D> w_norm(n);
    normalize_points(img_pts, n, img_norm.data(), T_img);
    normalize_points(world.data(), n, w_norm.data(), T_world);

    // Build A: each point gives 2 rows
    std::vector<double> A(static_cast<size_t>(2*n) * 9, 0.0);
    for (int i = 0; i < n; i++) {
        double x = w_norm[i].x, y = w_norm[i].y;
        double u = img_norm[i].x, v = img_norm[i].y;
        A[2*i*9 + 0] = x; A[2*i*9 + 1] = y; A[2*i*9 + 2] = 1;
        A[2*i*9 + 6] = -u*x; A[2*i*9 + 7] = -u*y; A[2*i*9 + 8] = -u;
        A[(2*i+1)*9 + 3] = x; A[(2*i+1)*9 + 4] = y; A[(2*i+1)*9 + 5] = 1;
        A[(2*i+1)*9 + 6] = -v*x; A[(2*i+1)*9 + 7] = -v*y; A[(2*i+1)*9 + 8] = -v;
    }

    // Solve via SVD of 2n×9 matrix
    // ATA * h = λ * h → smallest eigenvector
    double ATA[81] = {0};
    for (int i = 0; i < 2*n; i++)
        for (int a = 0; a < 9; a++)
            for (int b = 0; b < 9; b++)
                ATA[a*9+b] += A[i*9+a] * A[i*9+b];

    // Inverse power iteration for smallest eigenvalue of ATA
    double h[9];
    for (int i = 0; i < 9; i++) h[i] = (i == 8) ? 1.0 : 0.0;
    for (int iter = 0; iter < 100; iter++) {
        double ATA_reg[81];
        for (int i = 0; i < 81; i++) ATA_reg[i] = ATA[i];
        for (int i = 0; i < 9; i++) ATA_reg[i*9+i] += 1e-10;
        double rhs[9]; for (int i = 0; i < 9; i++) rhs[i] = h[i];
        if (!detail::solve_linear_system(9, ATA_reg, rhs)) break;
        double norm = 0;
        for (int i = 0; i < 9; i++) norm += rhs[i]*rhs[i];
        norm = std::sqrt(norm);
        if (norm < 1e-30) break;
        for (int i = 0; i < 9; i++) h[i] = rhs[i] / norm;
    }

    // Denormalize: H = T_img^{-1} * H_norm * T_world
    double T_img_inv[9];
    detail::mat3x3_inverse(T_img, T_img_inv);

    double Hn[9] = {h[0],h[1],h[2], h[3],h[4],h[5], h[6],h[7],h[8]};
    double H_temp[9];
    detail::mat3x3_mult(T_img_inv, Hn, H_temp);
    detail::mat3x3_mult(H_temp, T_world, H);

    return true;
}

// Extract intrinsics from homography set via Zhang's method
bool extract_intrinsics(const std::vector<double*>& homographies, int nviews,
                         CameraIntrinsics& K) {
    // Build constraints: V * b = 0
    // Each homography gives 2 equations (h1^T B h2 = 0, h1^T B h1 = h2^T B h2)
    std::vector<double> V(static_cast<size_t>(2*nviews) * 6, 0.0);

    for (int v = 0; v < nviews; v++) {
        double* H = homographies[v];
        double h11=H[0], h12=H[1], h13=H[2];
        double h21=H[3], h22=H[4], h23=H[5];
        double h31=H[6], h32=H[7], h33=H[8];

        // v12 = [h11*h12, h11*h22+h21*h12, h21*h22, h31*h12+h11*h32, h31*h22+h21*h32, h31*h32]
        double v12[6] = {
            h11*h12,
            h11*h22 + h21*h12,
            h21*h22,
            h31*h12 + h11*h32,
            h31*h22 + h21*h32,
            h31*h32
        };
        // v11 - v22
        double v11_v22[6] = {
            h11*h11 - h12*h12,
            2*(h11*h21 - h12*h22),
            h21*h21 - h22*h22,
            2*(h31*h11 - h32*h12),
            2*(h31*h21 - h32*h22),
            h31*h31 - h32*h32
        };
        for (int j = 0; j < 6; j++) {
            V[2*v*6+j] = v12[j];
            V[(2*v+1)*6+j] = v11_v22[j];
        }
    }

    // Solve V^T V * b = λ * b via power iteration on smallest eigenvector
    double VTV[36] = {0};
    for (int i = 0; i < 2*nviews; i++)
        for (int a = 0; a < 6; a++)
            for (int b = 0; b < 6; b++)
                VTV[a*6+b] += V[i*6+a] * V[i*6+b];

    double b_vec[6];
    for (int i = 0; i < 6; i++) b_vec[i] = (i == 5) ? 1.0 : 0.0;
    for (int iter = 0; iter < 200; iter++) {
        double VTV_reg[36];
        for (int i = 0; i < 36; i++) VTV_reg[i] = VTV[i];
        for (int i = 0; i < 6; i++) VTV_reg[i*6+i] += 1e-10;
        double rhs[6]; for (int i = 0; i < 6; i++) rhs[i] = b_vec[i];
        if (!detail::solve_linear_system(6, VTV_reg, rhs)) break;
        double n = 0;
        for (int i = 0; i < 6; i++) n += rhs[i]*rhs[i];
        n = std::sqrt(n);
        if (n < 1e-30) break;
        for (int i = 0; i < 6; i++) b_vec[i] = rhs[i] / n;
    }

    double B11=b_vec[0], B12=b_vec[1], B22=b_vec[2], B13=b_vec[3], B23=b_vec[4], B33=b_vec[5];

    // Extract K from B = K^{-T} K^{-1}
    double v0 = (B12*B13 - B11*B23) / (B11*B22 - B12*B12 + 1e-30);
    double lambda = B33 - (B13*B13 + v0*(B12*B13 - B11*B23)) / (B11 + 1e-30);
    if (lambda <= 0) return false;
    double alpha = std::sqrt(lambda / (B11 + 1e-30));
    double beta  = std::sqrt(lambda * B11 / (B11*B22 - B12*B12 + 1e-30));
    double gamma = -B12 * alpha * alpha * beta / lambda;
    double u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda;

    K.fx = alpha; K.fy = beta;
    K.cx = u0; K.cy = v0;
    return true;
}

// Compute extrinsics from K and H
void extract_extrinsics(const CameraIntrinsics& K, const double H[9],
                        double R[9], double t[3]) {
    double K_inv[9];
    double k[9] = {K.fx, 0, K.cx, 0, K.fy, K.cy, 0, 0, 1};
    detail::mat3x3_inverse(k, K_inv);

    // R1 = λ * K^{-1} * h1, R2 = λ * K^{-1} * h2, R3 = R1 × R2
    double h1[3] = {H[0], H[3], H[6]};
    double h2[3] = {H[1], H[4], H[7]};
    double h3[3] = {H[2], H[5], H[8]};

    double r1[3], r2[3];
    detail::mat3x3_vec3_mult(K_inv, h1, r1);
    detail::mat3x3_vec3_mult(K_inv, h2, r2);

    double lambda = 1.0 / (detail::norm3(r1) + 1e-30);
    for (int i = 0; i < 3; i++) { r1[i] *= lambda; r2[i] *= lambda; }

    double r3[3];
    detail::cross3(r1, r2, r3);
    detail::normalize3(r3);

    // Reorthogonalize: SVD of [r1 r2 r3]
    // Simplified: Gram-Schmidt
    detail::cross3(r2, r3, r1); detail::normalize3(r1);
    detail::cross3(r3, r1, r2); detail::normalize3(r2);

    R[0]=r1[0]; R[1]=r2[0]; R[2]=r3[0];
    R[3]=r1[1]; R[4]=r2[1]; R[5]=r3[1];
    R[6]=r1[2]; R[7]=r2[2]; R[8]=r3[2];

    double t_n[3];
    detail::mat3x3_vec3_mult(K_inv, h3, t_n);
    for (int i = 0; i < 3; i++) t[i] = t_n[i] * lambda;
}

// LM optimization context for full calibration
struct CalibLMContext {
    const CheckerboardCorners* views;
    int nviews;
    int cols, rows;
    double square_size;
    bool estimate_distortion;
    int image_width, image_height;

    int num_params() const {
        int n = 4; // fx, fy, cx, cy
        if (estimate_distortion) n += 5; // k1,k2,k3,p1,p2
        n += 6 * nviews; // per-view rodrigues(3) + t(3)
        return n;
    }

    void pack(const CameraIntrinsics& K, const double* R, const double* t, double* p) const {
        int idx = 0;
        p[idx++] = K.fx; p[idx++] = K.fy; p[idx++] = K.cx; p[idx++] = K.cy;
        if (estimate_distortion) {
            p[idx++] = K.k1; p[idx++] = K.k2; p[idx++] = K.k3;
            p[idx++] = K.p1; p[idx++] = K.p2;
        }
        for (int v = 0; v < nviews; v++) {
            const double* Rv = R + 9*v;
            double trace = Rv[0]+Rv[4]+Rv[8];
            double cos_t = std::clamp((trace-1.0)*0.5, -1.0, 1.0);
            double theta = std::acos(cos_t);
            if (theta < 1e-10) { p[idx++]=0; p[idx++]=0; p[idx++]=0; }
            else {
                double factor = theta / (2.0*std::sin(theta));
                p[idx++] = factor*(Rv[7]-Rv[5]);
                p[idx++] = factor*(Rv[2]-Rv[6]);
                p[idx++] = factor*(Rv[3]-Rv[1]);
            }
            p[idx++] = t[3*v+0]; p[idx++] = t[3*v+1]; p[idx++] = t[3*v+2];
        }
    }

    void unpack(const double* p, CameraIntrinsics& K, double* R, double* t) const {
        int idx = 0;
        K.fx = p[idx++]; K.fy = p[idx++]; K.cx = p[idx++]; K.cy = p[idx++];
        if (estimate_distortion) {
            K.k1=p[idx++]; K.k2=p[idx++]; K.k3=p[idx++]; K.p1=p[idx++]; K.p2=p[idx++];
        }
        for (int v = 0; v < nviews; v++) {
            double r[3] = {p[idx], p[idx+1], p[idx+2]}; idx += 3;
            detail::rodrigues_to_matrix(r, R + 9*v);
            t[3*v+0]=p[idx++]; t[3*v+1]=p[idx++]; t[3*v+2]=p[idx++];
        }
    }
};

void calib_residuals(const double* params, double* res, void* ctx) {
    auto* c = static_cast<CalibLMContext*>(ctx);
    CameraIntrinsics K;
    std::vector<double> R(9*c->nviews), t(3*c->nviews);
    c->unpack(params, K, R.data(), t.data());

    int res_idx = 0;
    for (int v = 0; v < c->nviews; v++) {
        const auto& view = c->views[v];
        for (int i = 0; i < c->rows; i++) {
            for (int j = 0; j < c->cols; j++) {
                double Pw[3] = {j*c->square_size, i*c->square_size, 0.0};
                Point2D proj;
                detail::project_point_world(K, R.data()+9*v, t.data()+3*v, Pw, proj);
                int idx = i*c->cols+j;
                res[res_idx++] = proj.x - view.points[idx].x;
                res[res_idx++] = proj.y - view.points[idx].y;
            }
        }
    }
}

void calib_jacobian(const double* params, double* J, void* ctx) {
    auto* c = static_cast<CalibLMContext*>(ctx);
    int np = c->num_params();
    int nr = 2 * c->nviews * c->cols * c->rows;
    std::vector<double> pp(np), pm(np), rp(nr), rm(nr);

    for (int i = 0; i < np; i++) {
        for (int j = 0; j < np; j++) pp[j] = pm[j] = params[j];
        double eps = 1e-6;
        pp[i] += eps; pm[i] -= eps;
        calib_residuals(pp.data(), rp.data(), ctx);
        calib_residuals(pm.data(), rm.data(), ctx);
        for (int j = 0; j < nr; j++) J[j*np+i] = (rp[j]-rm[j])/(2.0*eps);
    }
}

} // anonymous namespace

CalibrationError process_checkerboard_calibrate(const CheckerboardCorners* corners,
                                                 int view_count,
                                                 int image_width, int image_height,
                                                 const CheckerboardConfig* config,
                                                 CameraIntrinsics* intrinsics,
                                                 CameraExtrinsics* extrinsics,
                                                 bool estimate_distortion) {
    if (!corners || !intrinsics || !config) return CalibrationError::NullInput;
    if (view_count < 3) return CalibrationError::InsufficientObservations;
    if (!extrinsics) return CalibrationError::NullInput;

    int cols = config->cols, rows = config->rows;
    double ss = config->square_size;

    // Step 1 & 2: Compute homographies & extract intrinsics
    std::vector<double*> Hs(view_count);
    std::vector<std::vector<double>> H_data(view_count, std::vector<double>(9));
    for (int v = 0; v < view_count; v++) {
        H_data[v].assign(9, 0.0);
        Hs[v] = H_data[v].data();
        if (!compute_homography(corners[v].points, cols, rows, ss, Hs[v]))
            return CalibrationError::SingularMatrix;
    }

    CameraIntrinsics K;
    K.fx = static_cast<double>(std::max(image_width, image_height));
    K.fy = K.fx;
    K.cx = image_width * 0.5;
    K.cy = image_height * 0.5;
    K.k1 = K.k2 = K.k3 = K.p1 = K.p2 = 0.0;

    if (!extract_intrinsics(Hs, view_count, K))
        return CalibrationError::SingularMatrix;

    // Step 3: Compute per-view extrinsics
    std::vector<double> R(9*view_count), t(3*view_count);
    for (int v = 0; v < view_count; v++)
        extract_extrinsics(K, Hs[v], R.data()+9*v, t.data()+3*v);

    // Step 4: LM optimization
    CalibLMContext ctx;
    ctx.views = corners;
    ctx.nviews = view_count;
    ctx.cols = cols; ctx.rows = rows;
    ctx.square_size = ss;
    ctx.estimate_distortion = estimate_distortion;
    ctx.image_width = image_width;
    ctx.image_height = image_height;

    int np = ctx.num_params();
    std::vector<double> params(np);
    ctx.pack(K, R.data(), t.data(), params.data());

    int nr = 2 * view_count * cols * rows;
    detail::levenberg_marquardt(nr, np, params.data(), calib_residuals, calib_jacobian,
                                 &ctx, 50, 1e-6);

    ctx.unpack(params.data(), K, R.data(), t.data());
    *intrinsics = K;

    // Copy extrinsics for the first view (or average)
    for (int i = 0; i < 9; i++) extrinsics[0].rotation[i] = R[i];
    extrinsics[0].translation[0] = t[0];
    extrinsics[0].translation[1] = t[1];
    extrinsics[0].translation[2] = t[2];

    // Also fill remaining views
    for (int v = 1; v < view_count && v < 10 && extrinsics; v++) {
        for (int i = 0; i < 9; i++) extrinsics[v].rotation[i] = R[9*v+i];
        extrinsics[v].translation[0] = t[3*v+0];
        extrinsics[v].translation[1] = t[3*v+1];
        extrinsics[v].translation[2] = t[3*v+2];
    }

    return CalibrationError::Ok;
}

} // namespace calibration
