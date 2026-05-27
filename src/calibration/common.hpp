#ifndef CALIBRATION_COMMON_HPP
#define CALIBRATION_COMMON_HPP

#include "calibration/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace calibration {
namespace detail {

// ============================================================
//  Basic vector / matrix operations
// ============================================================

inline double dot3(const double a[3], const double b[3]) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline void cross3(const double a[3], const double b[3], double result[3]) {
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

inline double norm3(const double v[3]) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

inline void normalize3(double v[3]) {
    double n = norm3(v);
    if (n > 1e-30) {
        v[0] /= n;
        v[1] /= n;
        v[2] /= n;
    }
}

inline void mat3x3_vec3_mult(const double M[9], const double v[3], double result[3]) {
    result[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
    result[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
    result[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];
}

inline void mat3x3_transpose_vec3_mult(const double M[9], const double v[3], double result[3]) {
    result[0] = M[0] * v[0] + M[3] * v[1] + M[6] * v[2];
    result[1] = M[1] * v[0] + M[4] * v[1] + M[7] * v[2];
    result[2] = M[2] * v[0] + M[5] * v[1] + M[8] * v[2];
}

inline void mat3x3_transpose(const double M[9], double result[9]) {
    result[0] = M[0]; result[1] = M[3]; result[2] = M[6];
    result[3] = M[1]; result[4] = M[4]; result[5] = M[7];
    result[6] = M[2]; result[7] = M[5]; result[8] = M[8];
}

inline void mat3x3_mult(const double A[9], const double B[9], double result[9]) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += A[3 * i + k] * B[3 * k + j];
            }
            result[3 * i + j] = sum;
        }
    }
}

// 3x3 matrix inverse using cofactor expansion
inline bool mat3x3_inverse(const double M[9], double inv[9]) {
    double det = M[0] * (M[4] * M[8] - M[5] * M[7])
               - M[1] * (M[3] * M[8] - M[5] * M[6])
               + M[2] * (M[3] * M[7] - M[4] * M[6]);

    if (std::abs(det) < 1e-30) return false;

    double inv_det = 1.0 / det;
    inv[0] = (M[4] * M[8] - M[5] * M[7]) * inv_det;
    inv[1] = (M[2] * M[7] - M[1] * M[8]) * inv_det;
    inv[2] = (M[1] * M[5] - M[2] * M[4]) * inv_det;
    inv[3] = (M[5] * M[6] - M[3] * M[8]) * inv_det;
    inv[4] = (M[0] * M[8] - M[2] * M[6]) * inv_det;
    inv[5] = (M[2] * M[3] - M[0] * M[5]) * inv_det;
    inv[6] = (M[3] * M[7] - M[4] * M[6]) * inv_det;
    inv[7] = (M[1] * M[6] - M[0] * M[7]) * inv_det;
    inv[8] = (M[0] * M[4] - M[1] * M[3]) * inv_det;
    return true;
}

// ============================================================
//  Rodrigues: axis-angle <-> rotation matrix
// ============================================================

inline void rodrigues_to_matrix(const double r[3], double R[9]) {
    double theta = norm3(r);
    if (theta < 1e-30) {
        R[0] = 1.0; R[1] = 0.0; R[2] = 0.0;
        R[3] = 0.0; R[4] = 1.0; R[5] = 0.0;
        R[6] = 0.0; R[7] = 0.0; R[8] = 1.0;
        return;
    }
    double u[3] = {r[0] / theta, r[1] / theta, r[2] / theta};
    double c = std::cos(theta);
    double s = std::sin(theta);
    double cc = 1.0 - c;

    R[0] = u[0] * u[0] * cc + c;
    R[1] = u[0] * u[1] * cc - u[2] * s;
    R[2] = u[0] * u[2] * cc + u[1] * s;
    R[3] = u[1] * u[0] * cc + u[2] * s;
    R[4] = u[1] * u[1] * cc + c;
    R[5] = u[1] * u[2] * cc - u[0] * s;
    R[6] = u[2] * u[0] * cc - u[1] * s;
    R[7] = u[2] * u[1] * cc + u[0] * s;
    R[8] = u[2] * u[2] * cc + c;
}

// ============================================================
//  Camera projection: 3D world point -> 2D image point
// ============================================================

// Project a 3D point (in camera coordinates) through pinhole model with distortion.
// K: intrinsics (fx, fy, cx, cy, k1, k2, k3, p1, p2)
// P: 3D point in camera coordinates
// Returns projected 2D image point
inline void project_point(const CameraIntrinsics& K, const double P[3], Point2D& p) {
    // Perspective division
    double xn = P[0] / P[2];  // Normalized x
    double yn = P[1] / P[2];  // Normalized y

    // Apply distortion
    double r2 = xn * xn + yn * yn;
    double r4 = r2 * r2;
    double r6 = r4 * r2;

    double radial = 1.0 + K.k1 * r2 + K.k2 * r4 + K.k3 * r6;
    double dx = 2.0 * K.p1 * xn * yn + K.p2 * (r2 + 2.0 * xn * xn);
    double dy = K.p1 * (r2 + 2.0 * yn * yn) + 2.0 * K.p2 * xn * yn;

    double xd = xn * radial + dx;
    double yd = yn * radial + dy;

    // To pixel coordinates
    p.x = K.fx * xd + K.cx;
    p.y = K.fy * yd + K.cy;
}

// Project with explicit world-to-camera transform (R, t)
inline void project_point_world(const CameraIntrinsics& K, const double R[9],
                                 const double t[3], const double Pw[3], Point2D& p) {
    // Transform to camera coordinates: Pc = R * Pw + t
    double Pc[3];
    mat3x3_vec3_mult(R, Pw, Pc);
    Pc[0] += t[0];
    Pc[1] += t[1];
    Pc[2] += t[2];

    project_point(K, Pc, p);
}

// ============================================================
//  Reprojection error computation
// ============================================================

inline double compute_reprojection_error(const CameraIntrinsics& K, const double R[9],
                                          const double t[3], const double rod_points[3][3],
                                          const RodObservation& obs) {
    double sum_sq = 0.0;
    double rod_pts[3][3] = {
        {rod_points[0][0], rod_points[0][1], rod_points[0][2]},
        {rod_points[1][0], rod_points[1][1], rod_points[1][2]},
        {rod_points[2][0], rod_points[2][1], rod_points[2][2]}
    };
    Point2D proj[3];
    project_point_world(K, R, t, rod_pts[0], proj[0]);
    project_point_world(K, R, t, rod_pts[1], proj[1]);
    project_point_world(K, R, t, rod_pts[2], proj[2]);

    double dx = proj[0].x - obs.marker_a.x;
    double dy = proj[0].y - obs.marker_a.y;
    sum_sq += dx * dx + dy * dy;

    dx = proj[1].x - obs.marker_b.x;
    dy = proj[1].y - obs.marker_b.y;
    sum_sq += dx * dx + dy * dy;

    dx = proj[2].x - obs.marker_c.x;
    dy = proj[2].y - obs.marker_c.y;
    sum_sq += dx * dx + dy * dy;

    return sum_sq;
}

// ============================================================
//  Linear system solver (Gaussian elimination with partial pivoting)
// ============================================================

// Solve A * x = b, where A is n x n, b is n x 1.
// A and b are modified in-place. Solution stored in b.
inline bool solve_linear_system(int n, double* A, double* b) {
    // Forward elimination with partial pivoting
    for (int col = 0; col < n; ++col) {
        // Find pivot
        int max_row = col;
        double max_val = std::abs(A[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            double val = std::abs(A[row * n + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }
        if (max_val < 1e-30) return false;

        // Swap rows
        if (max_row != col) {
            for (int j = col; j < n; ++j) {
                std::swap(A[col * n + j], A[max_row * n + j]);
            }
            std::swap(b[col], b[max_row]);
        }

        // Eliminate
        double pivot = A[col * n + col];
        for (int row = col + 1; row < n; ++row) {
            double factor = A[row * n + col] / pivot;
            A[row * n + col] = 0.0;
            for (int j = col + 1; j < n; ++j) {
                A[row * n + j] -= factor * A[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[i * n + j] * b[j];
        }
        if (std::abs(A[i * n + i]) < 1e-30) return false;
        b[i] = sum / A[i * n + i];
    }
    return true;
}

// ============================================================
//  Levenberg-Marquardt optimizer
// ============================================================

// A generic LM solver for problems of the form: minimize ||f(params)||^2
//
// num_residuals: number of residuals (observations)
// num_params: number of parameters to optimize
// initial_params: starting parameter values
// compute_residuals: callback to compute residual vector given current params
// compute_jacobian: callback to compute Jacobian matrix (num_residuals x num_params)
//
// Returns true if converged, false if failed.
// The optimized parameters are stored in params.

inline bool levenberg_marquardt(int num_residuals, int num_params,
                                 double* params,
                                 void (*compute_residuals)(const double* p, double* residuals, void* ctx),
                                 void (*compute_jacobian)(const double* p, double* J, void* ctx),
                                 void* ctx, int max_iter, double tol) {
    std::vector<double> residuals(num_residuals);
    std::vector<double> J(num_residuals * num_params);
    std::vector<double> JtJ(num_params * num_params);
    std::vector<double> JtR(num_params);
    std::vector<double> delta(num_params);

    double lambda = 1e-3;
    const double lambda_up = 10.0;
    const double lambda_down = 0.1;

    compute_residuals(params, residuals.data(), ctx);
    double cost = 0.0;
    for (int i = 0; i < num_residuals; ++i) {
        cost += residuals[i] * residuals[i];
    }
    cost *= 0.5;

    std::vector<double> new_params(num_params);
    std::vector<double> new_residuals(num_residuals);

    for (int iter = 0; iter < max_iter; ++iter) {
        compute_jacobian(params, J.data(), ctx);

        // Build J^T * J and J^T * r
        for (int i = 0; i < num_params; ++i) {
            JtR[i] = 0.0;
            for (int j = 0; j < num_residuals; ++j) {
                JtR[i] += J[j * num_params + i] * residuals[j];
            }
            for (int k = 0; k < num_params; ++k) {
                double sum = 0.0;
                for (int j = 0; j < num_residuals; ++j) {
                    sum += J[j * num_params + i] * J[j * num_params + k];
                }
                JtJ[i * num_params + k] = sum;
            }
        }

        // Augmented normal equations: (J^T * J + lambda * diag(J^T*J)) * delta = -J^T * r
        std::vector<double> A = JtJ;  // Copy
        for (int i = 0; i < num_params; ++i) {
            A[i * num_params + i] += lambda * std::max(1.0, JtJ[i * num_params + i]);
        }

        std::vector<double> b(num_params);
        for (int i = 0; i < num_params; ++i) {
            b[i] = -JtR[i];
        }

        if (!solve_linear_system(num_params, A.data(), b.data())) {
            lambda *= lambda_up;
            continue;
        }

        // Copy delta
        for (int i = 0; i < num_params; ++i) {
            delta[i] = b[i];
        }

        // Trial step
        for (int i = 0; i < num_params; ++i) {
            new_params[i] = params[i] + delta[i];
        }

        compute_residuals(new_params.data(), new_residuals.data(), ctx);
        double new_cost = 0.0;
        for (int i = 0; i < num_residuals; ++i) {
            new_cost += new_residuals[i] * new_residuals[i];
        }
        new_cost *= 0.5;

        // Compute predicted reduction
        double pred = 0.0;
        for (int i = 0; i < num_params; ++i) {
            pred += delta[i] * (lambda * std::max(1.0, JtJ[i * num_params + i]) * delta[i] - JtR[i]);
        }
        pred *= 0.5;
        if (pred < 1e-30) pred = 1e-30;

        double rho = (cost - new_cost) / pred;

        if (rho > 0.0) {
            // Accept step
            for (int i = 0; i < num_params; ++i) {
                params[i] = new_params[i];
            }
            residuals.swap(new_residuals);

            // Check convergence before updating cost: stop when cost reduction is negligible
            double old_cost = cost;
            cost = new_cost;
            lambda = std::max(1e-15, lambda * std::max(lambda_down, 1.0 - std::pow(2.0 * rho - 1.0, 3.0)));

            if (old_cost - new_cost < tol * std::max(1.0, old_cost)) {
                return true;
            }

            // Also check step size
            double delta_norm = 0.0;
            for (int i = 0; i < num_params; ++i) {
                delta_norm += delta[i] * delta[i];
            }
            if (std::sqrt(delta_norm) < tol) {
                return true;
            }
        } else {
            // Reject step
            lambda *= lambda_up;
        }
    }

    // Check final gradient
    compute_jacobian(params, J.data(), ctx);
    for (int i = 0; i < num_params; ++i) {
        JtR[i] = 0.0;
        for (int j = 0; j < num_residuals; ++j) {
            JtR[i] += J[j * num_params + i] * residuals[j];
        }
    }
    double grad_max = 0.0;
    for (int i = 0; i < num_params; ++i) {
        grad_max = std::max(grad_max, std::abs(JtR[i]));
    }
    if (grad_max < tol) return true;

    return false;  // Max iterations reached
}

// ============================================================
//  SVD for small matrices (Jacobi method)
// ============================================================

// Compute SVD A = U * S * V^T for a 3x3 matrix.
// A is overwritten with U. S is diagonal (stored as vector). V is output.
inline void svd_3x3(double A[9], double S[3], double V[9]) {
    // Initialize V = identity
    V[0] = 1.0; V[1] = 0.0; V[2] = 0.0;
    V[3] = 0.0; V[4] = 1.0; V[5] = 0.0;
    V[6] = 0.0; V[7] = 0.0; V[8] = 1.0;

    // A^T * A (compute once for diagonalization)
    double ATA[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += A[k * 3 + i] * A[k * 3 + j];  // A^T * A
            }
            ATA[i * 3 + j] = sum;
        }
    }

    // Jacobi diagonalization of ATA (fixed small size)
    for (int sweep = 0; sweep < 20; ++sweep) {
        // Find largest off-diagonal
        double max_off = 0.0;
        int p = 0, q = 1;
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 3; ++j) {
                double val = std::abs(ATA[i * 3 + j]);
                if (val > max_off) {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off < 1e-15) break;

        // Compute Jacobi rotation
        double theta = 0.5 * std::atan2(2.0 * ATA[p * 3 + q], ATA[q * 3 + q] - ATA[p * 3 + p]);
        double c = std::cos(theta);
        double s = std::sin(theta);

        // Apply rotation to ATA and V
        // Rows p,q of ATA
        double aip, aiq;
        for (int i = 0; i < 3; ++i) {
            aip = ATA[i * 3 + p];
            aiq = ATA[i * 3 + q];
            ATA[i * 3 + p] = c * aip - s * aiq;
            ATA[i * 3 + q] = s * aip + c * aiq;
        }
        // Columns p,q
        for (int j = 0; j < 3; ++j) {
            aip = ATA[p * 3 + j];
            aiq = ATA[q * 3 + j];
            ATA[p * 3 + j] = c * aip - s * aiq;
            ATA[q * 3 + j] = s * aip + c * aiq;
        }
        // Update V
        for (int i = 0; i < 3; ++i) {
            aip = V[i * 3 + p];
            aiq = V[i * 3 + q];
            V[i * 3 + p] = c * aip - s * aiq;
            V[i * 3 + q] = s * aip + c * aiq;
        }
    }

    // Extract singular values (eigenvalues of ATA are sigma^2)
    for (int i = 0; i < 3; ++i) {
        double val = ATA[i * 3 + i];
        S[i] = val > 0.0 ? std::sqrt(val) : 0.0;
    }

    // Compute U = A * V * diag(1/S)
    // First compute A * V
    double AV[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += A[i * 3 + k] * V[k * 3 + j];
            }
            AV[i * 3 + j] = sum;
        }
    }

    // Scale by 1/S to get columns of U
    for (int j = 0; j < 3; ++j) {
        double inv_s = S[j] > 1e-30 ? 1.0 / S[j] : 0.0;
        for (int i = 0; i < 3; ++i) {
            A[i * 3 + j] = AV[i * 3 + j] * inv_s;
        }
    }
}

}  // namespace detail
}  // namespace calibration

#endif  // CALIBRATION_COMMON_HPP
