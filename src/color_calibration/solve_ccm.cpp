#include "common.hpp"
#include "color_calibration/algorithms.hpp"
#include <cstring>

namespace color_calibration {

// Solve optimal color correction matrix via linear least squares.
//
// 3x3:  For each patch i, we want M * measured_i ≈ reference_i
//       This gives 3*patch_count equations. Solve via normal equations M = R*P^T*(P*P^T)^-1
//       where P = [measured_i] (3 x N), R = [reference_i] (3 x N)
//
// 3x9:  Expand each measured into 9 features: [R, G, B, RG, RB, GB, R², G², B²]
//       Same least-squares approach with 9 features per patch.
ColorCalibrationError process_solve_ccm(const SolveCCMParams& params,
                                          SolvedMatrix* result) {
    if (!params.measured || !params.reference || !result) return ColorCalibrationError::NullInput;
    if (params.patch_count < 3) return ColorCalibrationError::InsufficientPatches;

    std::memset(result, 0, sizeof(SolvedMatrix));
    int N = params.patch_count;

    if (params.matrix_type == MatrixType::LINEAR_3X3 || params.matrix_type == MatrixType::LINEAR_4X3) {
        // --- 3x3 (or 3x4 with bias) solver ---
        // For each output channel (r, g, b), solve 3 (or 4) unknowns independently.
        // min ||X * coef - ref||² → coef = (X^T X)^-1 X^T ref

        int K = (params.matrix_type == MatrixType::LINEAR_4X3) ? 4 : 3;
        result->type = params.matrix_type;
        result->rows = 3;
        result->cols = K;

        for (int ch = 0; ch < 3; ch++) {
            // Build X^T X (K x K) and X^T y (K x 1)
            double XtX[16] = {0}; // max 4x4
            double Xty[4] = {0};

            for (int i = 0; i < N; i++) {
                double feat[4];
                feat[0] = params.measured[i].r;
                feat[1] = params.measured[i].g;
                feat[2] = params.measured[i].b;
                feat[3] = 1.0; // bias term for 4x3

                double ref;
                if (ch == 0) ref = params.reference[i].r;
                else if (ch == 1) ref = params.reference[i].g;
                else ref = params.reference[i].b;

                for (int a = 0; a < K; a++) {
                    Xty[a] += feat[a] * ref;
                    for (int b = 0; b < K; b++) {
                        XtX[a * K + b] += feat[a] * feat[b];
                    }
                }
            }

            // Regularize diagonal
            for (int a = 0; a < K; a++) XtX[a * K + a] += 1e-9;

            if (!detail::solve_linear_system(K, XtX, Xty)) {
                return ColorCalibrationError::SingularMatrix;
            }

            for (int k = 0; k < K; k++) {
                result->m[ch * K + k] = static_cast<float>(Xty[k]);
            }
        }
    } else {
        // --- 3x9 polynomial solver ---
        // Features: [R, G, B, RG, RB, GB, R², G², B²]
        const int K = 9;
        result->type = MatrixType::POLYNOMIAL_3X9;
        result->rows = 3;
        result->cols = 9;

        for (int ch = 0; ch < 3; ch++) {
            double XtX[81] = {0}; // 9x9
            double Xty[9] = {0};

            for (int i = 0; i < N; i++) {
                float r = params.measured[i].r;
                float g = params.measured[i].g;
                float b = params.measured[i].b;
                double feat[9] = {
                    r, g, b,
                    r * g, r * b, g * b,
                    r * r, g * g, b * b
                };

                double ref;
                if (ch == 0) ref = params.reference[i].r;
                else if (ch == 1) ref = params.reference[i].g;
                else ref = params.reference[i].b;

                for (int a = 0; a < K; a++) {
                    Xty[a] += feat[a] * ref;
                    for (int b = 0; b < K; b++) {
                        XtX[a * K + b] += feat[a] * feat[b];
                    }
                }
            }

            for (int a = 0; a < K; a++) XtX[a * K + a] += 1e-9;

            if (!detail::solve_linear_system(K, XtX, Xty)) {
                return ColorCalibrationError::SingularMatrix;
            }

            for (int k = 0; k < K; k++) {
                result->m[ch * K + k] = static_cast<float>(Xty[k]);
            }
        }
    }

    return ColorCalibrationError::Ok;
}

} // namespace color_calibration
