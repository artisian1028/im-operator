#include "common.hpp"
#include "calibration/algorithms.hpp"
#include <vector>
#include <cmath>

namespace calibration {

// ============================================================
//  DLT: Direct Linear Transform
//
//  N 3D↔2D correspondences → 3×4 camera matrix P
//  For each correspondence (X, x): 2 equations in 12 unknowns
//  Solve A * p = 0 via SVD, p = reshaped to 3×4
//  Then decompose P = K[R|t] via RQ decomposition
// ============================================================

namespace {

// RQ decomposition: A = R * Q where R is upper-triangular, Q is orthonormal
// This is done by QR on reversed rows/cols then reversing back
void rq_decompose(const double A[9], double R[9], double Q[9]) {
    // RQ = QR of reversed matrix
    double Ar[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            Ar[i*3+j] = A[(2-i)*3+(2-j)]; // reverse both rows and cols

    // Gram-Schmidt QR on Ar
    double q[9] = {0};
    for (int c = 0; c < 3; c++) {
        for (int r = 0; r < 3; r++) q[r*3+c] = Ar[r*3+c];
        for (int prev = 0; prev < c; prev++) {
            double dot = 0;
            for (int r = 0; r < 3; r++) dot += q[r*3+prev] * Ar[r*3+c];
            for (int r = 0; r < 3; r++) q[r*3+c] -= dot * q[r*3+prev];
        }
        double n = 0;
        for (int r = 0; r < 3; r++) n += q[r*3+c] * q[r*3+c];
        n = std::sqrt(n);
        if (n > 1e-30) for (int r = 0; r < 3; r++) q[r*3+c] /= n;
    }
    // Build R from Ar and Q
    double Qt[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            Qt[i*3+j] = q[j*3+i]; // transpose q

    double Rr[9] = {0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                Rr[i*3+j] += Ar[i*3+k] * q[k*3+j];

    // Reverse back
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            R[i*3+j] = Rr[(2-i)*3+(2-j)];
            Q[i*3+j] = q[(2-i)*3+(2-j)];
        }

    // Normalize R so R[2][2] > 0
    if (R[8] < 0) {
        for (int i = 0; i < 9; i++) { R[i] = -R[i]; Q[i] = -Q[i]; }
    }
}

} // anonymous namespace

CalibrationError process_dlt(const DltParams* params, DltResult* result) {
    if (!params || !result) return CalibrationError::NullInput;
    if (!params->correspondences || params->count < 6)
        return CalibrationError::InsufficientObservations;

    int n = params->count;
    std::vector<double> A(static_cast<size_t>(2*n) * 12, 0.0);

    // Normalize points
    double mx=0, my=0, MX=0, MY=0, MZ=0;
    for (int i = 0; i < n; i++) {
        mx += params->correspondences[i].image_x;
        my += params->correspondences[i].image_y;
        MX += params->correspondences[i].world_x;
        MY += params->correspondences[i].world_y;
        MZ += params->correspondences[i].world_z;
    }
    mx /= n; my /= n; MX /= n; MY /= n; MZ /= n;

    double simg = 0, sworld = 0;
    for (int i = 0; i < n; i++) {
        double di = std::sqrt(std::pow(params->correspondences[i].image_x-mx,2.0) +
                              std::pow(params->correspondences[i].image_y-my,2.0));
        double dw = std::sqrt(std::pow(params->correspondences[i].world_x-MX,2.0) +
                              std::pow(params->correspondences[i].world_y-MY,2.0) +
                              std::pow(params->correspondences[i].world_z-MZ,2.0));
        simg += di; sworld += dw;
    }
    simg = std::sqrt(2.0) * n / simg;
    sworld = std::sqrt(3.0) * n / sworld;

    // Build normalized A matrix
    for (int i = 0; i < n; i++) {
        double x = (params->correspondences[i].image_x - mx) * simg;
        double y = (params->correspondences[i].image_y - my) * simg;
        double X = (params->correspondences[i].world_x - MX) * sworld;
        double Y = (params->correspondences[i].world_y - MY) * sworld;
        double Z = (params->correspondences[i].world_z - MZ) * sworld;

        // x = P * X → cross product form
        A[2*i*12 + 0] = X; A[2*i*12 + 1] = Y; A[2*i*12 + 2] = Z; A[2*i*12 + 3] = 1;
        A[2*i*12 + 8] = -x*X; A[2*i*12 + 9] = -x*Y; A[2*i*12 + 10] = -x*Z; A[2*i*12 + 11] = -x;

        A[(2*i+1)*12 + 4] = X; A[(2*i+1)*12 + 5] = Y; A[(2*i+1)*12 + 6] = Z; A[(2*i+1)*12 + 7] = 1;
        A[(2*i+1)*12 + 8] = -y*X; A[(2*i+1)*12 + 9] = -y*Y; A[(2*i+1)*12 + 10] = -y*Z; A[(2*i+1)*12 + 11] = -y;
    }

    // Solve A^T A * p = 0 via inverse power iteration (finds smallest eigenvector)
    // Use a small regularization to make A^T A invertible
    double ATA[144] = {0};
    for (int i = 0; i < 2*n; i++)
        for (int a = 0; a < 12; a++)
            for (int b = 0; b < 12; b++)
                ATA[a*12+b] += A[i*12+a] * A[i*12+b];

    double p[12];
    for (int i = 0; i < 12; i++) p[i] = (i == 11) ? 1.0 : 0.0;
    for (int iter = 0; iter < 200; iter++) {
        // Inverse iteration: solve ATA * p_new = p_old
        double ATA_reg[144];
        for (int i = 0; i < 144; i++) ATA_reg[i] = ATA[i];
        for (int i = 0; i < 12; i++) ATA_reg[i*12+i] += 1e-10; // regularization
        double rhs[12];
        for (int i = 0; i < 12; i++) rhs[i] = p[i];
        if (!detail::solve_linear_system(12, ATA_reg, rhs)) break;
        double norm = 0;
        for (int i = 0; i < 12; i++) norm += rhs[i]*rhs[i];
        norm = std::sqrt(norm);
        if (norm < 1e-30) break;
        for (int i = 0; i < 12; i++) p[i] = rhs[i] / norm;
    }

    // Denormalize: P = T_img^{-1} * P_norm * T_world
    double T_img[9] = {simg,0,-simg*mx, 0,simg,-simg*my, 0,0,1};
    double T_img_inv[9];
    detail::mat3x3_inverse(T_img, T_img_inv);

    double Pn[12] = {p[0],p[1],p[2],p[3], p[4],p[5],p[6],p[7], p[8],p[9],p[10],p[11]};

    // Step 1: TP = T_img_inv * P_norm (3x3 × 3x4 = 3x4)
    double TP[12] = {0};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 3; k++)
                TP[i*4+j] += T_img_inv[i*3+k] * Pn[k*4+j];

    // Step 2: P = TP * T_world (3x4 × 4x4)
    // T_world = [sworld*I_3 | -sworld*M; 0 0 0 1]
    for (int i = 0; i < 3; i++) {
        result->P[i*4+0] = TP[i*4+0] * sworld;
        result->P[i*4+1] = TP[i*4+1] * sworld;
        result->P[i*4+2] = TP[i*4+2] * sworld;
        result->P[i*4+3] = -TP[i*4+0]*sworld*MX - TP[i*4+1]*sworld*MY - TP[i*4+2]*sworld*MZ + TP[i*4+3];
    }

    // Normalize P so ||P[2,0:3]|| = 1
    double n3 = std::sqrt(result->P[8]*result->P[8]+result->P[9]*result->P[9]+result->P[10]*result->P[10]);
    if (n3 > 1e-30) for (int i = 0; i < 12; i++) result->P[i] /= n3;

    // RQ decompose M = P[0:3,0:3] = K * R
    double M[9] = {result->P[0],result->P[1],result->P[2],
                   result->P[4],result->P[5],result->P[6],
                   result->P[8],result->P[9],result->P[10]};
    double K[9], Rmat[9], Q[9];
    rq_decompose(M, K, Rmat);

    // Normalize K so K[2][2]=1
    double scale = K[8];
    for (int i = 0; i < 9; i++) K[i] /= scale;

    result->K.fx = K[0]; result->K.fy = K[4];
    result->K.cx = K[2]; result->K.cy = K[5];
    result->K.k1 = result->K.k2 = result->K.k3 = result->K.p1 = result->K.p2 = 0;

    // Extract translation: t = K^{-1} * P[0:3,3]
    double Pcol[3] = {result->P[3]/scale, result->P[7]/scale, result->P[11]/scale};
    result->extrinsics.translation[0] = Pcol[0] - K[2]*Pcol[2]/K[0]; // simplify
    result->extrinsics.translation[1] = (Pcol[1] - K[5]*Pcol[2]) / K[4];
    result->extrinsics.translation[2] = Pcol[2];

    // Compute residual
    double total_err = 0;
    for (int i = 0; i < n; i++) {
        double X = params->correspondences[i].world_x;
        double Y = params->correspondences[i].world_y;
        double Z = params->correspondences[i].world_z;
        double w = result->P[8]*X + result->P[9]*Y + result->P[10]*Z + result->P[11];
        double u = (result->P[0]*X + result->P[1]*Y + result->P[2]*Z + result->P[3]) / w;
        double v = (result->P[4]*X + result->P[5]*Y + result->P[6]*Z + result->P[7]) / w;
        total_err += std::pow(u-params->correspondences[i].image_x,2.0) +
                     std::pow(v-params->correspondences[i].image_y,2.0);
    }
    result->residual = std::sqrt(total_err / n);

    return CalibrationError::Ok;
}

} // namespace calibration
