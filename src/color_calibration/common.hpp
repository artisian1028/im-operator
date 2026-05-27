#ifndef COLOR_CALIBRATION_COMMON_HPP
#define COLOR_CALIBRATION_COMMON_HPP

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace color_calibration {
namespace detail {

inline int safe_max_val(int bit_depth) {
    return (1 << std::min(bit_depth, 16)) - 1;
}

inline int read_pixel(const uint8_t* data, int x, int y, int w, int c, int bd) {
    if (bd <= 8) return data[(static_cast<size_t>(y) * w + x) * 3 + c];
    return reinterpret_cast<const uint16_t*>(data)[(static_cast<size_t>(y) * w + x) * 3 + c];
}

inline float read_pixel_norm(const uint8_t* data, int x, int y, int w, int c, int bd) {
    return static_cast<float>(read_pixel(data, x, y, w, c, bd)) / static_cast<float>(safe_max_val(bd));
}

// Gaussian elimination for Ax = b (A is n x n, b is n)
// A and b are modified in-place, solution stored in b.
inline bool solve_linear_system(int n, double* A, double* b) {
    for (int col = 0; col < n; ++col) {
        int max_row = col;
        double max_val = std::abs(A[col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            double val = std::abs(A[row * n + col]);
            if (val > max_val) { max_val = val; max_row = row; }
        }
        if (max_val < 1e-30) return false;

        if (max_row != col) {
            for (int j = col; j < n; ++j) std::swap(A[col * n + j], A[max_row * n + j]);
            std::swap(b[col], b[max_row]);
        }

        double pivot = A[col * n + col];
        for (int row = col + 1; row < n; ++row) {
            double factor = A[row * n + col] / pivot;
            A[row * n + col] = 0.0;
            for (int j = col + 1; j < n; ++j) A[row * n + j] -= factor * A[col * n + j];
            b[row] -= factor * b[col];
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) sum -= A[i * n + j] * b[j];
        if (std::abs(A[i * n + i]) < 1e-30) return false;
        b[i] = sum / A[i * n + i];
    }
    return true;
}

} // namespace detail
} // namespace color_calibration

#endif
