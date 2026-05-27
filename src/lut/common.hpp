#ifndef LUT_SRC_COMMON_HPP
#define LUT_SRC_COMMON_HPP

#include "lut/algorithms.hpp"
#include <cstdint>
#include <algorithm>
#include <cmath>

namespace lut {
namespace detail {

inline int clamp_val(int v, int max_val) {
    return std::max(0, std::min(v, max_val));
}

inline int safe_max_val(int bit_depth) {
    if (bit_depth <= 0) return 255;
    if (bit_depth >= 16) return 65535;
    return (1 << bit_depth) - 1;
}

inline int read_pixel(const uint8_t* data, int x, int y, int width,
                      int channels, int bit_depth, int channel) {
    if (bit_depth <= 8) {
        return data[(static_cast<size_t>(y) * width + x) * channels + channel];
    } else {
        const uint16_t* data16 = reinterpret_cast<const uint16_t*>(data);
        return data16[(static_cast<size_t>(y) * width + x) * channels + channel];
    }
}

inline void write_pixel(uint8_t* data, int x, int y, int width,
                        int channels, int bit_depth, int channel, int value) {
    if (bit_depth <= 8) {
        data[(static_cast<size_t>(y) * width + x) * channels + channel] =
            static_cast<uint8_t>(value);
    } else {
        uint16_t* data16 = reinterpret_cast<uint16_t*>(data);
        data16[(static_cast<size_t>(y) * width + x) * channels + channel] =
            static_cast<uint16_t>(value);
    }
}

// Trilinear interpolation within a 3D LUT.
// lut: flat array of size^3 * 3 floats (R-major, RGB interleaved)
// rn/gn/bn: normalized input in [0, 1]
// size: samples per dimension
// r_out/g_out/b_out: normalized output in [0, 1]
inline void trilinear_lookup(const float* lut_data, int size,
                              float rn, float gn, float bn,
                              float& r_out, float& g_out, float& b_out) {
    float r_scaled = rn * static_cast<float>(size - 1);
    float g_scaled = gn * static_cast<float>(size - 1);
    float b_scaled = bn * static_cast<float>(size - 1);

    int r0 = std::max(0, std::min(static_cast<int>(r_scaled), size - 2));
    int g0 = std::max(0, std::min(static_cast<int>(g_scaled), size - 2));
    int b0 = std::max(0, std::min(static_cast<int>(b_scaled), size - 2));
    int r1 = std::min(r0 + 1, size - 1);
    int g1 = std::min(g0 + 1, size - 1);
    int b1 = std::min(b0 + 1, size - 1);

    float dr = r_scaled - static_cast<float>(r0);
    float dg = g_scaled - static_cast<float>(g0);
    float db = b_scaled - static_cast<float>(b0);

    auto sample = [lut_data, size](int ri, int gi, int bi, int ch) -> float {
        size_t idx = (static_cast<size_t>(ri) * size * size +
                       static_cast<size_t>(gi) * size + bi) * 3 + ch;
        return lut_data[idx];
    };

    // Interpolate along B for each corner of square at (r0,g0), (r0,g1), (r1,g0), (r1,g1)
    auto interp_b = [&](int ri, int gi) {
        float c000 = sample(ri, gi, b0, 0);
        float c001 = sample(ri, gi, b1, 0);
        float c010 = sample(ri, gi, b0, 1);
        float c011 = sample(ri, gi, b1, 1);
        float c020 = sample(ri, gi, b0, 2);
        float c021 = sample(ri, gi, b1, 2);
        return std::make_tuple(
            c000 + db * (c001 - c000),
            c010 + db * (c011 - c010),
            c020 + db * (c021 - c020)
        );
    };

    auto [v00_r, v00_g, v00_b] = interp_b(r0, g0);
    auto [v01_r, v01_g, v01_b] = interp_b(r0, g1);
    auto [v10_r, v10_g, v10_b] = interp_b(r1, g0);
    auto [v11_r, v11_g, v11_b] = interp_b(r1, g1);

    // Interpolate along G
    float v0_r = v00_r + dg * (v01_r - v00_r);
    float v0_g = v00_g + dg * (v01_g - v00_g);
    float v0_b = v00_b + dg * (v01_b - v00_b);

    float v1_r = v10_r + dg * (v11_r - v10_r);
    float v1_g = v10_g + dg * (v11_g - v10_g);
    float v1_b = v10_b + dg * (v11_b - v10_b);

    // Interpolate along R
    r_out = v0_r + dr * (v1_r - v0_r);
    g_out = v0_g + dr * (v1_g - v0_g);
    b_out = v0_b + dr * (v1_b - v0_b);
}

} // namespace detail
} // namespace lut

#endif // LUT_SRC_COMMON_HPP
