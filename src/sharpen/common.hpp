#ifndef SHARPEN_SRC_COMMON_HPP
#define SHARPEN_SRC_COMMON_HPP

#include "sharpen/algorithms.hpp"
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>

namespace sharpen {
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

// Make a Gaussian kernel (1D, separable)
inline void make_gaussian_1d(std::vector<float>& kernel, float sigma) {
    int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    int size = 2 * radius + 1;
    kernel.resize(size);
    float sum = 0.0f;
    float s2 = 2.0f * sigma * sigma;
    for (int i = 0; i < size; i++) {
        float x = static_cast<float>(i - radius);
        kernel[i] = std::exp(-(x * x) / s2);
        sum += kernel[i];
    }
    for (int i = 0; i < size; i++) kernel[i] /= sum;
}

// Separable Gaussian blur for a single channel on a float plane.
// Returns blurred plane (allocated internally).
inline void gaussian_blur_plane(const float* src, float* dst, int width, int height,
                                 float sigma) {
    int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> kernel;
    make_gaussian_1d(kernel, sigma);

    // Temp buffer for horizontal pass
    std::vector<float> tmp(static_cast<size_t>(width) * height);

    // Horizontal pass
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; k++) {
                int nx = std::clamp(x + k, 0, width - 1);
                sum += kernel[k + radius] *
                       src[static_cast<size_t>(y) * width + nx];
            }
            tmp[static_cast<size_t>(y) * width + x] = sum;
        }
    }

    // Vertical pass
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            for (int k = -radius; k <= radius; k++) {
                int ny = std::clamp(y + k, 0, height - 1);
                sum += kernel[k + radius] *
                       tmp[static_cast<size_t>(ny) * width + x];
            }
            dst[static_cast<size_t>(y) * width + x] = sum;
        }
    }
}

} // namespace detail
} // namespace sharpen

#endif // SHARPEN_SRC_COMMON_HPP
