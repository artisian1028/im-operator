#include "common.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace denoise {

// Bilateral filter: edge-preserving smoothing.
//
// For each pixel, weights are the product of:
//   G_spatial(||p - q||)  -- Gaussian on spatial distance
//   G_range(|I(p) - I(q)|) -- Gaussian on intensity difference
//
// I'(p) = sum_q w(p,q) * I(q) / sum_q w(p,q)
//
// strength controls sigma_range (default ~30 for 8-bit).
// sigma_space is fixed based on kernel radius.
//
// Optimization: range kernel is computed via LUT for 8-bit.

namespace {

template<int BPC>
void bilateral_core(const uint8_t* input, uint8_t* output,
                    int width, int height, int channels, int bit_depth,
                    float sigma_space, float sigma_range) {
    int max_val = detail::safe_max_val(bit_depth);
    // Kernel radius: 2 * ceil(sigma_space), cap at 5
    int radius = std::min(5, std::max(1, static_cast<int>(std::ceil(2.0f * sigma_space))));
    int kernel_w = 2 * radius + 1;

    // Precompute spatial Gaussian kernel
    std::vector<float> spatial_kernel(kernel_w * kernel_w);
    float s2_s = 2.0f * sigma_space * sigma_space;
    float spatial_sum = 0.0f;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            float dist2 = static_cast<float>(dx * dx + dy * dy);
            float w = std::exp(-dist2 / s2_s);
            int idx = (dy + radius) * kernel_w + (dx + radius);
            spatial_kernel[idx] = w;
            spatial_sum += w;
        }
    }

    // For 8-bit, precompute range LUT (0..255 -> exp weight)
    const bool use_lut = (BPC == 1);
    std::vector<float> range_lut;
    float s2_r = 2.0f * sigma_range * sigma_range;
    if (use_lut) {
        range_lut.resize(256);
        for (int i = 0; i < 256; i++) {
            float diff = static_cast<float>(i);
            range_lut[i] = std::exp(-(diff * diff) / s2_r);
        }
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                int center_val = detail::read_pixel(input, x, y, width,
                                                     channels, bit_depth, c);
                float weighted_sum = 0.0f;
                float weight_sum = 0.0f;

                for (int dy = -radius; dy <= radius; dy++) {
                    int ny = y + dy;
                    if (ny < 0) ny = 0;
                    if (ny >= height) ny = height - 1;
                    for (int dx = -radius; dx <= radius; dx++) {
                        int nx = x + dx;
                        if (nx < 0) nx = 0;
                        if (nx >= width) nx = width - 1;

                        int neighbor_val = detail::read_pixel(input, nx, ny, width,
                                                               channels, bit_depth, c);
                        float range_w;
                        if (use_lut) {
                            int abs_diff = std::abs(center_val - neighbor_val);
                            range_w = range_lut[abs_diff];
                        } else {
                            float diff = static_cast<float>(center_val - neighbor_val);
                            range_w = std::exp(-(diff * diff) / s2_r);
                        }

                        int k_idx = (dy + radius) * kernel_w + (dx + radius);
                        float w = spatial_kernel[k_idx] * range_w;
                        weighted_sum += w * static_cast<float>(neighbor_val);
                        weight_sum += w;
                    }
                }

                int result = detail::clamp_val(
                    static_cast<int>((weighted_sum / weight_sum) + 0.5f), max_val);
                detail::write_pixel(output, x, y, width, channels, bit_depth,
                                    c, result);
            }
        }
    }
}

} // anonymous namespace

DenoiseError process_bilateral(const uint8_t* input, uint8_t* output,
                                int width, int height, int channels,
                                int bit_depth, float strength) {
    DenoiseError err = validate_denoise_inputs(input, output, width, height,
                                                channels, bit_depth);
    if (err != DenoiseError::Ok) return err;

    if (width < 3 || height < 3) return DenoiseError::ImageTooSmall;

    // strength -> sigma_range: 1.0 -> 30, 2.0 -> 60 (for 8-bit)
    // scaled by max_val for higher bit depths
    float max_val = static_cast<float>(detail::safe_max_val(bit_depth));
    float sigma_range = strength * 30.0f * (max_val / 255.0f);
    // sigma_space: moderate (2.0) gives radius ~4
    float sigma_space = 2.0f;

    if (bit_depth <= 8) {
        bilateral_core<1>(input, output, width, height, channels,
                           bit_depth, sigma_space, sigma_range);
    } else {
        bilateral_core<2>(input, output, width, height, channels,
                           bit_depth, sigma_space, sigma_range);
    }

    return DenoiseError::Ok;
}

} // namespace denoise
