#include "common.hpp"

namespace sharpen {

// Adaptive sharpening: applies stronger sharpening to edges and weaker (or no)
// sharpening to flat regions, avoiding noise amplification.
//
// 1. Compute local variance as an edge strength measure
// 2. Compute unsharp mask detail
// 3. Blend: weight = clamp((variance - threshold) / range, 0, 1)
//    result = original + amount * weight * detail
SharpenError process_adaptive(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, const SharpenParams& params) {
    SharpenError err = validate_sharpen_inputs(input, output, width, height,
                                                channels, bit_depth);
    if (err != SharpenError::Ok) return err;

    if (width < 5 || height < 5) return SharpenError::ImageTooSmall;

    int max_val = detail::safe_max_val(bit_depth);
    float amount = std::max(0.0f, std::min(3.0f, params.amount));
    float sigma = std::max(0.3f, std::min(5.0f, params.radius));
    float threshold = std::max(0.0f, std::min(1.0f, params.threshold));

    size_t total = static_cast<size_t>(width) * height;

    // Compute luminance plane for edge detection
    std::vector<float> luma(total);
    for (size_t i = 0; i < total; i++) {
        int x = static_cast<int>(i % width);
        int y = static_cast<int>(i / width);
        float r = static_cast<float>(detail::read_pixel(input, x, y, width, 3, bit_depth, 0));
        float g = static_cast<float>(detail::read_pixel(input, x, y, width, 3, bit_depth, 1));
        float b = static_cast<float>(detail::read_pixel(input, x, y, width, 3, bit_depth, 2));
        luma[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }

    // Compute local variance (3x3 window) on luminance
    std::vector<float> variance(total);
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            size_t idx = static_cast<size_t>(y) * width + x;
            float v00 = luma[static_cast<size_t>(y - 1) * width + (x - 1)];
            float v01 = luma[static_cast<size_t>(y - 1) * width + x];
            float v02 = luma[static_cast<size_t>(y - 1) * width + (x + 1)];
            float v10 = luma[static_cast<size_t>(y) * width + (x - 1)];
            float v11 = luma[idx];
            float v12 = luma[static_cast<size_t>(y) * width + (x + 1)];
            float v20 = luma[static_cast<size_t>(y + 1) * width + (x - 1)];
            float v21 = luma[static_cast<size_t>(y + 1) * width + x];
            float v22 = luma[static_cast<size_t>(y + 1) * width + (x + 1)];

            float mean = (v00 + v01 + v02 + v10 + v11 + v12 + v20 + v21 + v22) / 9.0f;
            float var = ((v00 - mean) * (v00 - mean) + (v01 - mean) * (v01 - mean) +
                         (v02 - mean) * (v02 - mean) + (v10 - mean) * (v10 - mean) +
                         (v11 - mean) * (v11 - mean) + (v12 - mean) * (v12 - mean) +
                         (v20 - mean) * (v20 - mean) + (v21 - mean) * (v21 - mean) +
                         (v22 - mean) * (v22 - mean)) / 9.0f;
            variance[idx] = var;
        }
    }

    // Find max variance for normalization
    float max_var = 1.0f;
    for (size_t i = 0; i < total; i++) {
        if (variance[i] > max_var) max_var = variance[i];
    }
    float var_thresh = threshold * max_var;

    // Blur the luminance plane for unsharp mask detail
    std::vector<float> blurred_luma(total);
    detail::gaussian_blur_plane(luma.data(), blurred_luma.data(), width, height, sigma);

    // Apply per-channel adaptive unsharp mask
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                size_t idx = static_cast<size_t>(y) * width + x;
                float src = detail::read_pixel(input, x, y, width, 3, bit_depth, c);

                // Compute unsharp detail from blurred luminance (shared across channels)
                float detail_luma = luma[idx] - blurred_luma[idx];

                // Edge weight from local variance
                float weight = (max_var > var_thresh && variance[idx] > var_thresh)
                               ? std::min(1.0f, (variance[idx] - var_thresh) / (max_var - var_thresh))
                               : 0.0f;

                // Adaptive mix: strong at edges, weak at flat areas
                float result = src + amount * weight * detail_luma;
                detail::write_pixel(output, x, y, width, 3, bit_depth, c,
                                    detail::clamp_val(static_cast<int>(result + 0.5f), max_val));
            }
        }
    }

    return SharpenError::Ok;
}

} // namespace sharpen
