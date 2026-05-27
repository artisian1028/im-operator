#include "common.hpp"
#include "local_contrast/algorithms.hpp"
#include <vector>

namespace local_contrast {

// Large-radius unsharp mask for local contrast enhancement (Lightroom "Clarity").
//
// 1. Extract luminance L from RGB
// 2. Apply large-radius Gaussian blur → L_blur
// 3. detail = L - L_blur
// 4. Apply S-curve to detail (boost weak details, suppress strong edges)
// 5. L_out = L_blur + amount * detail_out
// 6. Restore color: RGB_out = (L_out / L) * RGB
LocalContrastError process_unsharp(const uint8_t* input, uint8_t* output,
                                    int width, int height, int channels,
                                    int bit_depth, const LocalContrastParams& params) {
    auto err = validate_local_contrast_inputs(input, output, width, height, channels, bit_depth);
    if (err != LocalContrastError::Ok) return err;

    if (width < 5 || height < 5) return LocalContrastError::ImageTooSmall;

    int max_val = detail::safe_max_val(bit_depth);
    float mv = static_cast<float>(max_val);
    float amount = std::max(0.0f, std::min(2.0f, params.amount));
    if (amount <= 0.0f) {
        // Passthrough: copy input to output
        size_t sz = static_cast<size_t>(width) * height * 3 * (bit_depth <= 8 ? 1 : 2);
        std::memcpy(output, input, sz);
        return LocalContrastError::Ok;
    }

    float sigma = std::max(3.0f, std::min(50.0f, params.radius));
    int radius = static_cast<int>(std::ceil(sigma * 2.0f));
    radius = std::min(radius, std::min(width, height) / 2 - 1);

    size_t total = static_cast<size_t>(width) * height;
    std::vector<float> L(total);
    std::vector<float> L_blur(total);
    std::vector<float> r_in(total), g_in(total), b_in(total);

    // Pass 1: extract luminance
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = static_cast<size_t>(y) * width + x;
            float r = detail::read_pixel(input, x, y, width, channels, bit_depth, 0) / mv;
            float g = detail::read_pixel(input, x, y, width, channels, bit_depth, 1) / mv;
            float b = detail::read_pixel(input, x, y, width, channels, bit_depth, 2) / mv;
            r_in[idx] = r; g_in[idx] = g; b_in[idx] = b;
            L[idx] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }
    }

    // Pass 2: separable Gaussian blur to get L_blur
    // Horizontal pass
    std::vector<float> L_tmp(total);
    float sigma2 = 2.0f * sigma * sigma;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f, wsum = 0.0f;
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = std::clamp(x + dx, 0, width - 1);
                float w = std::exp(-static_cast<float>(dx * dx) / sigma2);
                sum += L[static_cast<size_t>(y) * width + nx] * w;
                wsum += w;
            }
            L_tmp[static_cast<size_t>(y) * width + x] = sum / wsum;
        }
    }
    // Vertical pass
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f, wsum = 0.0f;
            for (int dy = -radius; dy <= radius; dy++) {
                int ny = std::clamp(y + dy, 0, height - 1);
                float w = std::exp(-static_cast<float>(dy * dy) / sigma2);
                sum += L_tmp[static_cast<size_t>(ny) * width + x] * w;
                wsum += w;
            }
            L_blur[static_cast<size_t>(y) * width + x] = sum / wsum;
        }
    }

    // Pass 3: enhance and reconstruct
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = static_cast<size_t>(y) * width + x;
            float luma = L[idx];
            if (luma <= 0.0f) {
                detail::write_pixel(output, x, y, width, channels, bit_depth, 0, 0);
                detail::write_pixel(output, x, y, width, channels, bit_depth, 1, 0);
                detail::write_pixel(output, x, y, width, channels, bit_depth, 2, 0);
                continue;
            }

            float detail_val = luma - L_blur[idx];

            // S-curve on detail: attenuate strong edges, boost weak texture
            float ad = std::abs(detail_val);
            float sign = (detail_val >= 0.0f) ? 1.0f : -1.0f;
            float detail_out = sign * (ad / (ad + 0.02f)) * 0.15f;

            float L_out = L_blur[idx] + amount * detail_out * 3.0f;
            L_out = std::max(0.0f, std::min(1.0f, L_out));

            float scale = L_out / luma;
            float ro = std::max(0.0f, std::min(1.0f, r_in[idx] * scale));
            float go = std::max(0.0f, std::min(1.0f, g_in[idx] * scale));
            float bo = std::max(0.0f, std::min(1.0f, b_in[idx] * scale));

            detail::write_pixel(output, x, y, width, channels, bit_depth, 0,
                                detail::clamp_val(static_cast<int>(ro * mv + 0.5f), max_val));
            detail::write_pixel(output, x, y, width, channels, bit_depth, 1,
                                detail::clamp_val(static_cast<int>(go * mv + 0.5f), max_val));
            detail::write_pixel(output, x, y, width, channels, bit_depth, 2,
                                detail::clamp_val(static_cast<int>(bo * mv + 0.5f), max_val));
        }
    }

    return LocalContrastError::Ok;
}

} // namespace local_contrast
