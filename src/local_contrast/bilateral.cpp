#include "common.hpp"
#include "local_contrast/algorithms.hpp"
#include <vector>

namespace local_contrast {

// Bilateral-filter-based local contrast enhancement.
// More edge-preserving than unsharp: the base layer is obtained via bilateral filter
// instead of Gaussian, so edges don't get halo artifacts.
//
// 1. Extract luminance L
// 2. Bilateral filter → L_base (small spatial sigma, large range sigma)
// 3. L_detail = L - L_base
// 4. Enhance detail, preserve edges
// 5. Restore color
LocalContrastError process_bilateral(const uint8_t* input, uint8_t* output,
                                      int width, int height, int channels,
                                      int bit_depth, const LocalContrastParams& params) {
    auto err = validate_local_contrast_inputs(input, output, width, height, channels, bit_depth);
    if (err != LocalContrastError::Ok) return err;

    if (width < 5 || height < 5) return LocalContrastError::ImageTooSmall;

    int max_val = detail::safe_max_val(bit_depth);
    float mv = static_cast<float>(max_val);
    float amount = std::max(0.0f, std::min(2.0f, params.amount));
    if (amount <= 0.0f) {
        size_t sz = static_cast<size_t>(width) * height * 3 * (bit_depth <= 8 ? 1 : 2);
        std::memcpy(output, input, sz);
        return LocalContrastError::Ok;
    }

    int radius = 2; // bilateral spatial radius
    float sigma_r = 0.1f; // range sigma

    size_t total = static_cast<size_t>(width) * height;
    std::vector<float> L(total), L_base(total);
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

    // Pass 2: bilateral filter base layer
    float sigma_s2 = 2.0f * static_cast<float>(radius * radius);
    float sigma_r2 = 2.0f * sigma_r * sigma_r;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float center = L[static_cast<size_t>(y) * width + x];
            float sum = 0.0f, wsum = 0.0f;

            for (int dy = -radius; dy <= radius; dy++) {
                int ny = std::clamp(y + dy, 0, height - 1);
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = std::clamp(x + dx, 0, width - 1);
                    float neighbor = L[static_cast<size_t>(ny) * width + nx];

                    float ds = static_cast<float>(dx*dx + dy*dy);
                    float spatial_w = std::exp(-ds / sigma_s2);

                    float dr = center - neighbor;
                    float range_w = std::exp(-dr * dr / sigma_r2);

                    float w = spatial_w * range_w;
                    sum += neighbor * w;
                    wsum += w;
                }
            }
            L_base[static_cast<size_t>(y) * width + x] = (wsum > 0.0f) ? sum / wsum : center;
        }
    }

    // Pass 3: enhance detail and reconstruct
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

            float detail_val = luma - L_base[idx];
            float L_out = L_base[idx] + amount * detail_val * 2.0f;
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
