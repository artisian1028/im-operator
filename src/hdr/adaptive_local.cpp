#include "common.hpp"
#include <vector>

namespace hdr {

HdrError process_adaptive_local(const uint8_t* input, uint8_t* output,
                                 int width, int height, int channels,
                                 int bit_depth, const HdrParams& params) {
    HdrError err = validate_hdr_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != HdrError::Ok) return err;
    if (width < 5 || height < 5) return HdrError::ImageTooSmall;

    float sat = std::max(0.0f, std::min(2.0f, params.saturation));
    float strength = std::max(0.0f, std::min(2.0f, params.strength));
    float inv_gamma = 1.0f / std::max(0.5f, std::min(4.0f, params.gamma));
    float exposure_mul = std::pow(2.0f, std::max(-8.0f, std::min(8.0f, params.exposure)));

    size_t total = static_cast<size_t>(width) * height;
    std::vector<float> L(total);
    std::vector<float> base(total);
    std::vector<float> r_vals(total);
    std::vector<float> g_vals(total);
    std::vector<float> b_vals(total);

    // Pass 1: read and extract luminance
    float max_L = 0.0f;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = static_cast<size_t>(y) * width + x;
            float r = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 0) * exposure_mul;
            float g = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 1) * exposure_mul;
            float b = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 2) * exposure_mul;

            r_vals[idx] = r; g_vals[idx] = g; b_vals[idx] = b;
            L[idx] = detail::rgb_to_luma(r, g, b);
            if (L[idx] > max_L) max_L = L[idx];
        }
    }

    // Pass 2: approximate bilateral filter for base layer
    int radius = 1;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f, weight_sum = 0.0f;
            float center = L[static_cast<size_t>(y) * width + x];

            for (int dy = -radius; dy <= radius; dy++) {
                int ny = std::clamp(y + dy, 0, height - 1);
                for (int dx = -radius; dx <= radius; dx++) {
                    int nx = std::clamp(x + dx, 0, width - 1);
                    float neighbor = L[static_cast<size_t>(ny) * width + nx];
                    float ds = static_cast<float>(dx * dx + dy * dy);
                    float spatial_w = std::exp(-ds / 4.5f);
                    float dr = center - neighbor;
                    float range_w = std::exp(-dr * dr / (0.02f * max_L * max_L + 1e-6f));
                    float w = spatial_w * range_w;
                    sum += neighbor * w;
                    weight_sum += w;
                }
            }
            base[static_cast<size_t>(y) * width + x] = (weight_sum > 0.0f) ? sum / weight_sum : center;
        }
    }

    // Pass 3: compress base, preserve detail, reconstruct
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t idx = static_cast<size_t>(y) * width + x;
            float r = r_vals[idx], g = g_vals[idx], b = b_vals[idx];
            float luma = L[idx];

            if (luma <= 0.0f) {
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 0, 0.0f);
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 1, 0.0f);
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 2, 0.0f);
                continue;
            }

            float detail = luma - base[idx];
            float base_compressed = base[idx] / (1.0f + base[idx]);
            base_compressed = base[idx] * (1.0f - strength) + base_compressed * strength;
            float Lout = base_compressed + detail * strength;
            Lout = std::max(0.0f, std::min(1.0f, Lout));
            float scale = Lout / luma;

            float ro = detail::safe_pow(r * scale, inv_gamma);
            float go = detail::safe_pow(g * scale, inv_gamma);
            float bo = detail::safe_pow(b * scale, inv_gamma);

            if (bit_depth == 0) {
                ro = detail::clamp_val_f(ro); go = detail::clamp_val_f(go); bo = detail::clamp_val_f(bo);
            }
            detail::write_pixel_f(output, x, y, width, channels, bit_depth, 0, ro);
            detail::write_pixel_f(output, x, y, width, channels, bit_depth, 1, go);
            detail::write_pixel_f(output, x, y, width, channels, bit_depth, 2, bo);
        }
    }

    return HdrError::Ok;
}

} // namespace hdr
