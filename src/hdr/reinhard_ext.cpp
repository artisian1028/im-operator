#include "common.hpp"
#include <vector>

namespace hdr {

HdrError process_reinhard_ext(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, const HdrParams& params) {
    HdrError err = validate_hdr_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != HdrError::Ok) return err;

    float key = std::max(0.01f, std::min(1.0f, params.key));
    float wp2 = params.white_point * params.white_point;
    wp2 = std::max(0.25f, std::min(400.0f, wp2));
    float sat = std::max(0.0f, std::min(2.0f, params.saturation));
    float inv_gamma = 1.0f / std::max(0.5f, std::min(4.0f, params.gamma));

    // Compute log-average luminance
    size_t total = static_cast<size_t>(width) * height;
    float log_sum = 0.0f;
    const float eps = 1e-6f;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 0);
            float g = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 1);
            float b = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 2);
            float L = detail::rgb_to_luma(r, g, b);
            log_sum += std::log(eps + L);
        }
    }
    float log_avg = log_sum / static_cast<float>(total);
    float L_avg = std::exp(log_avg);
    float scale_factor = key / (eps + L_avg);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 0);
            float g = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 1);
            float b = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 2);

            float L = detail::rgb_to_luma(r, g, b);
            if (L <= 0.0f) {
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 0, 0.0f);
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 1, 0.0f);
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 2, 0.0f);
                continue;
            }

            float Lm = L * scale_factor;
            float Lout = Lm * (1.0f + Lm / wp2) / (1.0f + Lm);
            float scale = detail::safe_pow(Lout / L, sat);

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
