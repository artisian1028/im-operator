#include "common.hpp"

namespace hdr {

HdrError process_drago(const uint8_t* input, uint8_t* output,
                        int width, int height, int channels,
                        int bit_depth, const HdrParams& params) {
    HdrError err = validate_hdr_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != HdrError::Ok) return err;

    float bias = std::max(0.3f, std::min(1.0f, params.key));
    float sat = std::max(0.0f, std::min(2.0f, params.saturation));
    float inv_gamma = 1.0f / std::max(0.5f, std::min(4.0f, params.gamma));
    float exposure_mul = std::pow(2.0f, std::max(-8.0f, std::min(8.0f, params.exposure)));

    // Pass 1: find max luminance
    float max_L = 0.0f;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 0);
            float g = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 1);
            float b = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 2);
            float L = detail::rgb_to_luma(r * exposure_mul, g * exposure_mul, b * exposure_mul);
            if (L > max_L) max_L = L;
        }
    }
    max_L = std::max(max_L, 1e-6f);
    float log_max = std::log10(1.0f + max_L * 100.0f);

    // Pass 2: tone map
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 0);
            float g = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 1);
            float b = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 2);

            r *= exposure_mul; g *= exposure_mul; b *= exposure_mul;
            float L = detail::rgb_to_luma(r, g, b);
            if (L <= 0.0f) {
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 0, 0.0f);
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 1, 0.0f);
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 2, 0.0f);
                continue;
            }

            float local_bias = bias + (1.0f - bias) * (L / max_L);
            float Lout = std::log10(1.0f + L * 100.0f) / log_max;
            Lout = std::pow(Lout, local_bias);
            Lout = std::max(0.0f, std::min(1.0f, Lout));
            float scale = Lout / L;

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
