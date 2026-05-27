#include "common.hpp"

namespace hdr {

HdrError process_filmic_aces(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth, const HdrParams& params) {
    HdrError err = validate_hdr_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != HdrError::Ok) return err;

    float strength = std::max(0.0f, std::min(2.0f, params.strength));
    float sat = std::max(0.0f, std::min(2.0f, params.saturation));
    float inv_gamma = 1.0f / std::max(0.5f, std::min(4.0f, params.gamma));
    float exposure_mul = std::pow(2.0f, std::max(-8.0f, std::min(8.0f, params.exposure)));

    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 0);
            float g = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 1);
            float bv = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 2);

            r *= exposure_mul; g *= exposure_mul; bv *= exposure_mul;
            float L = detail::rgb_to_luma(r, g, bv);
            if (L <= 0.0f) {
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 0, 0.0f);
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 1, 0.0f);
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 2, 0.0f);
                continue;
            }

            float xs = L * strength;
            float Lout = xs * (a * xs + b) / (xs * (c * xs + d) + e);
            float scale = Lout / L;

            float ro = detail::safe_pow(r * scale, inv_gamma);
            float go = detail::safe_pow(g * scale, inv_gamma);
            float bo = detail::safe_pow(bv * scale, inv_gamma);

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
