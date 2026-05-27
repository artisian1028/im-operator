#include "common.hpp"

namespace hdr {

// Reinhard global tone mapping: L_out = L_in / (1 + L_in)
// Works with both integer (bit_depth 8/16) and float (bit_depth=0) input.
HdrError process_reinhard(const uint8_t* input, uint8_t* output,
                           int width, int height, int channels,
                           int bit_depth, const HdrParams& params) {
    HdrError err = validate_hdr_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != HdrError::Ok) return err;

    float sat = std::max(0.0f, std::min(2.0f, params.saturation));
    float inv_gamma = 1.0f / std::max(0.5f, std::min(4.0f, params.gamma));
    float exposure_mul = std::pow(2.0f, std::max(-8.0f, std::min(8.0f, params.exposure)));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 0);
            float g = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 1);
            float b = detail::read_pixel_f(input, x, y, width, channels, bit_depth, 2);

            // For int: read_pixel_f returns [0,1] normalized; for float: raw value
            r *= exposure_mul;
            g *= exposure_mul;
            b *= exposure_mul;

            float L = detail::rgb_to_luma(r, g, b);
            if (L <= 0.0f) {
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 0, 0.0f);
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 1, 0.0f);
                detail::write_pixel_f(output, x, y, width, channels, bit_depth, 2, 0.0f);
                continue;
            }

            // Reinhard: L' = L / (1 + L)
            float Lout = L / (1.0f + L);

            // Color preservation
            float scale = detail::safe_pow(Lout / L, sat);
            float ro = detail::safe_pow(r * scale, inv_gamma);
            float go = detail::safe_pow(g * scale, inv_gamma);
            float bo = detail::safe_pow(b * scale, inv_gamma);

            // For float output, clamp to [0,1]; for int, write_pixel_f handles max_val
            if (bit_depth == 0) {
                ro = detail::clamp_val_f(ro);
                go = detail::clamp_val_f(go);
                bo = detail::clamp_val_f(bo);
            }

            detail::write_pixel_f(output, x, y, width, channels, bit_depth, 0, ro);
            detail::write_pixel_f(output, x, y, width, channels, bit_depth, 1, go);
            detail::write_pixel_f(output, x, y, width, channels, bit_depth, 2, bo);
        }
    }

    return HdrError::Ok;
}

} // namespace hdr
