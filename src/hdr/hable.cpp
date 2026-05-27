#include "common.hpp"

namespace hdr {

HdrError process_hable(const uint8_t* input, uint8_t* output,
                        int width, int height, int channels,
                        int bit_depth, const HdrParams& params) {
    HdrError err = validate_hdr_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != HdrError::Ok) return err;

    float strength = std::max(0.0f, std::min(2.0f, params.strength));
    float sat = std::max(0.0f, std::min(2.0f, params.saturation));
    float inv_gamma = 1.0f / std::max(0.5f, std::min(4.0f, params.gamma));
    float exposure_mul = std::pow(2.0f, std::max(-8.0f, std::min(8.0f, params.exposure)));

    const float A = 0.22f, Bv = 0.30f, C = 0.10f, D = 0.20f, E = 0.01f, F = 0.30f;
    const float W = 11.2f;
    float hw = (W * (A * W + C * Bv) + D * E) / (W * (A * W + Bv) + D * F);
    float hw_norm = hw - E / F;
    float inv_range = 1.0f / hw_norm;

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
            float hx = (xs * (A * xs + C * Bv) + D * E) / (xs * (A * xs + Bv) + D * F);
            hx -= E / F;
            float Lout = std::max(0.0f, std::min(1.0f, hx * inv_range));
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
