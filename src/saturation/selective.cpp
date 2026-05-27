#include "common.hpp"

namespace saturation {

// Selective: per-channel saturation control.
// Uses the luminance-saturation model independently for each channel.
//
// For each channel c:
//   luma = 0.299*R + 0.587*G + 0.114*B
//   c' = luma + sat_c * (c - luma)
//
// sat_c = 0: that channel becomes grayscale
// sat_c = 1: identity
// sat_c > 1: that channel's color is boosted
SaturationError process_selective(const uint8_t* input, uint8_t* output,
                                   int width, int height, int channels,
                                   int bit_depth, const SaturationParams& params) {
    SaturationError err = validate_saturation_inputs(input, output, width, height,
                                                       channels, bit_depth);
    if (err != SaturationError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);
    float mv = static_cast<float>(max_val);
    float r_sat = std::max(0.0f, std::min(3.0f, params.r_sat));
    float g_sat = std::max(0.0f, std::min(3.0f, params.g_sat));
    float b_sat = std::max(0.0f, std::min(3.0f, params.b_sat));

    // Also apply global saturation as a multiplier on each channel
    if (params.saturation != 1.0f) {
        r_sat *= params.saturation;
        g_sat *= params.saturation;
        b_sat *= params.saturation;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = detail::read_pixel(input, x, y, width, channels, bit_depth, 0) / mv;
            float g = detail::read_pixel(input, x, y, width, channels, bit_depth, 1) / mv;
            float b = detail::read_pixel(input, x, y, width, channels, bit_depth, 2) / mv;

            float luma = 0.299f * r + 0.587f * g + 0.114f * b;

            float ro = std::max(0.0f, std::min(1.0f, luma + r_sat * (r - luma)));
            float go = std::max(0.0f, std::min(1.0f, luma + g_sat * (g - luma)));
            float bo = std::max(0.0f, std::min(1.0f, luma + b_sat * (b - luma)));

            detail::write_pixel(output, x, y, width, channels, bit_depth, 0,
                                detail::clamp_val(static_cast<int>(ro * mv + 0.5f), max_val));
            detail::write_pixel(output, x, y, width, channels, bit_depth, 1,
                                detail::clamp_val(static_cast<int>(go * mv + 0.5f), max_val));
            detail::write_pixel(output, x, y, width, channels, bit_depth, 2,
                                detail::clamp_val(static_cast<int>(bo * mv + 0.5f), max_val));
        }
    }

    return SaturationError::Ok;
}

} // namespace saturation
