#include "common.hpp"

namespace saturation {

// Channel Mixer: saturate by mixing a fraction of other channels into each
// channel. Creates a cross-channel "bleed" that increases perceived saturation.
//
// R' = R + s*(R - G)*k  + s*(R - B)*k
// G' = G + s*(G - R)*k  + s*(G - B)*k
// B' = B + s*(B - R)*k  + s*(B - G)*k
//
// where s = saturation - 1, k = mixing coefficient (0.5 default)
SaturationError process_channel_mixer(const uint8_t* input, uint8_t* output,
                                       int width, int height, int channels,
                                       int bit_depth, const SaturationParams& params) {
    SaturationError err = validate_saturation_inputs(input, output, width, height,
                                                       channels, bit_depth);
    if (err != SaturationError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);
    float mv = static_cast<float>(max_val);
    float sat = std::max(0.0f, std::min(3.0f, params.saturation));
    float s = sat - 1.0f; // deviation from identity

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = detail::read_pixel(input, x, y, width, channels, bit_depth, 0) / mv;
            float g = detail::read_pixel(input, x, y, width, channels, bit_depth, 1) / mv;
            float b = detail::read_pixel(input, x, y, width, channels, bit_depth, 2) / mv;

            // Cross-channel differences amplify color separation
            float ro = r + s * ((r - g) * 0.5f + (r - b) * 0.5f);
            float go = g + s * ((g - r) * 0.5f + (g - b) * 0.5f);
            float bo = b + s * ((b - r) * 0.5f + (b - g) * 0.5f);

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
