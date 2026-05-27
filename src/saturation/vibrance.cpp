#include "common.hpp"

namespace saturation {

// Vibrance: intelligent saturation that protects already-saturated colors
// (especially skin tones in the orange range) and boosts muted colors more.
//
// Weight = 1 - current_saturation → muted colors get more boost.
// Skin tone protection: reduce weight in the orange hue range.
SaturationError process_vibrance(const uint8_t* input, uint8_t* output,
                                  int width, int height, int channels,
                                  int bit_depth, const SaturationParams& params) {
    SaturationError err = validate_saturation_inputs(input, output, width, height,
                                                       channels, bit_depth);
    if (err != SaturationError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);
    float mv = static_cast<float>(max_val);
    float vib = std::max(0.0f, std::min(3.0f, params.vibrance));

    // If saturation param is set (not 1.0), use it as override for vibrance too
    float amount = (params.saturation != 1.0f) ? params.saturation : vib;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = detail::read_pixel(input, x, y, width, channels, bit_depth, 0) / mv;
            float g = detail::read_pixel(input, x, y, width, channels, bit_depth, 1) / mv;
            float b = detail::read_pixel(input, x, y, width, channels, bit_depth, 2) / mv;

            float mx = std::max({r, g, b});
            float mn = std::min({r, g, b});
            float current_sat = (mx > 0.0f) ? (mx - mn) / mx : 0.0f;

            // Muted colors get higher weight (1 - current_sat)
            float weight = 1.0f - current_sat;

            // Skin tone protection: orange hue range (~15° - 45°)
            // Detect via R > G > B pattern
            if (r > g && g > b) {
                float skin_strength = (r - b) > 0.0f ? (g - b) / (r - b) : 0.0f;
                if (skin_strength > 0.3f && skin_strength < 0.7f) {
                    weight *= 0.3f; // reduce boost on skin tones
                }
            }

            // Effective saturation boost
            float boost = 1.0f + (amount - 1.0f) * weight;

            // Apply via luminance-saturation approach
            float luma = 0.299f * r + 0.587f * g + 0.114f * b;
            float ro = std::max(0.0f, std::min(1.0f, luma + (r - luma) * boost));
            float go = std::max(0.0f, std::min(1.0f, luma + (g - luma) * boost));
            float bo = std::max(0.0f, std::min(1.0f, luma + (b - luma) * boost));

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
