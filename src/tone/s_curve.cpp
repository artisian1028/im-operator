#include "common.hpp"
#include <cmath>

namespace tone {

// S-curve contrast: applies a sigmoid with shadow/highlight softness control.
//
//   out = 0.5 + (in - 0.5) * k_contrast
//   where k_contrast is derived from the contrast parameter
//
// Shadows/highlights are blended using a cubic weighting function.
ToneError process_s_curve(const uint8_t* input, uint8_t* output,
                           int width, int height, int channels,
                           int bit_depth, const ToneParams& params) {
    ToneError err = validate_tone_inputs(input, output, width, height,
                                          channels, bit_depth);
    if (err != ToneError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);
    float mv = static_cast<float>(max_val);
    float contrast = std::max(-1.0f, std::min(2.0f, params.contrast));
    float shadows = std::max(-1.0f, std::min(1.0f, params.shadows));
    float highlights = std::max(-1.0f, std::min(1.0f, params.highlights));

    // S-curve via tanh: out = 0.5 + tanh(k * (x - 0.5)) / (2 * tanh(k/2))
    float k = 1.0f + contrast * 4.0f;
    float denom = 2.0f * std::tanh(k * 0.5f);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                int val = detail::read_pixel(input, x, y, width, channels, bit_depth, c);
                float norm = static_cast<float>(val) / mv;

                // S-curve
                float sc = 0.5f + std::tanh(k * (norm - 0.5f)) / denom;

                // Shadow lift: lighten dark areas
                float shadow_weight = (1.0f - norm) * (1.0f - norm);
                sc += shadows * shadow_weight * 0.15f;

                // Highlight recovery: darken bright areas
                float highlight_weight = norm * norm;
                sc += highlights * highlight_weight * 0.15f;

                detail::write_pixel(output, x, y, width, channels, bit_depth, c,
                                    detail::clamp_val(static_cast<int>(sc * mv + 0.5f), max_val));
            }
        }
    }

    return ToneError::Ok;
}

} // namespace tone
