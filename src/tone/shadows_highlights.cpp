#include "common.hpp"
#include <cmath>
#include <vector>

namespace tone {

// Shadows / Highlights: independent shadow lifting and highlight recovery
// using a per-pixel weighting mask based on luminance.
//
// Shadows: brighten dark pixels (weight = 1 - luminance)
// Highlights: darken bright pixels (weight = luminance)
//
// The effect is applied smoothly using a Gaussian-like rolloff to avoid
// harsh transitions.
ToneError process_shadows_highlights(const uint8_t* input, uint8_t* output,
                                      int width, int height, int channels,
                                      int bit_depth, const ToneParams& params) {
    ToneError err = validate_tone_inputs(input, output, width, height,
                                          channels, bit_depth);
    if (err != ToneError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);
    float mv = static_cast<float>(max_val);

    float shadows = std::max(-1.0f, std::min(1.0f, params.shadows));
    float highlights = std::max(-1.0f, std::min(1.0f, params.highlights));

    size_t total = static_cast<size_t>(width) * height;

    // Compute per-pixel luminance
    std::vector<float> luma(total);
    for (size_t i = 0; i < total; i++) {
        int x = static_cast<int>(i % width);
        int y = static_cast<int>(i / width);
        float r = detail::read_pixel(input, x, y, width, 3, bit_depth, 0);
        float g = detail::read_pixel(input, x, y, width, 3, bit_depth, 1);
        float b = detail::read_pixel(input, x, y, width, 3, bit_depth, 2);
        luma[i] = (0.299f * r + 0.587f * g + 0.114f * b) / mv;
    }

    // Apply adjustments
    for (size_t i = 0; i < total; i++) {
        int x = static_cast<int>(i % width);
        int y = static_cast<int>(i / width);

        float l = luma[i];
        float r = detail::read_pixel(input, x, y, width, 3, bit_depth, 0) / mv;
        float g = detail::read_pixel(input, x, y, width, 3, bit_depth, 1) / mv;
        float b = detail::read_pixel(input, x, y, width, 3, bit_depth, 2) / mv;

        // Shadow weight: strongest at luma=0, zero at luma=0.5+
        float sw = std::max(0.0f, 1.0f - 2.0f * l);
        sw = sw * sw; // smooth rolloff

        // Highlight weight: strongest at luma=1, zero at luma=0.5-
        float hw = std::max(0.0f, 2.0f * l - 1.0f);
        hw = hw * hw;

        // Shadow lift: brighten
        float shadow_lift = shadows * 0.3f * sw;
        // Highlight recovery: darken
        float highlight_drop = -highlights * 0.3f * hw;

        r += shadow_lift + highlight_drop;
        g += shadow_lift + highlight_drop;
        b += shadow_lift + highlight_drop;

        detail::write_pixel(output, x, y, width, 3, bit_depth, 0,
                            detail::clamp_val(static_cast<int>(r * mv + 0.5f), max_val));
        detail::write_pixel(output, x, y, width, 3, bit_depth, 1,
                            detail::clamp_val(static_cast<int>(g * mv + 0.5f), max_val));
        detail::write_pixel(output, x, y, width, 3, bit_depth, 2,
                            detail::clamp_val(static_cast<int>(b * mv + 0.5f), max_val));
    }

    return ToneError::Ok;
}

} // namespace tone
