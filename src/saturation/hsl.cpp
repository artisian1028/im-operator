#include "common.hpp"

namespace saturation {

// HSL saturation: convert RGB → HSL, scale S, convert back.
// Saturation = 0 → grayscale, 1 → identity, >1 → boosted.
SaturationError process_hsl(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int bit_depth, const SaturationParams& params) {
    SaturationError err = validate_saturation_inputs(input, output, width, height,
                                                       channels, bit_depth);
    if (err != SaturationError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);
    float mv = static_cast<float>(max_val);
    float sat = std::max(0.0f, std::min(3.0f, params.saturation));

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = detail::read_pixel(input, x, y, width, channels, bit_depth, 0) / mv;
            float g = detail::read_pixel(input, x, y, width, channels, bit_depth, 1) / mv;
            float b = detail::read_pixel(input, x, y, width, channels, bit_depth, 2) / mv;

            float h, s, l;
            detail::rgb_to_hsl(r, g, b, h, s, l);
            s = std::min(1.0f, s * sat);
            detail::hsl_to_rgb(h, s, l, r, g, b);

            detail::write_pixel(output, x, y, width, channels, bit_depth, 0,
                                detail::clamp_val(static_cast<int>(r * mv + 0.5f), max_val));
            detail::write_pixel(output, x, y, width, channels, bit_depth, 1,
                                detail::clamp_val(static_cast<int>(g * mv + 0.5f), max_val));
            detail::write_pixel(output, x, y, width, channels, bit_depth, 2,
                                detail::clamp_val(static_cast<int>(b * mv + 0.5f), max_val));
        }
    }

    return SaturationError::Ok;
}

} // namespace saturation
