#include "common.hpp"

namespace color_temp {

// Illuminant preset: apply pre-computed RGB multipliers for a named
// standard illuminant to shift the color temperature.
//
// The multipliers are correction factors: they correct FROM the scene
// illuminant TO D65. For example, if the scene was shot under tungsten (~2850K),
// apply PRESET=TUNGSTEN_100W to warm-balance back to neutral.
ColorTempError process_preset(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, int /*kelvin*/,
                               IlluminantPreset preset,
                               float r_gain, float b_gain) {
    ColorTempError err = validate_color_temp_inputs(input, output, width, height,
                                                      channels, bit_depth);
    if (err != ColorTempError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);

    float r_mul, b_mul;
    if (r_gain != 1.0f || b_gain != 1.0f) {
        r_mul = r_gain;
        b_mul = b_gain;
    } else {
        illuminant_to_rgb_multipliers(preset, r_mul, b_mul);
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int r = detail::read_pixel(input, x, y, width, channels, bit_depth, 0);
            int g = detail::read_pixel(input, x, y, width, channels, bit_depth, 1);
            int b = detail::read_pixel(input, x, y, width, channels, bit_depth, 2);

            int ro = detail::clamp_val(static_cast<int>(r * r_mul + 0.5f), max_val);
            int go = g;
            int bo = detail::clamp_val(static_cast<int>(b * b_mul + 0.5f), max_val);

            detail::write_pixel(output, x, y, width, channels, bit_depth, 0, ro);
            detail::write_pixel(output, x, y, width, channels, bit_depth, 1, go);
            detail::write_pixel(output, x, y, width, channels, bit_depth, 2, bo);
        }
    }

    return ColorTempError::Ok;
}

} // namespace color_temp
