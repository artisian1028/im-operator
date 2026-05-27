#include "common.hpp"

namespace color_temp {

// Kelvin-based color temperature: convert input image to appear as if
// illuminated by a blackbody radiator at the given Kelvin temperature.
//
// The multipliers convert FROM the current scene temp TO D65 (neutral).
// Input is assumed to be D65-balanced; the multiplies apply the color cast.
// For correction (white balance), use PRESET or WHITE_BALANCE instead.
ColorTempError process_kelvin(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, int kelvin,
                               IlluminantPreset /*preset*/,
                               float r_gain, float b_gain) {
    ColorTempError err = validate_color_temp_inputs(input, output, width, height,
                                                      channels, bit_depth);
    if (err != ColorTempError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);

    // If user specified explicit gains via r_gain/b_gain, use those
    float r_mul, b_mul;
    if (r_gain != 1.0f || b_gain != 1.0f) {
        r_mul = r_gain;
        b_mul = b_gain;
    } else {
        kelvin_to_rgb_multipliers(kelvin, r_mul, b_mul);
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int r = detail::read_pixel(input, x, y, width, channels, bit_depth, 0);
            int g = detail::read_pixel(input, x, y, width, channels, bit_depth, 1);
            int b = detail::read_pixel(input, x, y, width, channels, bit_depth, 2);

            int ro = detail::clamp_val(static_cast<int>(r * r_mul + 0.5f), max_val);
            int go = g; // G unchanged
            int bo = detail::clamp_val(static_cast<int>(b * b_mul + 0.5f), max_val);

            detail::write_pixel(output, x, y, width, channels, bit_depth, 0, ro);
            detail::write_pixel(output, x, y, width, channels, bit_depth, 1, go);
            detail::write_pixel(output, x, y, width, channels, bit_depth, 2, bo);
        }
    }

    return ColorTempError::Ok;
}

} // namespace color_temp
