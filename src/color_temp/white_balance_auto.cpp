#include "common.hpp"

namespace color_temp {

// Auto white balance: estimate scene illuminant from the input via gray-world
// and apply correction multipliers to neutralize color casts.
//
// Computes R_mean, G_mean, B_mean → correction = G_mean / R_mean for R, etc.
ColorTempError process_white_balance_auto(const uint8_t* input, uint8_t* output,
                                            int width, int height, int channels,
                                            int bit_depth, int /*kelvin*/,
                                            IlluminantPreset /*preset*/,
                                            float r_gain, float b_gain) {
    ColorTempError err = validate_color_temp_inputs(input, output, width, height,
                                                      channels, bit_depth);
    if (err != ColorTempError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);
    size_t total = static_cast<size_t>(width) * height;

    float r_mul, b_mul;

    if (r_gain != 1.0f || b_gain != 1.0f) {
        r_mul = r_gain;
        b_mul = b_gain;
    } else {
        // Gray-world: compute mean per channel and derive correction
        double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                sum_r += detail::read_pixel(input, x, y, width, channels, bit_depth, 0);
                sum_g += detail::read_pixel(input, x, y, width, channels, bit_depth, 1);
                sum_b += detail::read_pixel(input, x, y, width, channels, bit_depth, 2);
            }
        }
        double count = static_cast<double>(total);
        double mean_r = sum_r / count;
        double mean_g = sum_g / count;
        double mean_b = sum_b / count;

        r_mul = (mean_r > 0.0) ? static_cast<float>(mean_g / mean_r) : 1.0f;
        b_mul = (mean_b > 0.0) ? static_cast<float>(mean_g / mean_b) : 1.0f;
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
