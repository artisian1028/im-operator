#include "common.hpp"

namespace color_temp {

// Manual: apply user-supplied R/B multipliers directly.
ColorTempError process_manual_temp(const uint8_t* input, uint8_t* output,
                                    int width, int height, int channels,
                                    int bit_depth, int /*kelvin*/,
                                    IlluminantPreset /*preset*/,
                                    float r_gain, float b_gain) {
    ColorTempError err = validate_color_temp_inputs(input, output, width, height,
                                                      channels, bit_depth);
    if (err != ColorTempError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int r = detail::read_pixel(input, x, y, width, channels, bit_depth, 0);
            int g = detail::read_pixel(input, x, y, width, channels, bit_depth, 1);
            int b = detail::read_pixel(input, x, y, width, channels, bit_depth, 2);

            int ro = detail::clamp_val(static_cast<int>(r * r_gain + 0.5f), max_val);
            int go = g;
            int bo = detail::clamp_val(static_cast<int>(b * b_gain + 0.5f), max_val);

            detail::write_pixel(output, x, y, width, channels, bit_depth, 0, ro);
            detail::write_pixel(output, x, y, width, channels, bit_depth, 1, go);
            detail::write_pixel(output, x, y, width, channels, bit_depth, 2, bo);
        }
    }

    return ColorTempError::Ok;
}

} // namespace color_temp
