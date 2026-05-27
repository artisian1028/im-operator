#include "common.hpp"
#include <algorithm>

namespace white_balance {

// White Patch (Max RGB): the brightest pixel in each channel is assumed to be
// the white point. Gains scale each channel to max_val.
WhiteBalanceError process_white_patch(const uint8_t* input, uint8_t* output,
                                       int width, int height, int channels,
                                       int bit_depth, float /*p*/,
                                       const WBCoefficients& /*manual_gains*/) {
    WhiteBalanceError err = validate_white_balance_inputs(input, output, width, height,
                                                            channels, bit_depth);
    if (err != WhiteBalanceError::Ok) return err;

    if (width < 1 || height < 1) return WhiteBalanceError::ImageTooSmall;

    int max_val = detail::safe_max_val(bit_depth);

    // Find max per channel
    int max_r = 0, max_g = 0, max_b = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int r = detail::read_pixel(input, x, y, width, channels, bit_depth, 0);
            int g = detail::read_pixel(input, x, y, width, channels, bit_depth, 1);
            int b = detail::read_pixel(input, x, y, width, channels, bit_depth, 2);
            if (r > max_r) max_r = r;
            if (g > max_g) max_g = g;
            if (b > max_b) max_b = b;
        }
    }

    // Compute gains: scale each channel so its max reaches max_val
    float gain_r = (max_r > 0) ? static_cast<float>(max_val) / static_cast<float>(max_r) : 1.0f;
    float gain_g = (max_g > 0) ? static_cast<float>(max_val) / static_cast<float>(max_g) : 1.0f;
    float gain_b = (max_b > 0) ? static_cast<float>(max_val) / static_cast<float>(max_b) : 1.0f;

    // Normalize gains so G stays at 1.0 (preserve overall brightness)
    if (gain_g > 0.0f) {
        gain_r /= gain_g;
        gain_b /= gain_g;
        gain_g = 1.0f;
    }

    // Apply gains
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int r = detail::read_pixel(input, x, y, width, channels, bit_depth, 0);
            int g = detail::read_pixel(input, x, y, width, channels, bit_depth, 1);
            int b = detail::read_pixel(input, x, y, width, channels, bit_depth, 2);

            int r_out = detail::clamp_val(static_cast<int>(r * gain_r + 0.5f), max_val);
            int g_out = detail::clamp_val(static_cast<int>(g * gain_g + 0.5f), max_val);
            int b_out = detail::clamp_val(static_cast<int>(b * gain_b + 0.5f), max_val);

            detail::write_pixel(output, x, y, width, channels, bit_depth, 0, r_out);
            detail::write_pixel(output, x, y, width, channels, bit_depth, 1, g_out);
            detail::write_pixel(output, x, y, width, channels, bit_depth, 2, b_out);
        }
    }

    return WhiteBalanceError::Ok;
}

} // namespace white_balance
