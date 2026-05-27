#include "common.hpp"
#include <algorithm>

namespace white_balance {

// Gray World: assumes the average of each channel should be a neutral gray.
// Computes per-channel gains and applies them.
WhiteBalanceError process_gray_world(const uint8_t* input, uint8_t* output,
                                      int width, int height, int channels,
                                      int bit_depth, float /*p*/,
                                      const WBCoefficients& /*manual_gains*/) {
    WhiteBalanceError err = validate_white_balance_inputs(input, output, width, height,
                                                            channels, bit_depth);
    if (err != WhiteBalanceError::Ok) return err;

    if (width < 1 || height < 1) return WhiteBalanceError::ImageTooSmall;

    int max_val = detail::safe_max_val(bit_depth);
    size_t pixel_count = static_cast<size_t>(width) * height;

    // Compute mean per channel
    double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            sum_r += detail::read_pixel(input, x, y, width, channels, bit_depth, 0);
            sum_g += detail::read_pixel(input, x, y, width, channels, bit_depth, 1);
            sum_b += detail::read_pixel(input, x, y, width, channels, bit_depth, 2);
        }
    }

    double count = static_cast<double>(pixel_count);
    double mean_r = sum_r / count;
    double mean_g = sum_g / count;
    double mean_b = sum_b / count;

    // Compute gains: mean_g / mean_c (preserving overall brightness)
    float gain_r = (mean_g > 0.0) ? static_cast<float>(mean_g / mean_r) : 1.0f;
    float gain_g = 1.0f;
    float gain_b = (mean_g > 0.0) ? static_cast<float>(mean_g / mean_b) : 1.0f;

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
