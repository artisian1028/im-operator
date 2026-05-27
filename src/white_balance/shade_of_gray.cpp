#include "common.hpp"
#include <cmath>

namespace white_balance {

// Shade of Gray: generalization of Gray World using Minkowski norm.
// For each channel c, compute: (1/N * sum(I_c(x)^p))^(1/p)
// Gains are: G / channel_norm, so the resulting average is neutral.
//
// p=1: equivalent to Gray World
// p=6: default, Finlaysson's recommended Shade of Gray
// p→∞: equivalent to White Patch
WhiteBalanceError process_shade_of_gray(const uint8_t* input, uint8_t* output,
                                         int width, int height, int channels,
                                         int bit_depth, float p,
                                         const WBCoefficients& /*manual_gains*/) {
    WhiteBalanceError err = validate_white_balance_inputs(input, output, width, height,
                                                            channels, bit_depth);
    if (err != WhiteBalanceError::Ok) return err;

    if (width < 1 || height < 1) return WhiteBalanceError::ImageTooSmall;

    int max_val = detail::safe_max_val(bit_depth);
    size_t pixel_count = static_cast<size_t>(width) * height;

    // Compute Minkowski norm per channel
    double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int r_val = detail::read_pixel(input, x, y, width, channels, bit_depth, 0);
            int g_val = detail::read_pixel(input, x, y, width, channels, bit_depth, 1);
            int b_val = detail::read_pixel(input, x, y, width, channels, bit_depth, 2);

            sum_r += std::pow(static_cast<double>(r_val), static_cast<double>(p));
            sum_g += std::pow(static_cast<double>(g_val), static_cast<double>(p));
            sum_b += std::pow(static_cast<double>(b_val), static_cast<double>(p));
        }
    }

    double count = static_cast<double>(pixel_count);
    double norm_r = std::pow(sum_r / count, 1.0 / static_cast<double>(p));
    double norm_g = std::pow(sum_g / count, 1.0 / static_cast<double>(p));
    double norm_b = std::pow(sum_b / count, 1.0 / static_cast<double>(p));

    // Compute gains: norm_g / norm_c (preserving overall brightness at G)
    float gain_r = (norm_r > 0.0) ? static_cast<float>(norm_g / norm_r) : 1.0f;
    float gain_g = 1.0f;
    float gain_b = (norm_b > 0.0) ? static_cast<float>(norm_g / norm_b) : 1.0f;

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
