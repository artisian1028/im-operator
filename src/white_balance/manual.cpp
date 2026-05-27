#include "common.hpp"

namespace white_balance {

// Manual: applies user-supplied RGB gain coefficients directly.
WhiteBalanceError process_manual_wb(const uint8_t* input, uint8_t* output,
                                     int width, int height, int channels,
                                     int bit_depth, float /*p*/,
                                     const WBCoefficients& manual_gains) {
    WhiteBalanceError err = validate_white_balance_inputs(input, output, width, height,
                                                            channels, bit_depth);
    if (err != WhiteBalanceError::Ok) return err;

    if (width < 1 || height < 1) return WhiteBalanceError::ImageTooSmall;

    int max_val = detail::safe_max_val(bit_depth);

    float gain_r = manual_gains.r_gain;
    float gain_g = manual_gains.g_gain;
    float gain_b = manual_gains.b_gain;

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

// Gains estimation helper: compute gains without applying correction.
WBCoefficients compute_white_balance_gains(const uint8_t* input,
                                             int width, int height,
                                             int bit_depth,
                                             WhiteBalanceAlgorithm algorithm,
                                             float p) {
    WBCoefficients gains;

    if (!input || width <= 0 || height <= 0) return gains;

    int max_val = detail::safe_max_val(bit_depth);
    size_t pixel_count = static_cast<size_t>(width) * height;

    switch (algorithm) {
        case WhiteBalanceAlgorithm::GRAY_WORLD: {
            double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    sum_r += detail::read_pixel(input, x, y, width, 3, bit_depth, 0);
                    sum_g += detail::read_pixel(input, x, y, width, 3, bit_depth, 1);
                    sum_b += detail::read_pixel(input, x, y, width, 3, bit_depth, 2);
                }
            }
            double count = static_cast<double>(pixel_count);
            double mean_r = sum_r / count;
            double mean_g = sum_g / count;
            double mean_b = sum_b / count;
            gains.r_gain = (mean_g > 0.0) ? static_cast<float>(mean_g / mean_r) : 1.0f;
            gains.g_gain = 1.0f;
            gains.b_gain = (mean_g > 0.0) ? static_cast<float>(mean_g / mean_b) : 1.0f;
            break;
        }
        case WhiteBalanceAlgorithm::WHITE_PATCH: {
            int max_r = 0, max_g = 0, max_b = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int r = detail::read_pixel(input, x, y, width, 3, bit_depth, 0);
                    int g = detail::read_pixel(input, x, y, width, 3, bit_depth, 1);
                    int b = detail::read_pixel(input, x, y, width, 3, bit_depth, 2);
                    if (r > max_r) max_r = r;
                    if (g > max_g) max_g = g;
                    if (b > max_b) max_b = b;
                }
            }
            float gr = (max_r > 0) ? static_cast<float>(max_val) / static_cast<float>(max_r) : 1.0f;
            float gg = (max_g > 0) ? static_cast<float>(max_val) / static_cast<float>(max_g) : 1.0f;
            float gb = (max_b > 0) ? static_cast<float>(max_val) / static_cast<float>(max_b) : 1.0f;
            if (gg > 0.0f) {
                gains.r_gain = gr / gg;
                gains.g_gain = 1.0f;
                gains.b_gain = gb / gg;
            }
            break;
        }
        case WhiteBalanceAlgorithm::SHADE_OF_GRAY: {
            double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    sum_r += std::pow(static_cast<double>(
                        detail::read_pixel(input, x, y, width, 3, bit_depth, 0)),
                        static_cast<double>(p));
                    sum_g += std::pow(static_cast<double>(
                        detail::read_pixel(input, x, y, width, 3, bit_depth, 1)),
                        static_cast<double>(p));
                    sum_b += std::pow(static_cast<double>(
                        detail::read_pixel(input, x, y, width, 3, bit_depth, 2)),
                        static_cast<double>(p));
                }
            }
            double count = static_cast<double>(pixel_count);
            double nr = std::pow(sum_r / count, 1.0 / static_cast<double>(p));
            double ng = std::pow(sum_g / count, 1.0 / static_cast<double>(p));
            double nb = std::pow(sum_b / count, 1.0 / static_cast<double>(p));
            gains.r_gain = (nr > 0.0) ? static_cast<float>(ng / nr) : 1.0f;
            gains.g_gain = 1.0f;
            gains.b_gain = (nb > 0.0) ? static_cast<float>(ng / nb) : 1.0f;
            break;
        }
        case WhiteBalanceAlgorithm::MANUAL:
            break;
    }

    return gains;
}

} // namespace white_balance
