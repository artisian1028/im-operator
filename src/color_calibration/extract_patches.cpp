#include "common.hpp"
#include "color_calibration/algorithms.hpp"
#include <vector>
#include <algorithm>

namespace color_calibration {

// Extract median RGB from each detected patch.
ColorCalibrationError process_extract_patches(const uint8_t* input,
                                               int width, int height,
                                               int channels, int bit_depth,
                                               const ChartDetection* detection,
                                               ChartMeasurements* result) {
    if (!input || !detection || !result) return ColorCalibrationError::NullInput;
    if (!is_valid_dimensions(width, height)) return ColorCalibrationError::InvalidDimensions;
    if (channels != 3) return ColorCalibrationError::InvalidChannels;

    int count = 0;

    for (int i = 0; i < 24; i++) {
        if (!detection->valid[i]) {
            result->colors[i].r = 0.0f;
            result->colors[i].g = 0.0f;
            result->colors[i].b = 0.0f;
            continue;
        }

        const auto& p = detection->patches[i];
        int cx = static_cast<int>(p.cx * static_cast<float>(width));
        int cy = static_cast<int>(p.cy * static_cast<float>(height));
        int hw = std::max(2, static_cast<int>(p.half_w * static_cast<float>(width)));
        int hh = std::max(2, static_cast<int>(p.half_h * static_cast<float>(height)));

        int x0 = std::max(0, cx - hw);
        int x1 = std::min(width - 1, cx + hw);
        int y0 = std::max(0, cy - hh);
        int y1 = std::min(height - 1, cy + hh);

        // Collect all pixels in sampling window
        std::vector<float> r_vals, g_vals, b_vals;
        int n = (x1 - x0 + 1) * (y1 - y0 + 1);
        r_vals.reserve(n); g_vals.reserve(n); b_vals.reserve(n);

        for (int y = y0; y <= y1; y++) {
            for (int x = x0; x <= x1; x++) {
                r_vals.push_back(detail::read_pixel_norm(input, x, y, width, 0, bit_depth));
                g_vals.push_back(detail::read_pixel_norm(input, x, y, width, 1, bit_depth));
                b_vals.push_back(detail::read_pixel_norm(input, x, y, width, 2, bit_depth));
            }
        }

        if (n < 4) continue;

        // Median (robust against outlier pixels)
        size_t mid = n / 2;
        std::nth_element(r_vals.begin(), r_vals.begin() + mid, r_vals.end());
        std::nth_element(g_vals.begin(), g_vals.begin() + mid, g_vals.end());
        std::nth_element(b_vals.begin(), b_vals.begin() + mid, b_vals.end());

        result->colors[i].r = r_vals[mid];
        result->colors[i].g = g_vals[mid];
        result->colors[i].b = b_vals[mid];
        count++;
    }

    result->count = count;
    if (count < 12) return ColorCalibrationError::InsufficientPatches;
    return ColorCalibrationError::Ok;
}

} // namespace color_calibration
