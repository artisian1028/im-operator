#include "common.hpp"
#include "color_calibration/algorithms.hpp"

namespace color_calibration {

// Detect X-Rite ColorChecker Classic (4x6 grid) in an RGB image.
//
// Strategy:
// 1. Scan center 80% area for dark border pixels (chart frame)
// 2. Find bounding box of the chart
// 3. Partition into 4 rows x 6 cols
// 4. Output 24 patch regions with 30% inset margin
ColorCalibrationError process_detect_chart(const uint8_t* input,
                                            int width, int height,
                                            int channels, int bit_depth,
                                            ChartDetection* result) {
    if (!input || !result) return ColorCalibrationError::NullInput;
    if (!is_valid_dimensions(width, height)) return ColorCalibrationError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return ColorCalibrationError::InvalidBitDepth;
    if (channels != 3) return ColorCalibrationError::InvalidChannels;
    if (width < 40 || height < 30) return ColorCalibrationError::InvalidDimensions;

    int max_val = detail::safe_max_val(bit_depth);
    int dark_thresh = static_cast<int>(max_val * 0.15f);

    // Scan center region for dark border pixels
    int cx_min = width / 10, cx_max = width * 9 / 10;
    int cy_min = height / 10, cy_max = height * 9 / 10;

    // Find left/right/top/bottom of dark border
    int left = width, right = 0, top = height, bottom = 0;
    int dark_count = 0;

    for (int y = cy_min; y < cy_max; y += 2) {
        for (int x = cx_min; x < cx_max; x += 2) {
            int r = detail::read_pixel(input, x, y, width, 0, bit_depth);
            int g = detail::read_pixel(input, x, y, width, 1, bit_depth);
            int b = detail::read_pixel(input, x, y, width, 2, bit_depth);
            int lum = (r + g + b) / 3;

            if (lum < dark_thresh) {
                if (x < left) left = x;
                if (x > right) right = x;
                if (y < top) top = y;
                if (y > bottom) bottom = y;
                dark_count++;
            }
        }
    }

    // If insufficient dark pixels found, fall back to entire image
    if (dark_count < 100 || right <= left || bottom <= top) {
        left = width / 10;
        right = width * 9 / 10;
        top = height / 10;
        bottom = height * 9 / 10;
    }

    // Expand slightly to include chart area (dark border surrounds the patches)
    int border_margin = std::min(right - left, bottom - top) / 20;
    left = std::max(0, left - border_margin);
    right = std::min(width - 1, right + border_margin);
    top = std::max(0, top - border_margin);
    bottom = std::min(height - 1, bottom + border_margin);

    int chart_w = right - left;
    int chart_h = bottom - top;
    if (chart_w < 20 || chart_h < 15) return ColorCalibrationError::ChartNotFound;

    // Partition into 4 rows x 6 columns (plus border inset)
    // ColorChecker has ~15% border around patches
    const float border_ratio = 0.08f;
    float pw = static_cast<float>(chart_w) / 6.0f;
    float ph = static_cast<float>(chart_h) / 4.0f;

    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 6; col++) {
            int idx = row * 6 + col;
            float center_x = (left + (col + 0.5f) * pw) / static_cast<float>(width);
            float center_y = (top + (row + 0.5f) * ph) / static_cast<float>(height);
            float half_w = (pw * 0.35f) / static_cast<float>(width);  // 30% inset
            float half_h = (ph * 0.35f) / static_cast<float>(height);

            result->patches[idx].cx = center_x;
            result->patches[idx].cy = center_y;
            result->patches[idx].half_w = half_w;
            result->patches[idx].half_h = half_h;
            result->valid[idx] = true;
        }
    }

    return ColorCalibrationError::Ok;
}

} // namespace color_calibration
