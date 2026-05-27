#include "common.hpp"
#include "optimized.hpp"

namespace imop {

static const int kL7Weights7[49] = {
      1,  4,  9, 16,  9,  4,  1,
      4,  9, 16, 25, 16,  9,  4,
      9, 16, 25, 36, 25, 16,  9,
     16, 25, 36,  0, 36, 25, 16,
      9, 16, 25, 36, 25, 16,  9,
      4,  9, 16, 25, 16,  9,  4,
      1,  4,  9, 16,  9,  4,  1,
};

static void l7_core_8bit(const uint8_t* bayer, uint8_t* rgb,
                         int width, int height, const PatternOffsets& po) {
    using namespace pixel;
    const int w = width;
    const int h = height;
    const int r_row = po.r_row;
    const int r_col = po.r_col;
    const int b_row = po.b_row;
    const int b_col = po.b_col;

    DLOOP(y, 3, h - 3)
    {
        size_t row_y = static_cast<size_t>(y) * w;

        for (int x = 3; x < w - 3; x++) {
            int r_sum = 0, r_wsum = 0, g_sum = 0, g_wsum = 0, b_sum = 0, b_wsum = 0;
            const int* wptr = kL7Weights7;

            for (int dy = -3; dy <= 3; dy++) {
                int ny = y + dy;
                size_t row_ny = static_cast<size_t>(ny) * w;
                for (int dx = -3; dx <= 3; dx++) {
                    int wv = *wptr++;
                    if (wv == 0) continue;
                    int nx = x + dx;
                    int is_r = ((ny & 1) == r_row) && ((nx & 1) == r_col);
                    int is_b = ((ny & 1) == b_row) && ((nx & 1) == b_col);
                    int pv = bayer[row_ny + nx];
                    if (is_r) { r_sum += pv * wv; r_wsum += wv; }
                    else if (is_b) { b_sum += pv * wv; b_wsum += wv; }
                    else { g_sum += pv * wv; g_wsum += wv; }
                }
            }

            int at_r = ((y & 1) == r_row) && ((x & 1) == r_col);
            int at_b = ((y & 1) == b_row) && ((x & 1) == b_col);
            int at_g = !at_r && !at_b;

            int r_val = at_r ? bayer[row_y + x] : (r_wsum > 0 ? (r_sum + r_wsum / 2) / r_wsum : 0);
            int g_val = at_g ? bayer[row_y + x] : (g_wsum > 0 ? (g_sum + g_wsum / 2) / g_wsum : 0);
            int b_val = at_b ? bayer[row_y + x] : (b_wsum > 0 ? (b_sum + b_wsum / 2) / b_wsum : 0);

            size_t rgb_idx = (row_y + x) * 3;
            rgb[rgb_idx + 0] = static_cast<uint8_t>(r_val < 0 ? 0 : (r_val > 255 ? 255 : r_val));
            rgb[rgb_idx + 1] = static_cast<uint8_t>(g_val < 0 ? 0 : (g_val > 255 ? 255 : g_val));
            rgb[rgb_idx + 2] = static_cast<uint8_t>(b_val < 0 ? 0 : (b_val > 255 ? 255 : b_val));
        }
    }
}

DemosaicError process_l7(const uint8_t* bayer, uint8_t* rgb,
                int width, int height, BayerPattern pattern, int bit_depth, bool is_packed) {
    DemosaicError err = validate_demosaic_inputs(bayer, rgb, width, height, bit_depth);
    if (err != DemosaicError::Ok) return err;
    if (width < 8 || height < 8) return DemosaicError::ImageTooSmall;

    using namespace pixel;

    PatternOffsets po = PatternOffsets::from_pattern(pattern);
    const int max_val = safe_max_val(bit_depth);

    if (bit_depth <= 8 && !is_packed && has_avx2()) {
        process_l7_optimized(bayer, rgb, width, height, po, bit_depth, is_packed);
        detail::fill_rgb_borders(rgb, width, height, bit_depth, algorithm_window_size(DemosaicAlgorithm::L7) / 2);
        return DemosaicError::Ok;
    }

    if (bit_depth <= 8 && !is_packed) {
        l7_core_8bit(bayer, rgb, width, height, po);
        detail::fill_rgb_borders(rgb, width, height, bit_depth, algorithm_window_size(DemosaicAlgorithm::L7) / 2);
        return DemosaicError::Ok;
    }

    DLOOP(y, 3, height - 3)
    {
        for (int x = 3; x < width - 3; x++) {
            int r_sum = 0, r_wsum = 0, g_sum = 0, g_wsum = 0, b_sum = 0, b_wsum = 0;
            const int* wptr = kL7Weights7;

            for (int dy = -3; dy <= 3; dy++) {
                int ny = y + dy;
                for (int dx = -3; dx <= 3; dx++) {
                    int wv = *wptr++;
                    if (wv == 0) continue;
                    int nx = x + dx;
                    bool match_r = is_r_at(po, ny, nx);
                    bool match_b = is_b_at(po, ny, nx);
                    int pv = get_raw(bayer, nx, ny, width, bit_depth, is_packed, 0, height);
                    if (match_r) { r_sum += pv * wv; r_wsum += wv; }
                    else if (match_b) { b_sum += pv * wv; b_wsum += wv; }
                    else { g_sum += pv * wv; g_wsum += wv; }
                }
            }
            bool at_r = is_r_at(po, y, x);
            bool at_b = is_b_at(po, y, x);
            bool at_g = !at_r && !at_b;

            int r_val = at_r ? get_raw(bayer, x, y, width, bit_depth, is_packed, 0, height) : (r_wsum > 0 ? (r_sum + r_wsum / 2) / r_wsum : 0);
            int g_val = at_g ? get_raw(bayer, x, y, width, bit_depth, is_packed, 0, height) : (g_wsum > 0 ? (g_sum + g_wsum / 2) / g_wsum : 0);
            int b_val = at_b ? get_raw(bayer, x, y, width, bit_depth, is_packed, 0, height) : (b_wsum > 0 ? (b_sum + b_wsum / 2) / b_wsum : 0);

            if (r_val < 0) r_val = 0; else if (r_val > max_val) r_val = max_val;
            if (g_val < 0) g_val = 0; else if (g_val > max_val) g_val = max_val;
            if (b_val < 0) b_val = 0; else if (b_val > max_val) b_val = max_val;

            set_rgb_raw(rgb, x, y, width, r_val, g_val, b_val, bit_depth);
        }
    }

    detail::fill_rgb_borders(rgb, width, height, bit_depth, algorithm_window_size(DemosaicAlgorithm::L7) / 2);

    return DemosaicError::Ok;
}

} // namespace imop
