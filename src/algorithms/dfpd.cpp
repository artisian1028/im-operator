#include "common.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace imop {

static void dfpd_core_8bit(const uint8_t* bayer, uint8_t* rgb,
                           int width, int height, const PatternOffsets& po) {
    using namespace pixel;
    const int w = width, h = height;
    const size_t total = static_cast<size_t>(w) * h;
    const int kSentinel = -1;
    std::vector<int> G(total, kSentinel);

    DLOOP(y, 2, h - 2)
    {
        size_t row_y     = static_cast<size_t>(y) * w;
        size_t row_ym1   = static_cast<size_t>(y - 1) * w;
        size_t row_ym2   = static_cast<size_t>(y - 2) * w;
        size_t row_yp1   = static_cast<size_t>(y + 1) * w;
        size_t row_yp2   = static_cast<size_t>(y + 2) * w;

        for (int x = 2; x < w - 2; x++) {
            size_t idx = row_y + x;
            if (!is_r_at(po, y, x) && !is_b_at(po, y, x)) {
                G[idx] = bayer[idx];
            } else {
                int v_xm1 = bayer[row_y + (x - 1)];
                int v_xp1 = bayer[row_y + (x + 1)];
                int v_xm2 = bayer[row_y + (x - 2)];
                int v_xp2 = bayer[row_y + (x + 2)];
                int v_ym1 = bayer[row_ym1 + x];
                int v_yp1 = bayer[row_yp1 + x];
                int v_ym2 = bayer[row_ym2 + x];
                int v_yp2 = bayer[row_yp2 + x];
                int v_c   = bayer[idx];

                int gh = std::abs(v_xm1 - v_xp1) + std::abs(2 * v_c - v_xm2 - v_xp2);
                int gv = std::abs(v_ym1 - v_yp1) + std::abs(2 * v_c - v_ym2 - v_yp2);
                int g_h = (v_xm1 + v_xp1 + 1) >> 1;
                int g_v = (v_ym1 + v_yp1 + 1) >> 1;

                if (gh < gv) {
                    int correction = v_c - ((v_xm2 + v_xp2 + 1) >> 1);
                    G[idx] = g_h + ((correction + 1) >> 1);
                } else if (gv < gh) {
                    int correction = v_c - ((v_ym2 + v_yp2 + 1) >> 1);
                    G[idx] = g_v + ((correction + 1) >> 1);
                } else {
                    G[idx] = (g_h + g_v + 1) >> 1;
                }
            }
        }
    }

    auto interp_g = [&](int px, int py) -> int {
        size_t i = static_cast<size_t>(py) * w + px;
        if (G[i] != kSentinel) return G[i];

        size_t row_y = static_cast<size_t>(py) * w;
        size_t row_ym1 = static_cast<size_t>(py - 1) * w;
        size_t row_ym2 = static_cast<size_t>(py - 2) * w;
        size_t row_yp1 = static_cast<size_t>(py + 1) * w;
        size_t row_yp2 = static_cast<size_t>(py + 2) * w;

        int v_xm1 = bayer[row_y + (px - 1)];
        int v_xp1 = bayer[row_y + (px + 1)];
        int v_xm2 = bayer[row_y + (px - 2)];
        int v_xp2 = bayer[row_y + (px + 2)];
        int v_ym1 = bayer[row_ym1 + px];
        int v_yp1 = bayer[row_yp1 + px];
        int v_ym2 = bayer[row_ym2 + px];
        int v_yp2 = bayer[row_yp2 + px];
        int v_c   = bayer[i];

        int lgh = std::abs(v_xm1 - v_xp1) + std::abs(2 * v_c - v_xm2 - v_xp2);
        int lgv = std::abs(v_ym1 - v_yp1) + std::abs(2 * v_c - v_ym2 - v_yp2);
        int g_h = (v_xm1 + v_xp1 + 1) >> 1;
        int g_v = (v_ym1 + v_yp1 + 1) >> 1;

        if (lgh < lgv) {
            int corr = v_c - ((v_xm2 + v_xp2 + 1) >> 1);
            return g_h + ((corr + 1) >> 1);
        } else if (lgv < lgh) {
            int corr = v_c - ((v_ym2 + v_yp2 + 1) >> 1);
            return g_v + ((corr + 1) >> 1);
        }
        return (g_h + g_v + 1) >> 1;
    };

    DLOOP(y, 3, h - 3)
    {
        size_t row_y   = static_cast<size_t>(y) * w;
        size_t row_ym1 = static_cast<size_t>(y - 1) * w;
        size_t row_yp1 = static_cast<size_t>(y + 1) * w;

        for (int x = 3; x < w - 3; x++) {
            int g_val = interp_g(x, y);
            int r_val, b_val_out;

            if (is_r_at(po, y, x)) {
                r_val = bayer[row_y + x];
                int cd_sum = 0, cd_cnt = 0;
                int diag[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
                for (int k = 0; k < 4; k++) {
                    int nx = x + diag[k][0], ny = y + diag[k][1];
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h && is_b_at(po, ny, nx)) {
                        cd_sum += static_cast<int>(bayer[static_cast<size_t>(ny) * w + nx]) - interp_g(nx, ny);
                        cd_cnt++;
                    }
                }
                b_val_out = cd_cnt > 0 ? (g_val + (cd_sum + cd_cnt / 2) / cd_cnt) : g_val;
            } else if (is_b_at(po, y, x)) {
                b_val_out = bayer[row_y + x];
                int cd_sum = 0, cd_cnt = 0;
                int diag[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
                for (int k = 0; k < 4; k++) {
                    int nx = x + diag[k][0], ny = y + diag[k][1];
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h && is_r_at(po, ny, nx)) {
                        cd_sum += static_cast<int>(bayer[static_cast<size_t>(ny) * w + nx]) - interp_g(nx, ny);
                        cd_cnt++;
                    }
                }
                r_val = cd_cnt > 0 ? (g_val + (cd_sum + cd_cnt / 2) / cd_cnt) : g_val;
            } else {
                int r_cd = 0, b_cd = 0, r_cnt = 0, b_cnt = 0;
                if (is_r_at(po, y-1, x)) { r_cd += static_cast<int>(bayer[row_ym1 + x]) - interp_g(x, y-1); r_cnt++; }
                if (is_r_at(po, y+1, x)) { r_cd += static_cast<int>(bayer[row_yp1 + x]) - interp_g(x, y+1); r_cnt++; }
                if (is_r_at(po, y, x-1)) { r_cd += static_cast<int>(bayer[row_y + (x-1)]) - interp_g(x-1, y); r_cnt++; }
                if (is_r_at(po, y, x+1)) { r_cd += static_cast<int>(bayer[row_y + (x+1)]) - interp_g(x+1, y); r_cnt++; }
                if (is_b_at(po, y-1, x)) { b_cd += static_cast<int>(bayer[row_ym1 + x]) - interp_g(x, y-1); b_cnt++; }
                if (is_b_at(po, y+1, x)) { b_cd += static_cast<int>(bayer[row_yp1 + x]) - interp_g(x, y+1); b_cnt++; }
                if (is_b_at(po, y, x-1)) { b_cd += static_cast<int>(bayer[row_y + (x-1)]) - interp_g(x-1, y); b_cnt++; }
                if (is_b_at(po, y, x+1)) { b_cd += static_cast<int>(bayer[row_y + (x+1)]) - interp_g(x+1, y); b_cnt++; }
                r_val = r_cnt > 0 ? (g_val + (r_cd + r_cnt / 2) / r_cnt) : g_val;
                b_val_out = b_cnt > 0 ? (g_val + (b_cd + b_cnt / 2) / b_cnt) : g_val;
            }

            set_rgb_8_clamp(rgb, x, y, width, r_val, g_val, b_val_out);
        }
    }
}

DemosaicError process_dfpd(const uint8_t* bayer, uint8_t* rgb,
                  int width, int height, BayerPattern pattern, int bit_depth, bool is_packed) {
    DemosaicError err = validate_demosaic_inputs(bayer, rgb, width, height, bit_depth);
    if (err != DemosaicError::Ok) return err;
    if (width < 12 || height < 12) return DemosaicError::ImageTooSmall;

    using namespace pixel;

    PatternOffsets po = PatternOffsets::from_pattern(pattern);

    if (bit_depth <= 8 && !is_packed) {
        dfpd_core_8bit(bayer, rgb, width, height, po);
        detail::fill_rgb_borders(rgb, width, height, bit_depth, algorithm_window_size(DemosaicAlgorithm::DFPD) / 2);
        return DemosaicError::Ok;
    }
    size_t total = static_cast<size_t>(width) * height;
    const float kSentinel = -std::numeric_limits<float>::infinity();
    std::vector<float> G(total, kSentinel);

    auto raw = [&](int px, int py) -> float {
        return static_cast<float>(get_raw(bayer, px, py, width, bit_depth, is_packed, 0, height));
    };

    DLOOP(y, 2, height - 2)
    {
        for (int x = 2; x < width - 2; x++) {
            size_t idx = static_cast<size_t>(y) * width + x;
            if (!is_r_at(po, y, x) && !is_b_at(po, y, x)) {
                G[idx] = raw(x, y);
            } else {
                float gh = std::abs(raw(x-1, y) - raw(x+1, y)) +
                           std::abs(2.0f * raw(x, y) - raw(x-2, y) - raw(x+2, y));
                float gv = std::abs(raw(x, y-1) - raw(x, y+1)) +
                           std::abs(2.0f * raw(x, y) - raw(x, y-2) - raw(x, y+2));
                float g_h = (raw(x-1, y) + raw(x+1, y)) * 0.5f;
                float g_v = (raw(x, y-1) + raw(x, y+1)) * 0.5f;
                if (gh < gv) {
                    float correction = raw(x, y) - (raw(x-2, y) + raw(x+2, y)) * 0.5f;
                    G[idx] = g_h + correction * 0.5f;
                } else if (gv < gh) {
                    float correction = raw(x, y) - (raw(x, y-2) + raw(x, y+2)) * 0.5f;
                    G[idx] = g_v + correction * 0.5f;
                } else {
                    G[idx] = (g_h + g_v) * 0.5f;
                }
            }
        }
    }

    auto interp_g = [&](int px, int py) -> float {
        size_t i = static_cast<size_t>(py) * width + px;
        if (G[i] != kSentinel) return G[i];
        float lgh = std::abs(raw(px-1, py) - raw(px+1, py)) +
                    std::abs(2.0f * raw(px, py) - raw(px-2, py) - raw(px+2, py));
        float lgv = std::abs(raw(px, py-1) - raw(px, py+1)) +
                    std::abs(2.0f * raw(px, py) - raw(px, py-2) - raw(px, py+2));
        float g_h = (raw(px-1, py) + raw(px+1, py)) * 0.5f;
        float g_v = (raw(px, py-1) + raw(px, py+1)) * 0.5f;
        if (lgh < lgv) {
            float corr = raw(px, py) - (raw(px-2, py) + raw(px+2, py)) * 0.5f;
            return g_h + corr * 0.5f;
        } else if (lgv < lgh) {
            float corr = raw(px, py) - (raw(px, py-2) + raw(px, py+2)) * 0.5f;
            return g_v + corr * 0.5f;
        }
        return (g_h + g_v) * 0.5f;
    };

    DLOOP(y, 3, height - 3)
    {
        for (int x = 3; x < width - 3; x++) {
            float g_val = interp_g(x, y);
            int r_val = 0, b_val_out = 0;

            if (is_r_at(po, y, x)) {
                r_val = static_cast<int>(raw(x, y) + 0.5f);
                float cd_sum = 0.0f; int cd_cnt = 0;
                int diag[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
                for (int k = 0; k < 4; k++) {
                    int nx = x + diag[k][0], ny = y + diag[k][1];
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && is_b_at(po, ny, nx)) {
                        cd_sum += raw(nx, ny) - interp_g(nx, ny); cd_cnt++;
                    }
                }
                b_val_out = static_cast<int>(g_val + (cd_cnt > 0 ? cd_sum / cd_cnt : 0.0f) + 0.5f);
            } else if (is_b_at(po, y, x)) {
                b_val_out = static_cast<int>(raw(x, y) + 0.5f);
                float cd_sum = 0.0f; int cd_cnt = 0;
                int diag[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
                for (int k = 0; k < 4; k++) {
                    int nx = x + diag[k][0], ny = y + diag[k][1];
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && is_r_at(po, ny, nx)) {
                        cd_sum += raw(nx, ny) - interp_g(nx, ny); cd_cnt++;
                    }
                }
                r_val = static_cast<int>(g_val + (cd_cnt > 0 ? cd_sum / cd_cnt : 0.0f) + 0.5f);
            } else {
                float r_cd = 0.0f, b_cd = 0.0f; int r_cnt = 0, b_cnt = 0;
                if (is_r_at(po, y-1, x)) { r_cd += raw(x, y-1) - interp_g(x, y-1); r_cnt++; }
                if (is_r_at(po, y+1, x)) { r_cd += raw(x, y+1) - interp_g(x, y+1); r_cnt++; }
                if (is_r_at(po, y, x-1)) { r_cd += raw(x-1, y) - interp_g(x-1, y); r_cnt++; }
                if (is_r_at(po, y, x+1)) { r_cd += raw(x+1, y) - interp_g(x+1, y); r_cnt++; }
                if (is_b_at(po, y-1, x)) { b_cd += raw(x, y-1) - interp_g(x, y-1); b_cnt++; }
                if (is_b_at(po, y+1, x)) { b_cd += raw(x, y+1) - interp_g(x, y+1); b_cnt++; }
                if (is_b_at(po, y, x-1)) { b_cd += raw(x-1, y) - interp_g(x-1, y); b_cnt++; }
                if (is_b_at(po, y, x+1)) { b_cd += raw(x+1, y) - interp_g(x+1, y); b_cnt++; }
                r_val = static_cast<int>(g_val + (r_cnt > 0 ? r_cd / r_cnt : 0.0f) + 0.5f);
                b_val_out = static_cast<int>(g_val + (b_cnt > 0 ? b_cd / b_cnt : 0.0f) + 0.5f);
            }

            int g_out = static_cast<int>(g_val + 0.5f);
            set_rgb_clamped(rgb, x, y, width, r_val, g_out, b_val_out, bit_depth);
        }
    }

    detail::fill_rgb_borders(rgb, width, height, bit_depth, algorithm_window_size(DemosaicAlgorithm::DFPD) / 2);

    return DemosaicError::Ok;
}

} // namespace imop
