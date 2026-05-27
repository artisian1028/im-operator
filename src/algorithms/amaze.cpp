#include "common.hpp"
#include <cmath>
#include <algorithm>

namespace imop {

static void amaze_core_8bit(const uint8_t* bayer, uint8_t* rgb,
                            int width, int height, const PatternOffsets& po) {
    using namespace pixel;
    const int w = width, h = height;
    size_t total = static_cast<size_t>(w) * h;
    auto& pool = detail::thread_local_pool();
    auto& G = pool.get(total, 0);
    auto& R = pool.get(total, 1);
    auto& B = pool.get(total, 2);
    std::fill(G.begin(), G.end(), -1.0f);
    std::fill(R.begin(), R.end(), 0.0f);
    std::fill(B.begin(), B.end(), 0.0f);

    DLOOP(y, 2, h - 2)
    {
        size_t row_y   = static_cast<size_t>(y) * w;
        size_t row_ym1 = static_cast<size_t>(y - 1) * w;
        size_t row_ym2 = static_cast<size_t>(y - 2) * w;
        size_t row_yp1 = static_cast<size_t>(y + 1) * w;
        size_t row_yp2 = static_cast<size_t>(y + 2) * w;

        for (int x = 2; x < w - 2; ++x) {
            size_t idx = row_y + x;
            if (!is_r_at(po, y, x) && !is_b_at(po, y, x)) {
                G[idx] = static_cast<float>(bayer[idx]);
            } else {
                float raw_c  = static_cast<float>(bayer[idx]);
                float raw_hm1 = static_cast<float>(bayer[row_y + (x - 1)]);
                float raw_hp1 = static_cast<float>(bayer[row_y + (x + 1)]);
                float raw_hm2 = static_cast<float>(bayer[row_y + (x - 2)]);
                float raw_hp2 = static_cast<float>(bayer[row_y + (x + 2)]);
                float raw_vm1 = static_cast<float>(bayer[row_ym1 + x]);
                float raw_vp1 = static_cast<float>(bayer[row_yp1 + x]);
                float raw_vm2 = static_cast<float>(bayer[row_ym2 + x]);
                float raw_vp2 = static_cast<float>(bayer[row_yp2 + x]);

                float gradH = std::abs(raw_hm1 - raw_hp1) +
                              std::abs(2.0f * raw_c - raw_hm2 - raw_hp2);
                float gradV = std::abs(raw_vm1 - raw_vp1) +
                              std::abs(2.0f * raw_c - raw_vm2 - raw_vp2);
                float wh = 1.0f / (1.0f + gradH);
                float wv = 1.0f / (1.0f + gradV);
                float sum_w = wh + wv;
                wh /= sum_w; wv /= sum_w;
                float gh = (raw_hm1 + raw_hp1) * 0.5f;
                float gv = (raw_vm1 + raw_vp1) * 0.5f;
                float laplacianH = 2.0f * raw_c - raw_hm2 - raw_hp2;
                float laplacianV = 2.0f * raw_c - raw_vm2 - raw_vp2;
                gh += laplacianH * 0.25f;
                gv += laplacianV * 0.25f;
                G[idx] = wh * gh + wv * gv;
            }
        }
    }

    DLOOP(y, 1, h - 1)
    {
        size_t row_y   = static_cast<size_t>(y) * w;
        size_t row_ym1 = static_cast<size_t>(y - 1) * w;
        size_t row_yp1 = static_cast<size_t>(y + 1) * w;

        for (int x = 1; x < w - 1; ++x) {
            size_t idx = row_y + x;
            float g = G[idx];
            bool at_r = is_r_at(po, y, x);
            bool at_b = is_b_at(po, y, x);

            if (at_r) {
                R[idx] = static_cast<float>(bayer[idx]);
                float cd_sum = 0.0f; int cd_cnt = 0;
                int diags[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
                for (int k = 0; k < 4; k++) {
                    int nx = x + diags[k][0], ny = y + diags[k][1];
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h && is_b_at(po, ny, nx)) {
                        cd_sum += static_cast<float>(bayer[static_cast<size_t>(ny) * w + nx])
                                  - G[static_cast<size_t>(ny) * w + nx];
                        cd_cnt++;
                    }
                }
                B[idx] = g + (cd_cnt > 0 ? cd_sum / static_cast<float>(cd_cnt) : 0.0f);
            } else if (at_b) {
                B[idx] = static_cast<float>(bayer[idx]);
                float cd_sum = 0.0f; int cd_cnt = 0;
                int diags[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
                for (int k = 0; k < 4; k++) {
                    int nx = x + diags[k][0], ny = y + diags[k][1];
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h && is_r_at(po, ny, nx)) {
                        cd_sum += static_cast<float>(bayer[static_cast<size_t>(ny) * w + nx])
                                  - G[static_cast<size_t>(ny) * w + nx];
                        cd_cnt++;
                    }
                }
                R[idx] = g + (cd_cnt > 0 ? cd_sum / static_cast<float>(cd_cnt) : 0.0f);
            } else {
                float r_cd = 0.0f, b_cd = 0.0f; int r_cnt = 0, b_cnt = 0;
                if (is_r_at(po, y-1, x)) { r_cd += static_cast<float>(bayer[row_ym1 + x]) - G[row_ym1 + x]; r_cnt++; }
                if (is_r_at(po, y+1, x)) { r_cd += static_cast<float>(bayer[row_yp1 + x]) - G[row_yp1 + x]; r_cnt++; }
                if (is_r_at(po, y, x-1)) { r_cd += static_cast<float>(bayer[row_y + (x - 1)]) - G[row_y + (x - 1)]; r_cnt++; }
                if (is_r_at(po, y, x+1)) { r_cd += static_cast<float>(bayer[row_y + (x + 1)]) - G[row_y + (x + 1)]; r_cnt++; }
                if (is_b_at(po, y-1, x)) { b_cd += static_cast<float>(bayer[row_ym1 + x]) - G[row_ym1 + x]; b_cnt++; }
                if (is_b_at(po, y+1, x)) { b_cd += static_cast<float>(bayer[row_yp1 + x]) - G[row_yp1 + x]; b_cnt++; }
                if (is_b_at(po, y, x-1)) { b_cd += static_cast<float>(bayer[row_y + (x - 1)]) - G[row_y + (x - 1)]; b_cnt++; }
                if (is_b_at(po, y, x+1)) { b_cd += static_cast<float>(bayer[row_y + (x + 1)]) - G[row_y + (x + 1)]; b_cnt++; }
                R[idx] = g + (r_cnt > 0 ? r_cd / static_cast<float>(r_cnt) : 0.0f);
                B[idx] = g + (b_cnt > 0 ? b_cd / static_cast<float>(b_cnt) : 0.0f);
            }
        }
    }

    detail::fill_intermediate_borders(R.data(), G.data(), B.data(), width, height, 2);

    const int max_val = 255;
    DLOOP(y, 0, height) {
        for (int x = 0; x < width; x++) {
            size_t idx = static_cast<size_t>(y) * width + x;
            int ro = static_cast<int>(R[idx] + 0.5f); if (ro<0) ro=0; else if (ro>max_val) ro=max_val;
            int go = static_cast<int>(G[idx] + 0.5f); if (go<0) go=0; else if (go>max_val) go=max_val;
            int bo = static_cast<int>(B[idx] + 0.5f); if (bo<0) bo=0; else if (bo>max_val) bo=max_val;
            set_rgb_8(rgb, x, y, width, ro, go, bo);
        }
    }
}

DemosaicError process_amaze(const uint8_t* bayer, uint8_t* rgb,
                           int width, int height, BayerPattern pattern, int bit_depth, bool is_packed) {
    DemosaicError err = validate_demosaic_inputs(bayer, rgb, width, height, bit_depth);
    if (err != DemosaicError::Ok) return err;
    if (width < 6 || height < 6) return DemosaicError::ImageTooSmall;

    using namespace pixel;

    PatternOffsets po = PatternOffsets::from_pattern(pattern);

    if (bit_depth <= 8 && !is_packed) {
        amaze_core_8bit(bayer, rgb, width, height, po);
        detail::fill_rgb_borders(rgb, width, height, bit_depth, algorithm_window_size(DemosaicAlgorithm::AMAZE) / 2);
        return DemosaicError::Ok;
    }

    auto raw = [&](int px, int py) -> float {
        return static_cast<float>(get_raw(bayer, px, py, width, bit_depth, is_packed, 0, height));
    };

    size_t total = static_cast<size_t>(width) * height;
    auto& pool = detail::thread_local_pool();
    auto& G = pool.get(total, 0);
    auto& R = pool.get(total, 1);
    auto& B = pool.get(total, 2);
    std::fill(G.begin(), G.end(), -1.0f);
    std::fill(R.begin(), R.end(), 0.0f);
    std::fill(B.begin(), B.end(), 0.0f);

    DLOOP(y, 2, height - 2)
    {
        for (int x = 2; x < width - 2; ++x) {
            size_t idx = static_cast<size_t>(y) * width + x;
            if (!is_r_at(po, y, x) && !is_b_at(po, y, x)) {
                G[idx] = raw(x, y);
            } else {
                float gradH = std::abs(raw(x-1, y) - raw(x+1, y)) +
                              std::abs(2.0f * raw(x, y) - raw(x-2, y) - raw(x+2, y));
                float gradV = std::abs(raw(x, y-1) - raw(x, y+1)) +
                              std::abs(2.0f * raw(x, y) - raw(x, y-2) - raw(x, y+2));
                float wh = 1.0f / (1.0f + gradH);
                float wv = 1.0f / (1.0f + gradV);
                float sum_w = wh + wv;
                wh /= sum_w; wv /= sum_w;
                float gh = (raw(x-1, y) + raw(x+1, y)) * 0.5f;
                float gv = (raw(x, y-1) + raw(x, y+1)) * 0.5f;
                float laplacianH = 2.0f * raw(x, y) - raw(x-2, y) - raw(x+2, y);
                float laplacianV = 2.0f * raw(x, y) - raw(x, y-2) - raw(x, y+2);
                gh += laplacianH * 0.25f;
                gv += laplacianV * 0.25f;
                G[idx] = wh * gh + wv * gv;
            }
        }
    }

    DLOOP(y, 1, height - 1)
    {
        for (int x = 1; x < width - 1; ++x) {
            if (is_r_at(po, y, x) || is_b_at(po, y, x)) {
                detail::interpolate_rb_at_rb_positions(R.data(), B.data(), G.data(), x, y, width, po, raw);
            } else {
                detail::interpolate_rb_at_g_positions(R.data(), B.data(), G.data(), x, y, width, po, raw);
            }
        }
    }

    detail::fill_intermediate_borders(R.data(), G.data(), B.data(), width, height, 2);

    detail::write_float_planes_to_rgb(R.data(), G.data(), B.data(), rgb, width, height, bit_depth);

    return DemosaicError::Ok;
}

} // namespace imop
