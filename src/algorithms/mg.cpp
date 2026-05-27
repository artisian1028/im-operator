#include "common.hpp"
#include "optimized.hpp"
#include <cmath>
#include <algorithm>

namespace imop {

DemosaicError process_mg(const uint8_t* bayer, uint8_t* rgb,
                        int width, int height, BayerPattern pattern, int bit_depth, bool is_packed) {
    DemosaicError err = validate_demosaic_inputs(bayer, rgb, width, height, bit_depth);
    if (err != DemosaicError::Ok) return err;
    if (width < 6 || height < 6) return DemosaicError::ImageTooSmall;

    using namespace pixel;

    PatternOffsets po = PatternOffsets::from_pattern(pattern);

    if (has_avx2()) {
        process_mg_optimized(bayer, rgb, width, height, po, bit_depth, is_packed);
        return DemosaicError::Ok;
    }

    if (bit_depth <= 8 && !is_packed) {
        const int w = width;
        const int h = height;
        const int max_val = safe_max_val(bit_depth);
        const size_t total = static_cast<size_t>(width) * height;
        const int r_row = po.r_row;
        const int r_col = po.r_col;
        const int b_row = po.b_row;
        const int b_col = po.b_col;

        std::vector<int> G(total, -1);
        std::vector<int> R(total, 0);
        std::vector<int> B(total, 0);

        DLOOP(y, 0, h) {
            size_t ry = static_cast<size_t>(y) * w;
            for (int x = 0; x < w; x++) {
                bool at_r = ((y & 1) == r_row) && ((x & 1) == r_col);
                bool at_b = ((y & 1) == b_row) && ((x & 1) == b_col);
                if (!at_r && !at_b) {
                    G[ry + x] = static_cast<int>(bayer[ry + x]);
                }
            }
        }

        DLOOP(y, 2, h - 2) {
            size_t ry   = static_cast<size_t>(y) * w;
            size_t rym1 = static_cast<size_t>(y - 1) * w;
            size_t rym2 = static_cast<size_t>(y - 2) * w;
            size_t ryp1 = static_cast<size_t>(y + 1) * w;
            size_t ryp2 = static_cast<size_t>(y + 2) * w;
            for (int x = 2; x < w - 2; x++) {
                bool at_r = ((y & 1) == r_row) && ((x & 1) == r_col);
                bool at_b = ((y & 1) == b_row) && ((x & 1) == b_col);
                if (at_r || at_b) {
                    int c  = static_cast<int>(bayer[ry + x]);
                    int n  = static_cast<int>(bayer[rym1 + x]);
                    int s  = static_cast<int>(bayer[ryp1 + x]);
                    int e  = static_cast<int>(bayer[ry + (x + 1)]);
                    int wv = static_cast<int>(bayer[ry + (x - 1)]);
                    int n2 = static_cast<int>(bayer[rym2 + x]);
                    int s2 = static_cast<int>(bayer[ryp2 + x]);
                    int e2 = static_cast<int>(bayer[ry + (x + 2)]);
                    int w2 = static_cast<int>(bayer[ry + (x - 2)]);
                    int gv = (4*c + 2*(n+s+e+wv) - (n2+s2+e2+w2) + 4) >> 3;
                    if (gv < 0) gv = 0; else if (gv > max_val) gv = max_val;
                    G[ry + x] = gv;
                }
            }
        }

        DLOOP(y, 1, h - 1) {
            size_t ry   = static_cast<size_t>(y) * w;
            size_t rym1 = static_cast<size_t>(y - 1) * w;
            size_t ryp1 = static_cast<size_t>(y + 1) * w;
            for (int x = 1; x < w - 1; x++) {
                size_t idx = ry + x;
                int g = G[idx];
                bool at_r = ((y & 1) == r_row) && ((x & 1) == r_col);
                bool at_b = ((y & 1) == b_row) && ((x & 1) == b_col);

                if (at_r) {
                    R[idx] = static_cast<int>(bayer[ry + x]);
                    int sum_cd = 0, cnt = 0;
                    if (((y-1)&1)==b_row && ((x-1)&1)==b_col) {
                        sum_cd += static_cast<int>(bayer[rym1+(x-1)]) - G[rym1+(x-1)]; cnt++;
                    }
                    if (((y-1)&1)==b_row && ((x+1)&1)==b_col) {
                        sum_cd += static_cast<int>(bayer[rym1+(x+1)]) - G[rym1+(x+1)]; cnt++;
                    }
                    if (((y+1)&1)==b_row && ((x-1)&1)==b_col) {
                        sum_cd += static_cast<int>(bayer[ryp1+(x-1)]) - G[ryp1+(x-1)]; cnt++;
                    }
                    if (((y+1)&1)==b_row && ((x+1)&1)==b_col) {
                        sum_cd += static_cast<int>(bayer[ryp1+(x+1)]) - G[ryp1+(x+1)]; cnt++;
                    }
                    B[idx] = g + (cnt > 0 ? ((sum_cd + cnt/2) / cnt) : 0);
                } else if (at_b) {
                    B[idx] = static_cast<int>(bayer[ry + x]);
                    int sum_cd = 0, cnt = 0;
                    if (((y-1)&1)==r_row && ((x-1)&1)==r_col) {
                        sum_cd += static_cast<int>(bayer[rym1+(x-1)]) - G[rym1+(x-1)]; cnt++;
                    }
                    if (((y-1)&1)==r_row && ((x+1)&1)==r_col) {
                        sum_cd += static_cast<int>(bayer[rym1+(x+1)]) - G[rym1+(x+1)]; cnt++;
                    }
                    if (((y+1)&1)==r_row && ((x-1)&1)==r_col) {
                        sum_cd += static_cast<int>(bayer[ryp1+(x-1)]) - G[ryp1+(x-1)]; cnt++;
                    }
                    if (((y+1)&1)==r_row && ((x+1)&1)==r_col) {
                        sum_cd += static_cast<int>(bayer[ryp1+(x+1)]) - G[ryp1+(x+1)]; cnt++;
                    }
                    R[idx] = g + (cnt > 0 ? ((sum_cd + cnt/2) / cnt) : 0);
                } else {
                    int r_cd = 0, b_cd = 0, rc = 0, bc = 0;
                    if (((y-1)&1)==r_row && ((x)&1)==r_col) {
                        r_cd += static_cast<int>(bayer[rym1+x]) - G[rym1+x]; rc++;
                    }
                    if (((y+1)&1)==r_row && ((x)&1)==r_col) {
                        r_cd += static_cast<int>(bayer[ryp1+x]) - G[ryp1+x]; rc++;
                    }
                    if (((y)&1)==r_row && ((x-1)&1)==r_col) {
                        r_cd += static_cast<int>(bayer[ry+(x-1)]) - G[ry+(x-1)]; rc++;
                    }
                    if (((y)&1)==r_row && ((x+1)&1)==r_col) {
                        r_cd += static_cast<int>(bayer[ry+(x+1)]) - G[ry+(x+1)]; rc++;
                    }
                    if (((y-1)&1)==b_row && ((x)&1)==b_col) {
                        b_cd += static_cast<int>(bayer[rym1+x]) - G[rym1+x]; bc++;
                    }
                    if (((y+1)&1)==b_row && ((x)&1)==b_col) {
                        b_cd += static_cast<int>(bayer[ryp1+x]) - G[ryp1+x]; bc++;
                    }
                    if (((y)&1)==b_row && ((x-1)&1)==b_col) {
                        b_cd += static_cast<int>(bayer[ry+(x-1)]) - G[ry+(x-1)]; bc++;
                    }
                    if (((y)&1)==b_row && ((x+1)&1)==b_col) {
                        b_cd += static_cast<int>(bayer[ry+(x+1)]) - G[ry+(x+1)]; bc++;
                    }
                    R[idx] = g + (rc > 0 ? (r_cd + rc/2) / rc : 0);
                    B[idx] = g + (bc > 0 ? (b_cd + bc/2) / bc : 0);
                }
            }
        }

        detail::fill_intermediate_borders_int(R.data(), G.data(), B.data(), width, height, 2);

        DLOOP(y, 0, h) {
            size_t ry = static_cast<size_t>(y) * w;
            for (int x = 0; x < w; x++) {
                size_t idx = ry + x;
                int ro = R[idx]; if (ro<0) ro=0; else if (ro>max_val) ro=max_val;
                int go = G[idx]; if (go<0) go=0; else if (go>max_val) go=max_val;
                int bo = B[idx]; if (bo<0) bo=0; else if (bo>max_val) bo=max_val;
                set_rgb_8(rgb, x, y, width, ro, go, bo);
            }
        }
        return DemosaicError::Ok;
    }

    size_t total = static_cast<size_t>(width) * height;

    auto raw = [&](int px, int py) -> float {
        return static_cast<float>(get_raw(bayer, px, py, width, bit_depth, is_packed, 0, height));
    };

    std::vector<float> G(total, -1.0f);
    std::vector<float> R(total, 0.0f);
    std::vector<float> B(total, 0.0f);

    DLOOP(y, 0, height)
    {
        for (int x = 0; x < width; x++) {
            if (!is_r_at(po, y, x) && !is_b_at(po, y, x)) {
                G[static_cast<size_t>(y) * width + x] = raw(x, y);
            }
        }
    }

    DLOOP(y, 2, height - 2)
    {
        for (int x = 2; x < width - 2; x++) {
            if (is_r_at(po, y, x) || is_b_at(po, y, x)) {
                size_t idx = static_cast<size_t>(y) * width + x;
                float n  = raw(x, y - 1);
                float s  = raw(x, y + 1);
                float e  = raw(x + 1, y);
                float w  = raw(x - 1, y);
                float n2 = raw(x, y - 2);
                float s2 = raw(x, y + 2);
                float e2 = raw(x + 2, y);
                float w2 = raw(x - 2, y);
                float c  = raw(x, y);
                G[idx] = (4.0f * c + 2.0f * (n + s + e + w) - (n2 + s2 + e2 + w2)) / 8.0f;
            }
        }
    }

    DLOOP(y, 1, height - 1)
    {
        for (int x = 1; x < width - 1; x++) {
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
