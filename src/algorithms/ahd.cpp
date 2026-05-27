#include "common.hpp"
#include <cmath>
#include <algorithm>

namespace imop {

static void apply_median_filter_3x3(float* plane, float* filtered, int width, int height) {
    if (width < 3 || height < 3) return;
#define MM_SWAP(a, b) do { float _t = w[a]; if (_t > w[b]) { w[a] = w[b]; w[b] = _t; } } while(0)
    DLOOP(y, 1, height - 1)
    {
        for (int x = 1; x < width - 1; x++) {
            float w[9];
            int vi = 0;
            for (int dy = -1; dy <= 1; dy++) {
                size_t row_off = static_cast<size_t>(y + dy) * width;
                for (int dx = -1; dx <= 1; dx++) {
                    w[vi++] = plane[row_off + (x + dx)];
                }
            }
            MM_SWAP(0,1); MM_SWAP(3,4); MM_SWAP(6,7);
            MM_SWAP(1,2); MM_SWAP(4,5); MM_SWAP(7,8);
            MM_SWAP(0,1); MM_SWAP(3,4); MM_SWAP(6,7);
            MM_SWAP(0,3); MM_SWAP(4,7); MM_SWAP(1,4);
            MM_SWAP(3,6); MM_SWAP(2,5); MM_SWAP(5,8);
            MM_SWAP(1,3); MM_SWAP(4,6); MM_SWAP(2,5);
            MM_SWAP(2,3); MM_SWAP(5,6); MM_SWAP(2,4);
            MM_SWAP(3,5); MM_SWAP(3,4);
            filtered[static_cast<size_t>(y) * width + x] = w[4];
        }
    }
#undef MM_SWAP
    for (int y = 1; y < height - 1; y++) {
        size_t frow_off = static_cast<size_t>(y) * width;
        for (int x = 1; x < width - 1; x++) {
            size_t idx = static_cast<size_t>(y) * width + x;
            plane[idx] = filtered[frow_off + x];
        }
    }
}

DemosaicError process_ahd(const uint8_t* bayer, uint8_t* rgb,
                 int width, int height, BayerPattern pattern, int bit_depth, bool is_packed) {
    DemosaicError err = validate_demosaic_inputs(bayer, rgb, width, height, bit_depth);
    if (err != DemosaicError::Ok) return err;
    if (width < 6 || height < 6) return DemosaicError::ImageTooSmall;

    using namespace pixel;

    PatternOffsets po = PatternOffsets::from_pattern(pattern);
    int max_val = safe_max_val(bit_depth);

    auto raw = [&](int px, int py) -> float {
        return static_cast<float>(get_clamped(bayer, px, py, width, height, bit_depth, is_packed));
    };

    size_t total = static_cast<size_t>(width) * height;
    auto& pool = detail::thread_local_pool();
    auto& Rh = pool.get(total, 0);
    auto& Gh = pool.get(total, 1);
    auto& Bh = pool.get(total, 2);
    auto& Rv = pool.get(total, 3);
    auto& Gv = pool.get(total, 4);
    auto& Bv = pool.get(total, 5);
    auto& Lh = pool.get(total, 6);
    auto& Mh = pool.get(total, 7);
    auto& Lv = pool.get(total, 8);
    auto& Mv = pool.get(total, 9);
    std::fill(Rh.begin(), Rh.end(), 0.0f);
    std::fill(Gh.begin(), Gh.end(), 0.0f);
    std::fill(Bh.begin(), Bh.end(), 0.0f);
    std::fill(Rv.begin(), Rv.end(), 0.0f);
    std::fill(Gv.begin(), Gv.end(), 0.0f);
    std::fill(Bv.begin(), Bv.end(), 0.0f);

    DLOOP(y, 0, height)
    {
        for (int x = 0; x < width; ++x) {
            size_t idx = static_cast<size_t>(y) * width + x;
            float r, g, b_val;
            if (!is_r_at(po, y, x) && !is_b_at(po, y, x)) {
                g = raw(x, y);
                if (is_r_at(po, y, x-1) || is_r_at(po, y, x+1)) {
                    r = (raw(x-1, y) + raw(x+1, y)) * 0.5f;
                    b_val = (raw(x, y-1) + raw(x, y+1)) * 0.5f;
                } else {
                    b_val = (raw(x-1, y) + raw(x+1, y)) * 0.5f;
                    r = (raw(x, y-1) + raw(x, y+1)) * 0.5f;
                }
            } else if (is_r_at(po, y, x)) {
                r = raw(x, y);
                g = (raw(x-1, y) + raw(x+1, y)) * 0.5f;
                b_val = (raw(x-1, y-1) + raw(x+1, y-1) + raw(x-1, y+1) + raw(x+1, y+1)) * 0.25f;
            } else {
                b_val = raw(x, y);
                g = (raw(x-1, y) + raw(x+1, y)) * 0.5f;
                r = (raw(x-1, y-1) + raw(x+1, y-1) + raw(x-1, y+1) + raw(x+1, y+1)) * 0.25f;
            }
            Rh[idx] = r; Gh[idx] = g; Bh[idx] = b_val;
        }
    }

    DLOOP(y, 0, height)
    {
        for (int x = 0; x < width; ++x) {
            size_t idx = static_cast<size_t>(y) * width + x;
            float r, g, b_val;
            if (!is_r_at(po, y, x) && !is_b_at(po, y, x)) {
                g = raw(x, y);
                if (is_r_at(po, y-1, x) || is_r_at(po, y+1, x)) {
                    r = (raw(x, y-1) + raw(x, y+1)) * 0.5f;
                    b_val = (raw(x-1, y) + raw(x+1, y)) * 0.5f;
                } else {
                    b_val = (raw(x, y-1) + raw(x, y+1)) * 0.5f;
                    r = (raw(x-1, y) + raw(x+1, y)) * 0.5f;
                }
            } else if (is_r_at(po, y, x)) {
                r = raw(x, y);
                g = (raw(x, y-1) + raw(x, y+1)) * 0.5f;
                b_val = (raw(x-1, y-1) + raw(x-1, y+1) + raw(x+1, y-1) + raw(x+1, y+1)) * 0.25f;
            } else {
                b_val = raw(x, y);
                g = (raw(x, y-1) + raw(x, y+1)) * 0.5f;
                r = (raw(x-1, y-1) + raw(x-1, y+1) + raw(x+1, y-1) + raw(x+1, y+1)) * 0.25f;
            }
            Rv[idx] = r; Gv[idx] = g; Bv[idx] = b_val;
        }
    }

    auto& filtered = pool.get(total, 10);
    for (size_t i = 0; i < total; ++i) {
        Lh[i] = Rh[i] - Gh[i]; Mh[i] = Bh[i] - Gh[i];
        Lv[i] = Rv[i] - Gv[i]; Mv[i] = Bv[i] - Gv[i];
    }

    apply_median_filter_3x3(Lh.data(), filtered.data(), width, height);
    apply_median_filter_3x3(Mh.data(), filtered.data(), width, height);
    apply_median_filter_3x3(Lv.data(), filtered.data(), width, height);
    apply_median_filter_3x3(Mv.data(), filtered.data(), width, height);

    auto compute_variance = [&](const std::vector<float>& plane, int cx, int cy) -> float {
        float mean = 0.0f; int cnt = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = std::max(0, std::min(cx + dx, width - 1));
                int ny = std::max(0, std::min(cy + dy, height - 1));
                mean += plane[static_cast<size_t>(ny) * width + nx]; cnt++;
            }
        }
        mean /= cnt;
        float var = 0.0f;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = std::max(0, std::min(cx + dx, width - 1));
                int ny = std::max(0, std::min(cy + dy, height - 1));
                float diff = plane[static_cast<size_t>(ny) * width + nx] - mean;
                var += diff * diff;
            }
        }
        return var / cnt;
    };

    DLOOP(y, 0, height)
    {
        for (int x = 0; x < width; ++x) {
            float varLh = compute_variance(Lh, x, y);
            float varMh = compute_variance(Mh, x, y);
            float varLv = compute_variance(Lv, x, y);
            float varMv = compute_variance(Mv, x, y);

            size_t idx = static_cast<size_t>(y) * width + x;
            float finalR, finalG, finalB;
            if ((varLh + varMh) < (varLv + varMv)) {
                finalR = Rh[idx]; finalG = Gh[idx]; finalB = Bh[idx];
            } else {
                finalR = Rv[idx]; finalG = Gv[idx]; finalB = Bv[idx];
            }

            int r_out = std::max(0, std::min(static_cast<int>(finalR + 0.5f), max_val));
            int g_out = std::max(0, std::min(static_cast<int>(finalG + 0.5f), max_val));
            int b_out = std::max(0, std::min(static_cast<int>(finalB + 0.5f), max_val));
            set_rgb_raw(rgb, x, y, width, r_out, g_out, b_out, bit_depth);
        }
    }

    return DemosaicError::Ok;
}

} // namespace imop
