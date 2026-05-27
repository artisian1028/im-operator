#ifndef IMOP_ALGORITHMS_INTERP_CORE_HPP
#define IMOP_ALGORITHMS_INTERP_CORE_HPP

#include "imop/pixel_utils.hpp"
#include "imop/algorithms.hpp"
#include <algorithm>
#include <cstdint>
#include <functional>

namespace imop {
namespace detail {

// ── Generic interpolation helpers ──────────────────────────────────

// Clamp an integer value to [0, max_val].
template <typename T>
inline T clamp_val(T v, T max_val) {
    if (v < 0) return 0;
    if (v > max_val) return max_val;
    return v;
}

// ── 5×5 HQLI core (also used as the fallback for SUPER_FAST's
//    more-general path; SUPER_FAST itself has a smaller 1×1 window) ──

// PixelAccessor: callable (x, y) -> int that retrieves the raw Bayer value.
// WriteFunc:       callable (x, y, r, g, b) that stores the RGB triplet.
template <typename PixelAccessor, typename WriteFunc>
void hqli_core(int width, int height, const PatternOffsets& po,
               int max_val, PixelAccessor&& pix, WriteFunc&& write) {
    using pixel::is_r_at;
    using pixel::is_b_at;

    DLOOP(y, 2, height - 2) {
        const bool is_gr_row = ((y & 1) == po.r_row);
        for (int x = 2; x < width - 2; x++) {
            int r_val, g_val, b_val;

            if (is_r_at(po, y, x)) {
                r_val = pix(x, y);
                g_val = (pix(x - 1, y) + pix(x + 1, y) +
                         pix(x, y - 1) + pix(x, y + 1) + 2) / 4;
                b_val = (pix(x - 1, y - 1) + pix(x + 1, y - 1) +
                         pix(x - 1, y + 1) + pix(x + 1, y + 1) + 2) / 4;
            } else if (is_b_at(po, y, x)) {
                r_val = (pix(x - 1, y - 1) + pix(x + 1, y - 1) +
                         pix(x - 1, y + 1) + pix(x + 1, y + 1) + 2) / 4;
                g_val = (pix(x - 1, y) + pix(x + 1, y) +
                         pix(x, y - 1) + pix(x, y + 1) + 2) / 4;
                b_val = pix(x, y);
            } else {
                g_val = pix(x, y);
                if (is_gr_row) {
                    r_val = (pix(x - 1, y) + pix(x + 1, y) + 1) / 2;
                    b_val = (pix(x, y - 1) + pix(x, y + 1) + 1) / 2;
                } else {
                    b_val = (pix(x - 1, y) + pix(x + 1, y) + 1) / 2;
                    r_val = (pix(x, y - 1) + pix(x, y + 1) + 1) / 2;
                }
            }

            write(x, y,
                  clamp_val(r_val, max_val),
                  clamp_val(g_val, max_val),
                  clamp_val(b_val, max_val));
        }
    }
}

// ── SUPER_FAST 1×1 nearest-neighbor core ───────────────────────────

template <typename PixelAccessor, typename WriteFunc>
void super_fast_core(int width, int height, const PatternOffsets& po,
                     int max_val, PixelAccessor&& pix, WriteFunc&& write) {
    using pixel::is_r_at;
    using pixel::is_b_at;

    DLOOP(y, 0, height) {
        const bool is_gr_row = ((y & 1) == po.r_row);
        for (int x = 0; x < width; x++) {
            int r_val, g_val, b_val;

            if (is_r_at(po, y, x)) {
                r_val = pix(x, y);
                g_val = (pix(x - 1, y) + pix(x + 1, y) +
                         pix(x, y - 1) + pix(x, y + 1) + 2) / 4;
                int nx = (x & 1) ? x - 1 : x + 1;
                int ny = (y & 1) ? y - 1 : y + 1;
                ny = std::max(0, std::min(ny, height - 1));
                nx = std::max(0, std::min(nx, width  - 1));
                b_val = pix(nx, ny);
            } else if (is_b_at(po, y, x)) {
                b_val = pix(x, y);
                g_val = (pix(x - 1, y) + pix(x + 1, y) +
                         pix(x, y - 1) + pix(x, y + 1) + 2) / 4;
                int nx = (x & 1) ? x - 1 : x + 1;
                int ny = (y & 1) ? y - 1 : y + 1;
                ny = std::max(0, std::min(ny, height - 1));
                nx = std::max(0, std::min(nx, width  - 1));
                r_val = pix(nx, ny);
            } else {
                g_val = pix(x, y);
                if (is_gr_row) {
                    r_val = (pix(x - 1, y) + pix(x + 1, y) + 1) / 2;
                    b_val = (pix(x, y - 1) + pix(x, y + 1) + 1) / 2;
                } else {
                    b_val = (pix(x - 1, y) + pix(x + 1, y) + 1) / 2;
                    r_val = (pix(x, y - 1) + pix(x, y + 1) + 1) / 2;
                }
            }

            write(x, y,
                  clamp_val(r_val, max_val),
                  clamp_val(g_val, max_val),
                  clamp_val(b_val, max_val));
        }
    }
}

// ── Pre-built accessors for common cases ────────────────────────────

// 8-bit unpacked raw accessor: direct array indexing.
struct Accessor8bit {
    const uint8_t* bayer;
    int w, h;
    int operator()(int x, int y) const {
        if (x < 0) x = 0; else if (x >= w) x = w - 1;
        if (y < 0) y = 0; else if (y >= h) y = h - 1;
        return static_cast<int>(bayer[static_cast<size_t>(y) * w + x]);
    }
};

// 16-bit unpacked raw accessor (also handles 10/12/14 bit via
// pixel::get_raw_16).
struct Accessor16bit {
    const uint8_t* bayer;
    int w, bit_depth;
    int operator()(int x, int y) const {
        return pixel::get_raw_16(bayer, x, y, w, bit_depth);
    }
};

// Packed-data accessor using get_clamped for bounds safety.
struct AccessorPacked {
    const uint8_t* bayer;
    int w, h, bit_depth;
    int operator()(int x, int y) const {
        return pixel::get_clamped(bayer, x, y, w, h, bit_depth, true);
    }
};

// ── Pre-built write functions ───────────────────────────────────────

inline auto make_writer_8bit(uint8_t* rgb, int width) {
    return [=](int x, int y, int r, int g, int b) {
        size_t idx = (static_cast<size_t>(y) * width + x) * 3;
        rgb[idx + 0] = static_cast<uint8_t>(r);
        rgb[idx + 1] = static_cast<uint8_t>(g);
        rgb[idx + 2] = static_cast<uint8_t>(b);
    };
}

using WriterFunc = std::function<void(int, int, int, int, int)>;

inline WriterFunc make_writer_generic(uint8_t* rgb, int width, int bit_depth) {
    if (bit_depth <= 8) return make_writer_8bit(rgb, width);
    return [=](int x, int y, int r, int g, int b) {
        pixel::set_rgb_raw(rgb, x, y, width, r, g, b, bit_depth);
    };
}

} // namespace detail
} // namespace imop

#endif // IMOP_ALGORITHMS_INTERP_CORE_HPP
