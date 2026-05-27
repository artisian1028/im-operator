#include "common.hpp"
#include "optimized.hpp"
#include <algorithm>
#include <mutex>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace imop {

#ifdef __AVX2__

static bool s_avx2_detected = false;
static std::once_flag s_avx2_once;

bool has_avx2() {
    std::call_once(s_avx2_once, [] {
        int regs[4];
#if defined(_MSC_VER)
        __cpuid(regs, 7);
#elif defined(__GNUC__) || defined(__clang__)
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            regs[0] = static_cast<int>(eax);
            regs[1] = static_cast<int>(ebx);
            regs[2] = static_cast<int>(ecx);
            regs[3] = static_cast<int>(edx);
        } else {
            regs[0] = regs[1] = regs[2] = regs[3] = 0;
        }
#else
        regs[0] = regs[1] = regs[2] = regs[3] = 0;
#endif
        s_avx2_detected = (regs[1] & (1 << 5)) != 0;
    });
    return s_avx2_detected;
}

static inline uint8_t vec_extract_epu8(__m256i v, int idx) {
    alignas(32) uint8_t tmp[32];
    _mm256_storeu_si256((__m256i*)tmp, v);
    return tmp[idx];
}

static inline __m256i avg4_epu8(__m256i a, __m256i b, __m256i c, __m256i d) {
    __m256i lo_a = _mm256_unpacklo_epi8(a, _mm256_setzero_si256());
    __m256i hi_a = _mm256_unpackhi_epi8(a, _mm256_setzero_si256());
    __m256i lo_b = _mm256_unpacklo_epi8(b, _mm256_setzero_si256());
    __m256i hi_b = _mm256_unpackhi_epi8(b, _mm256_setzero_si256());
    __m256i lo_c = _mm256_unpacklo_epi8(c, _mm256_setzero_si256());
    __m256i hi_c = _mm256_unpackhi_epi8(c, _mm256_setzero_si256());
    __m256i lo_d = _mm256_unpacklo_epi8(d, _mm256_setzero_si256());
    __m256i hi_d = _mm256_unpackhi_epi8(d, _mm256_setzero_si256());

    __m256i lo_sum = _mm256_add_epi16(_mm256_add_epi16(lo_a, lo_b),
                                       _mm256_add_epi16(lo_c, lo_d));
    lo_sum = _mm256_add_epi16(lo_sum, _mm256_set1_epi16(2));
    __m256i hi_sum = _mm256_add_epi16(_mm256_add_epi16(hi_a, hi_b),
                                       _mm256_add_epi16(hi_c, hi_d));
    hi_sum = _mm256_add_epi16(hi_sum, _mm256_set1_epi16(2));

    __m256i lo_res = _mm256_srli_epi16(lo_sum, 2);
    __m256i hi_res = _mm256_srli_epi16(hi_sum, 2);
    return _mm256_packus_epi16(lo_res, hi_res);
}

static inline __m256i avg2_epu8(__m256i a, __m256i b) {
    __m256i lo_a = _mm256_unpacklo_epi8(a, _mm256_setzero_si256());
    __m256i hi_a = _mm256_unpackhi_epi8(a, _mm256_setzero_si256());
    __m256i lo_b = _mm256_unpacklo_epi8(b, _mm256_setzero_si256());
    __m256i hi_b = _mm256_unpackhi_epi8(b, _mm256_setzero_si256());

    __m256i lo_sum = _mm256_add_epi16(_mm256_add_epi16(lo_a, lo_b), _mm256_set1_epi16(1));
    __m256i hi_sum = _mm256_add_epi16(_mm256_add_epi16(hi_a, hi_b), _mm256_set1_epi16(1));

    __m256i lo_res = _mm256_srli_epi16(lo_sum, 1);
    __m256i hi_res = _mm256_srli_epi16(hi_sum, 1);
    return _mm256_packus_epi16(lo_res, hi_res);
}

static inline void store_rgb_interleaved_32(uint8_t* rgb, size_t base,
                                            __m256i r, __m256i g, __m256i b) {
    alignas(32) uint8_t tmp_r[32], tmp_g[32], tmp_b[32];
    _mm256_storeu_si256((__m256i*)tmp_r, r);
    _mm256_storeu_si256((__m256i*)tmp_g, g);
    _mm256_storeu_si256((__m256i*)tmp_b, b);
    for (int i = 0; i < 32; i++) {
        rgb[base + i * 3 + 0] = tmp_r[i];
        rgb[base + i * 3 + 1] = tmp_g[i];
        rgb[base + i * 3 + 2] = tmp_b[i];
    }
}

static void super_fast_avx2_8bit(const uint8_t* bayer, uint8_t* rgb,
                                  int width, int height, const PatternOffsets& po) {
    using namespace pixel;
    const int w = width;

    DLOOP(y, 0, height)
    {
        const int is_gr_row = ((y & 1) == po.r_row);
        size_t row_y   = static_cast<size_t>(y) * w;
        size_t row_ym1 = static_cast<size_t>((y > 0 ? y - 1 : 0)) * w;
        size_t row_yp1 = static_cast<size_t>((y < height - 1 ? y + 1 : height - 1)) * w;

        int x = 0;
        for (; x + 32 <= w; x += 32) {
            __m256i center = _mm256_loadu_si256((const __m256i*)(bayer + row_y + x));
            __m256i left   = _mm256_loadu_si256((const __m256i*)(bayer + row_y + (x > 0 ? x - 1 : 0)));
            __m256i right  = _mm256_loadu_si256((const __m256i*)(bayer + row_y + (x < w - 32 ? x + 1 : w - 32)));
            __m256i up     = _mm256_loadu_si256((const __m256i*)(bayer + row_ym1 + x));
            __m256i down   = _mm256_loadu_si256((const __m256i*)(bayer + row_yp1 + x));

            __m256i g_avg4 = avg4_epu8(left, right, up, down);
            __m256i g_avg2 = avg2_epu8(left, right);
            __m256i b_avg2 = avg2_epu8(up, down);

            uint8_t r_arr[32], g_arr[32], b_arr[32];

            for (int i = 0; i < 32; i++) {
                int px = x + i;
                int at_r = ((y & 1) == po.r_row) && ((px & 1) == po.r_col);
                int at_b = ((y & 1) == po.b_row) && ((px & 1) == po.b_col);

                if (at_r) {
                    r_arr[i] = vec_extract_epu8(center, i);
                    g_arr[i] = vec_extract_epu8(g_avg4, i);
                    int dx = (px & 1) ? -1 : 1, dy = (y & 1) ? -1 : 1;
                    int nx = px + dx, ny = y + dy;
                    if (ny < 0) ny = 0; if (ny >= height) ny = height - 1;
                    if (nx < 0) nx = 0; if (nx >= w) nx = w - 1;
                    b_arr[i] = bayer[static_cast<size_t>(ny) * w + nx];
                } else if (at_b) {
                    b_arr[i] = vec_extract_epu8(center, i);
                    g_arr[i] = vec_extract_epu8(g_avg4, i);
                    int dx = (px & 1) ? -1 : 1, dy = (y & 1) ? -1 : 1;
                    int nx = px + dx, ny = y + dy;
                    if (ny < 0) ny = 0; if (ny >= height) ny = height - 1;
                    if (nx < 0) nx = 0; if (nx >= w) nx = w - 1;
                    r_arr[i] = bayer[static_cast<size_t>(ny) * w + nx];
                } else {
                    g_arr[i] = vec_extract_epu8(center, i);
                    if (is_gr_row) {
                        r_arr[i] = vec_extract_epu8(g_avg2, i);
                        b_arr[i] = vec_extract_epu8(b_avg2, i);
                    } else {
                        b_arr[i] = vec_extract_epu8(g_avg2, i);
                        r_arr[i] = vec_extract_epu8(b_avg2, i);
                    }
                }
            }

            __m256i r_vec = _mm256_loadu_si256((__m256i*)r_arr);
            __m256i g_vec = _mm256_loadu_si256((__m256i*)g_arr);
            __m256i b_vec = _mm256_loadu_si256((__m256i*)b_arr);
            store_rgb_interleaved_32(rgb, (row_y + x) * 3, r_vec, g_vec, b_vec);
        }

        for (; x < w; x++) {
            int at_r = ((y & 1) == po.r_row) && ((x & 1) == po.r_col);
            int at_b = ((y & 1) == po.b_row) && ((x & 1) == po.b_col);

            uint8_t r_v, g_v, b_v;
            uint8_t cur_v = bayer[row_y + x];
            uint8_t l_v   = bayer[row_y + (x > 0 ? x - 1 : 0)];
            uint8_t rp_v  = bayer[row_y + (x < w - 1 ? x + 1 : w - 1)];
            uint8_t u_v   = bayer[row_ym1 + x];
            uint8_t d_v   = bayer[row_yp1 + x];

            if (at_r) {
                r_v = cur_v;
                g_v = static_cast<uint8_t>(((static_cast<int>(l_v) + rp_v + u_v + d_v + 2) >> 2));
                int dx = (x & 1) ? -1 : 1, dy = (y & 1) ? -1 : 1;
                int ny = y + dy, nx = x + dx;
                if (ny < 0) ny = 0; if (ny >= height) ny = height - 1;
                if (nx < 0) nx = 0; if (nx >= w) nx = w - 1;
                b_v = bayer[static_cast<size_t>(ny) * w + nx];
            } else if (at_b) {
                b_v = cur_v;
                g_v = static_cast<uint8_t>(((static_cast<int>(l_v) + rp_v + u_v + d_v + 2) >> 2));
                int dx = (x & 1) ? -1 : 1, dy = (y & 1) ? -1 : 1;
                int ny = y + dy, nx = x + dx;
                if (ny < 0) ny = 0; if (ny >= height) ny = height - 1;
                if (nx < 0) nx = 0; if (nx >= w) nx = w - 1;
                r_v = bayer[static_cast<size_t>(ny) * w + nx];
            } else {
                g_v = cur_v;
                if (is_gr_row) {
                    r_v = static_cast<uint8_t>(((static_cast<int>(l_v) + rp_v + 1) >> 1));
                    b_v = static_cast<uint8_t>(((static_cast<int>(u_v) + d_v + 1) >> 1));
                } else {
                    b_v = static_cast<uint8_t>(((static_cast<int>(l_v) + rp_v + 1) >> 1));
                    r_v = static_cast<uint8_t>(((static_cast<int>(u_v) + d_v + 1) >> 1));
                }
            }
            size_t rgb_idx = (row_y + x) * 3;
            rgb[rgb_idx + 0] = r_v;
            rgb[rgb_idx + 1] = g_v;
            rgb[rgb_idx + 2] = b_v;
        }
    }
}

void process_super_fast_optimized(const uint8_t* bayer, uint8_t* rgb,
                                  int width, int height, const PatternOffsets& po, int bit_depth, bool is_packed) {
    using namespace pixel;

    if (bit_depth <= 8 && !is_packed) {
        super_fast_avx2_8bit(bayer, rgb, width, height, po);
        return;
    }

    DLOOP(y, 0, height)
    {
        for (int x = 0; x < width; x++) {
            int r = 0, g = 0, b = 0;

            if (is_r_at(po, y, x)) {
                r = get_clamped(bayer, x, y, width, height, bit_depth, is_packed);
                g = (get_clamped(bayer, x-1, y, width, height, bit_depth, is_packed) +
                     get_clamped(bayer, x+1, y, width, height, bit_depth, is_packed) +
                     get_clamped(bayer, x, y-1, width, height, bit_depth, is_packed) +
                     get_clamped(bayer, x, y+1, width, height, bit_depth, is_packed) + 2) / 4;
                b = get_clamped(bayer, (x&1)?x-1:x+1, (y&1)?y-1:y+1, width, height, bit_depth, is_packed);
            } else if (is_b_at(po, y, x)) {
                b = get_clamped(bayer, x, y, width, height, bit_depth, is_packed);
                g = (get_clamped(bayer, x-1, y, width, height, bit_depth, is_packed) +
                     get_clamped(bayer, x+1, y, width, height, bit_depth, is_packed) +
                     get_clamped(bayer, x, y-1, width, height, bit_depth, is_packed) +
                     get_clamped(bayer, x, y+1, width, height, bit_depth, is_packed) + 2) / 4;
                r = get_clamped(bayer, (x&1)?x-1:x+1, (y&1)?y-1:y+1, width, height, bit_depth, is_packed);
            } else {
                g = get_clamped(bayer, x, y, width, height, bit_depth, is_packed);
                bool is_gr_row = ((y & 1) == po.r_row);
                if (is_gr_row) {
                    r = (get_clamped(bayer, x-1, y, width, height, bit_depth, is_packed) +
                         get_clamped(bayer, x+1, y, width, height, bit_depth, is_packed) + 1) / 2;
                    b = (get_clamped(bayer, x, y-1, width, height, bit_depth, is_packed) +
                         get_clamped(bayer, x, y+1, width, height, bit_depth, is_packed) + 1) / 2;
                } else {
                    b = (get_clamped(bayer, x-1, y, width, height, bit_depth, is_packed) +
                         get_clamped(bayer, x+1, y, width, height, bit_depth, is_packed) + 1) / 2;
                    r = (get_clamped(bayer, x, y-1, width, height, bit_depth, is_packed) +
                         get_clamped(bayer, x, y+1, width, height, bit_depth, is_packed) + 1) / 2;
                }
            }
            set_rgb_clamped(rgb, x, y, width, r, g, b, bit_depth);
        }
    }
}

static void hqli_avx2_8bit(const uint8_t* bayer, uint8_t* rgb,
                            int width, int height, const PatternOffsets& po) {
    using namespace pixel;
    const int w = width;

    DLOOP(y, 2, height - 2)
    {
        const int is_gr_row = ((y & 1) == po.r_row);
        size_t row_y   = static_cast<size_t>(y) * w;
        size_t row_ym1 = static_cast<size_t>(y - 1) * w;
        size_t row_yp1 = static_cast<size_t>(y + 1) * w;

        int x = 2;
        for (; x + 32 <= w - 2; x += 32) {
            __m256i center = _mm256_loadu_si256((const __m256i*)(bayer + row_y + x));
            __m256i left   = _mm256_loadu_si256((const __m256i*)(bayer + row_y + x - 1));
            __m256i right  = _mm256_loadu_si256((const __m256i*)(bayer + row_y + x + 1));
            __m256i up     = _mm256_loadu_si256((const __m256i*)(bayer + row_ym1 + x));
            __m256i down   = _mm256_loadu_si256((const __m256i*)(bayer + row_yp1 + x));
            __m256i up_l   = _mm256_loadu_si256((const __m256i*)(bayer + row_ym1 + x - 1));
            __m256i up_r   = _mm256_loadu_si256((const __m256i*)(bayer + row_ym1 + x + 1));
            __m256i dn_l   = _mm256_loadu_si256((const __m256i*)(bayer + row_yp1 + x - 1));
            __m256i dn_r   = _mm256_loadu_si256((const __m256i*)(bayer + row_yp1 + x + 1));

            __m256i g_avg4 = avg4_epu8(left, right, up, down);
            __m256i g_avg2 = avg2_epu8(left, right);
            __m256i b_avg2 = avg2_epu8(up, down);
            __m256i diag_avg = avg4_epu8(up_l, up_r, dn_l, dn_r);

            uint8_t r_val[32], g_val[32], b_val[32];
            for (int i = 0; i < 32; i++) {
                int px = x + i;
                int at_r = ((y & 1) == po.r_row) && ((px & 1) == po.r_col);
                int at_b = ((y & 1) == po.b_row) && ((px & 1) == po.b_col);

                if (at_r) {
                    r_val[i] = bayer[row_y + px];
                    g_val[i] = static_cast<uint8_t>(((int)(bayer[row_y+(px-1)]) + (int)(bayer[row_y+(px+1)]) +
                                                       (int)(bayer[row_ym1+px]) + (int)(bayer[row_yp1+px]) + 2) >> 2);
                    b_val[i] = static_cast<uint8_t>(((int)(bayer[row_ym1+(px-1)]) + (int)(bayer[row_ym1+(px+1)]) +
                                                       (int)(bayer[row_yp1+(px-1)]) + (int)(bayer[row_yp1+(px+1)]) + 2) >> 2);
                } else if (at_b) {
                    r_val[i] = static_cast<uint8_t>(((int)(bayer[row_ym1+(px-1)]) + (int)(bayer[row_ym1+(px+1)]) +
                                                       (int)(bayer[row_yp1+(px-1)]) + (int)(bayer[row_yp1+(px+1)]) + 2) >> 2);
                    g_val[i] = static_cast<uint8_t>(((int)(bayer[row_y+(px-1)]) + (int)(bayer[row_y+(px+1)]) +
                                                       (int)(bayer[row_ym1+px]) + (int)(bayer[row_yp1+px]) + 2) >> 2);
                    b_val[i] = bayer[row_y + px];
                } else {
                    g_val[i] = bayer[row_y + px];
                    if (is_gr_row) {
                        r_val[i] = static_cast<uint8_t>(((int)(bayer[row_y+(px-1)]) + (int)(bayer[row_y+(px+1)]) + 1) >> 1);
                        b_val[i] = static_cast<uint8_t>(((int)(bayer[row_ym1+px]) + (int)(bayer[row_yp1+px]) + 1) >> 1);
                    } else {
                        b_val[i] = static_cast<uint8_t>(((int)(bayer[row_y+(px-1)]) + (int)(bayer[row_y+(px+1)]) + 1) >> 1);
                        r_val[i] = static_cast<uint8_t>(((int)(bayer[row_ym1+px]) + (int)(bayer[row_yp1+px]) + 1) >> 1);
                    }
                }
            }

            __m256i r_vec = _mm256_loadu_si256((__m256i*)r_val);
            __m256i g_vec = _mm256_loadu_si256((__m256i*)g_val);
            __m256i b_vec = _mm256_loadu_si256((__m256i*)b_val);
            store_rgb_interleaved_32(rgb, (row_y + x) * 3, r_vec, g_vec, b_vec);
        }

        for (; x < w - 2; x++) {
            int at_r = ((y & 1) == po.r_row) && ((x & 1) == po.r_col);
            int at_b = ((y & 1) == po.b_row) && ((x & 1) == po.b_col);
            uint8_t r_v, g_v, b_v;

            if (at_r) {
                r_v = bayer[row_y + x];
                g_v = static_cast<uint8_t>(((static_cast<int>(bayer[row_y + (x-1)]) + bayer[row_y + (x+1)] +
                                              bayer[row_ym1 + x] + bayer[row_yp1 + x] + 2) >> 2));
                b_v = static_cast<uint8_t>(((static_cast<int>(bayer[row_ym1 + (x-1)]) + bayer[row_ym1 + (x+1)] +
                                              bayer[row_yp1 + (x-1)] + bayer[row_yp1 + (x+1)] + 2) >> 2));
            } else if (at_b) {
                r_v = static_cast<uint8_t>(((static_cast<int>(bayer[row_ym1 + (x-1)]) + bayer[row_ym1 + (x+1)] +
                                              bayer[row_yp1 + (x-1)] + bayer[row_yp1 + (x+1)] + 2) >> 2));
                g_v = static_cast<uint8_t>(((static_cast<int>(bayer[row_y + (x-1)]) + bayer[row_y + (x+1)] +
                                              bayer[row_ym1 + x] + bayer[row_yp1 + x] + 2) >> 2));
                b_v = bayer[row_y + x];
            } else {
                g_v = bayer[row_y + x];
                if (is_gr_row) {
                    r_v = static_cast<uint8_t>(((static_cast<int>(bayer[row_y + (x-1)]) + bayer[row_y + (x+1)] + 1) >> 1));
                    b_v = static_cast<uint8_t>(((static_cast<int>(bayer[row_ym1 + x]) + bayer[row_yp1 + x] + 1) >> 1));
                } else {
                    b_v = static_cast<uint8_t>(((static_cast<int>(bayer[row_y + (x-1)]) + bayer[row_y + (x+1)] + 1) >> 1));
                    r_v = static_cast<uint8_t>(((static_cast<int>(bayer[row_ym1 + x]) + bayer[row_yp1 + x] + 1) >> 1));
                }
            }
            set_rgb_8(rgb, x, y, width, r_v, g_v, b_v);
        }
    }
}

void process_hqli_optimized(const uint8_t* bayer, uint8_t* rgb,
                             int width, int height, const PatternOffsets& po, int bit_depth, bool is_packed) {
    if (bit_depth <= 8 && !is_packed) {
        hqli_avx2_8bit(bayer, rgb, width, height, po);
        return;
    }
}

static void l7_avx2_8bit(const uint8_t* bayer, uint8_t* rgb,
                          int width, int height, const PatternOffsets& po) {
    using namespace pixel;
    const int w = width;

    static const int kL7Weights7[49] = {
          1,  4,  9, 16,  9,  4,  1,
          4,  9, 16, 25, 16,  9,  4,
          9, 16, 25, 36, 25, 16,  9,
         16, 25, 36,  0, 36, 25, 16,
          9, 16, 25, 36, 25, 16,  9,
          4,  9, 16, 25, 16,  9,  4,
          1,  4,  9, 16,  9,  4,  1,
    };

    DLOOP(y, 3, height - 3)
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
                    int is_r = ((ny & 1) == po.r_row) && ((nx & 1) == po.r_col);
                    int is_b = ((ny & 1) == po.b_row) && ((nx & 1) == po.b_col);
                    int pv = bayer[row_ny + nx];
                    if (is_r) { r_sum += pv * wv; r_wsum += wv; }
                    else if (is_b) { b_sum += pv * wv; b_wsum += wv; }
                    else { g_sum += pv * wv; g_wsum += wv; }
                }
            }

            int at_r = ((y & 1) == po.r_row) && ((x & 1) == po.r_col);
            int at_b = ((y & 1) == po.b_row) && ((x & 1) == po.b_col);
            int at_g = !at_r && !at_b;

            int r_val = at_r ? bayer[row_y + x] : (r_wsum > 0 ? (r_sum + r_wsum / 2) / r_wsum : 0);
            int g_val = at_g ? bayer[row_y + x] : (g_wsum > 0 ? (g_sum + g_wsum / 2) / g_wsum : 0);
            int b_val = at_b ? bayer[row_y + x] : (b_wsum > 0 ? (b_sum + b_wsum / 2) / b_wsum : 0);

            set_rgb_8_clamp(rgb, x, y, width, r_val, g_val, b_val);
        }
    }
}

void process_l7_optimized(const uint8_t* bayer, uint8_t* rgb,
                           int width, int height, const PatternOffsets& po, int bit_depth, bool is_packed) {
    if (bit_depth <= 8 && !is_packed) {
        l7_avx2_8bit(bayer, rgb, width, height, po);
        return;
    }
}

static void mg_avx2_8bit(const uint8_t* bayer, uint8_t* rgb,
                          int width, int height, const PatternOffsets& po) {
    using namespace pixel;
    const size_t w = static_cast<size_t>(width);
    const size_t total = static_cast<size_t>(width) * height;
    const int r_row = po.r_row;
    const int r_col = po.r_col;
    const int b_row = po.b_row;
    const int b_col = po.b_col;

    std::vector<int> G(total, 0);
    std::vector<int> R(total, 0);
    std::vector<int> B(total, 0);

    DLOOP(y, 0, height) {
        size_t ry = static_cast<size_t>(y) * w;
        for (int x = 0; x < width; x++) {
            bool at_r = ((y & 1) == r_row) && ((x & 1) == r_col);
            bool at_b = ((y & 1) == b_row) && ((x & 1) == b_col);
            if (!at_r && !at_b) {
                G[ry + x] = static_cast<int>(bayer[ry + x]);
            }
        }
    }

    DLOOP(y, 2, height - 2) {
        size_t ry   = static_cast<size_t>(y) * w;
        size_t rym1 = static_cast<size_t>(y - 1) * w;
        size_t rym2 = static_cast<size_t>(y - 2) * w;
        size_t ryp1 = static_cast<size_t>(y + 1) * w;
        size_t ryp2 = static_cast<size_t>(y + 2) * w;

        int x = 2;
        for (; x + 16 <= width - 2; x += 16) {
            __m128i c_vec  = _mm_loadu_si128((const __m128i*)(bayer + ry + x));
            __m128i n_vec  = _mm_loadu_si128((const __m128i*)(bayer + rym1 + x));
            __m128i s_vec  = _mm_loadu_si128((const __m128i*)(bayer + ryp1 + x));
            __m128i e_vec  = _mm_loadu_si128((const __m128i*)(bayer + ry + x + 1));
            __m128i w_vec  = _mm_loadu_si128((const __m128i*)(bayer + ry + x - 1));
            __m128i n2_vec = _mm_loadu_si128((const __m128i*)(bayer + rym2 + x));
            __m128i s2_vec = _mm_loadu_si128((const __m128i*)(bayer + ryp2 + x));
            __m128i e2_vec = _mm_loadu_si128((const __m128i*)(bayer + ry + x + 2));
            __m128i w2_vec = _mm_loadu_si128((const __m128i*)(bayer + ry + x - 2));

            for (int i = 0; i < 16; i++) {
                int px = x + i;
                bool at_r = ((y & 1) == r_row) && ((px & 1) == r_col);
                bool at_b = ((y & 1) == b_row) && ((px & 1) == b_col);
                if (at_r || at_b) {
                    int c  = static_cast<int>(bayer[ry + px]);
                    int n  = static_cast<int>(bayer[rym1 + px]);
                    int s  = static_cast<int>(bayer[ryp1 + px]);
                    int e  = static_cast<int>(bayer[ry + px + 1]);
                    int wv = static_cast<int>(bayer[ry + px - 1]);
                    int n2 = static_cast<int>(bayer[rym2 + px]);
                    int s2 = static_cast<int>(bayer[ryp2 + px]);
                    int e2 = static_cast<int>(bayer[ry + px + 2]);
                    int w2 = static_cast<int>(bayer[ry + px - 2]);
                    int gv = (4*c + 2*(n+s+e+wv) - (n2+s2+e2+w2) + 4) >> 3;
                    if (gv < 0) gv = 0; else if (gv > 255) gv = 255;
                    G[ry + px] = gv;
                }
            }
        }

        for (; x < width - 2; x++) {
            bool at_r = ((y & 1) == r_row) && ((x & 1) == r_col);
            bool at_b = ((y & 1) == b_row) && ((x & 1) == b_col);
            if (at_r || at_b) {
                int n  = static_cast<int>(bayer[rym1 + x]);
                int s  = static_cast<int>(bayer[ryp1 + x]);
                int e  = static_cast<int>(bayer[ry + (x + 1)]);
                int wv = static_cast<int>(bayer[ry + (x - 1)]);
                int n2 = static_cast<int>(bayer[rym2 + x]);
                int s2 = static_cast<int>(bayer[ryp2 + x]);
                int e2 = static_cast<int>(bayer[ry + (x + 2)]);
                int w2 = static_cast<int>(bayer[ry + (x - 2)]);
                int c  = static_cast<int>(bayer[ry + x]);
                int gv = (4*c + 2*(n+s+e+wv) - (n2+s2+e2+w2) + 4) >> 3;
                if (gv < 0) gv = 0; else if (gv > 255) gv = 255;
                G[ry + x] = gv;
            }
        }
    }

    DLOOP(y, 1, height - 1) {
        size_t ry   = static_cast<size_t>(y) * w;
        size_t rym1 = static_cast<size_t>(y - 1) * w;
        size_t ryp1 = static_cast<size_t>(y + 1) * w;

        for (int x = 1; x < width - 1; x++) {
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

    const int mg_border = 2;
    detail::fill_intermediate_borders_int(R.data(), G.data(), B.data(), width, height, mg_border);

    DLOOP(y, 0, height) {
        size_t ry = static_cast<size_t>(y) * w;
        for (int x = 0; x < width; x++) {
            size_t idx = ry + x;
            int ro = R[idx]; if (ro<0) ro=0; else if (ro>255) ro=255;
            int go = G[idx]; if (go<0) go=0; else if (go>255) go=255;
            int bo = B[idx]; if (bo<0) bo=0; else if (bo>255) bo=255;
            set_rgb_8(rgb, x, y, width, ro, go, bo);
        }
    }
}

void process_mg_optimized(const uint8_t* bayer, uint8_t* rgb,
                          int width, int height, const PatternOffsets& po, int bit_depth, bool is_packed) {
    using namespace pixel;

    if (bit_depth <= 8 && !is_packed) {
        mg_avx2_8bit(bayer, rgb, width, height, po);
        return;
    }

    int max_val = safe_max_val(bit_depth);
    size_t w = static_cast<size_t>(width);
    size_t total = static_cast<size_t>(width) * height;

    std::vector<float> G(total, -1.0f);
    std::vector<float> R(total, 0.0f);
    std::vector<float> B(total, 0.0f);

    DLOOP(y, 0, height) {
        for (int x = 0; x < width; x++) {
            if (!is_r_at(po, y, x) && !is_b_at(po, y, x)) {
                G[static_cast<size_t>(y) * w + x] = static_cast<float>(get_raw(bayer, x, y, width, bit_depth, is_packed, 0, height));
            }
        }
    }

    DLOOP(y, 2, height - 2) {
        for (int x = 2; x < width - 2; x++) {
            if (is_r_at(po, y, x) || is_b_at(po, y, x)) {
                float n=static_cast<float>(get_raw(bayer,x,y-1,width,bit_depth,is_packed,0,height));
                float s=static_cast<float>(get_raw(bayer,x,y+1,width,bit_depth,is_packed,0,height));
                float e=static_cast<float>(get_raw(bayer,x+1,y,width,bit_depth,is_packed,0,height));
                float wv=static_cast<float>(get_raw(bayer,x-1,y,width,bit_depth,is_packed,0,height));
                float n2=static_cast<float>(get_raw(bayer,x,y-2,width,bit_depth,is_packed,0,height));
                float s2=static_cast<float>(get_raw(bayer,x,y+2,width,bit_depth,is_packed,0,height));
                float e2=static_cast<float>(get_raw(bayer,x+2,y,width,bit_depth,is_packed,0,height));
                float w2=static_cast<float>(get_raw(bayer,x-2,y,width,bit_depth,is_packed,0,height));
                float c=static_cast<float>(get_raw(bayer,x,y,width,bit_depth,is_packed,0,height));
                G[static_cast<size_t>(y)*w+x] = (4.0f*c + 2.0f*(n+s+e+wv) - (n2+s2+e2+w2)) / 8.0f;
            }
        }
    }

    DLOOP(y, 1, height - 1) {
        for (int x = 1; x < width - 1; x++) {
            size_t idx = static_cast<size_t>(y) * w + x;
            float g = G[idx];
            if (is_r_at(po, y, x)) {
                R[idx] = static_cast<float>(get_raw(bayer,x,y,width,bit_depth,is_packed,0,height));
                float cd_sum=0.0f; int cd_cnt=0;
                int diag[4][2]={{-1,-1},{1,-1},{-1,1},{1,1}};
                for (int k=0;k<4;k++){int nx=x+diag[k][0],ny=y+diag[k][1];
                    if(is_b_at(po,ny,nx)){
                        cd_sum+=static_cast<float>(get_raw(bayer,nx,ny,width,bit_depth,is_packed,0,height))-G[static_cast<size_t>(ny)*w+nx];cd_cnt++;}}
                B[idx]=g+(cd_cnt>0?cd_sum/static_cast<float>(cd_cnt):0.0f);
            } else if (is_b_at(po, y, x)) {
                B[idx]=static_cast<float>(get_raw(bayer,x,y,width,bit_depth,is_packed,0,height));
                float cd_sum=0.0f; int cd_cnt=0;
                int diag[4][2]={{-1,-1},{1,-1},{-1,1},{1,1}};
                for (int k=0;k<4;k++){int nx=x+diag[k][0],ny=y+diag[k][1];
                    if(is_r_at(po,ny,nx)){
                        cd_sum+=static_cast<float>(get_raw(bayer,nx,ny,width,bit_depth,is_packed,0,height))-G[static_cast<size_t>(ny)*w+nx];cd_cnt++;}}
                R[idx]=g+(cd_cnt>0?cd_sum/static_cast<float>(cd_cnt):0.0f);
            } else {
                float r_cd=0,b_cd=0; int rc=0,bc=0;
                if(is_r_at(po,y-1,x)){r_cd+=static_cast<float>(get_raw(bayer,x,y-1,width,bit_depth,is_packed,0,height))-G[static_cast<size_t>(y-1)*w+x];rc++;}
                if(is_r_at(po,y+1,x)){r_cd+=static_cast<float>(get_raw(bayer,x,y+1,width,bit_depth,is_packed,0,height))-G[static_cast<size_t>(y+1)*w+x];rc++;}
                if(is_r_at(po,y,x-1)){r_cd+=static_cast<float>(get_raw(bayer,x-1,y,width,bit_depth,is_packed,0,height))-G[static_cast<size_t>(y)*w+(x-1)];rc++;}
                if(is_r_at(po,y,x+1)){r_cd+=static_cast<float>(get_raw(bayer,x+1,y,width,bit_depth,is_packed,0,height))-G[static_cast<size_t>(y)*w+(x+1)];rc++;}
                if(is_b_at(po,y-1,x)){b_cd+=static_cast<float>(get_raw(bayer,x,y-1,width,bit_depth,is_packed,0,height))-G[static_cast<size_t>(y-1)*w+x];bc++;}
                if(is_b_at(po,y+1,x)){b_cd+=static_cast<float>(get_raw(bayer,x,y+1,width,bit_depth,is_packed,0,height))-G[static_cast<size_t>(y+1)*w+x];bc++;}
                if(is_b_at(po,y,x-1)){b_cd+=static_cast<float>(get_raw(bayer,x-1,y,width,bit_depth,is_packed,0,height))-G[static_cast<size_t>(y)*w+(x-1)];bc++;}
                if(is_b_at(po,y,x+1)){b_cd+=static_cast<float>(get_raw(bayer,x+1,y,width,bit_depth,is_packed,0,height))-G[static_cast<size_t>(y)*w+(x+1)];bc++;}
                R[idx]=g+(rc>0?r_cd/static_cast<float>(rc):0.0f);
                B[idx]=g+(bc>0?b_cd/static_cast<float>(bc):0.0f);
            }
        }
    }

    const int mg_border = 2;
    detail::fill_intermediate_borders(R.data(), G.data(), B.data(), width, height, mg_border);

    DLOOP(y, 0, height) {
        for (int x = 0; x < width; x++) {
            size_t idx = static_cast<size_t>(y) * w + x;
            int ro = std::max(0, std::min(static_cast<int>(R[idx] + 0.5f), max_val));
            int go = std::max(0, std::min(static_cast<int>(G[idx] + 0.5f), max_val));
            int bo = std::max(0, std::min(static_cast<int>(B[idx] + 0.5f), max_val));
            set_rgb_raw(rgb, x, y, width, ro, go, bo, bit_depth);
        }
    }
}

#else

bool has_avx2() { return false; }

void process_super_fast_optimized(const uint8_t*, uint8_t*, int, int, const PatternOffsets&, int, bool) {}
void process_hqli_optimized(const uint8_t*, uint8_t*, int, int, const PatternOffsets&, int, bool) {}
void process_l7_optimized(const uint8_t*, uint8_t*, int, int, const PatternOffsets&, int, bool) {}
void process_mg_optimized(const uint8_t*, uint8_t*, int, int, const PatternOffsets&, int, bool) {}

#endif // __AVX2__

} // namespace imop