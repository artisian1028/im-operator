#ifndef IMOP_ALGORITHMS_COMMON_HPP
#define IMOP_ALGORITHMS_COMMON_HPP

#include "imop/pixel_utils.hpp"
#include "imop/algorithms.hpp"
#include <cstdint>
#include <vector>
#include <cstring>

namespace imop {
namespace detail {

class FloatBufferPool {
    std::vector<std::vector<float>> pool_;
public:
    std::vector<float>& get(size_t min_size, size_t idx) {
        if (idx >= pool_.size()) pool_.resize(idx + 1);
        auto& buf = pool_[idx];
        if (buf.size() < min_size) {
            buf.resize(min_size, 0.0f);
        }
        return buf;
    }
    void clear_all() {
        for (auto& b : pool_) std::fill(b.begin(), b.end(), 0.0f);
    }
};

inline FloatBufferPool& thread_local_pool() {
    static thread_local FloatBufferPool pool;
    return pool;
}

} // namespace detail
} // namespace imop

#ifdef _OPENMP
#include <omp.h>
#define DLOOP(y, y0, ymax) _Pragma("omp parallel for schedule(static)") for (int y = y0; y < ymax; y++)
#else
#define DLOOP(y, y0, ymax) for (int y = y0; y < ymax; y++)
#endif

namespace imop {
namespace detail {

inline void fill_rgb_borders(uint8_t* rgb, int width, int height, int bit_depth, int border) {
    if (border <= 0 || width <= border * 2 || height <= border * 2) return;
    const int stride = (bit_depth <= 8) ? width * 3 : width * 6;
    const int ref_top = border, ref_bottom = height - 1 - border;

    for (int y = 0; y < border; y++) {
        std::memcpy(rgb + y * stride, rgb + ref_top * stride, stride);
    }
    for (int y = height - border; y < height; y++) {
        std::memcpy(rgb + y * stride, rgb + ref_bottom * stride, stride);
    }

    if (bit_depth <= 8) {
        for (int y = border; y < height - border; y++) {
            uint8_t* row = rgb + y * stride;
            uint8_t* ref_row_left  = rgb + y * stride + border * 3;
            uint8_t* ref_row_right = rgb + y * stride + (width - 1 - border) * 3;
            for (int x = 0; x < border; x++) {
                size_t d = x * 3;
                row[d + 0] = ref_row_left[0];
                row[d + 1] = ref_row_left[1];
                row[d + 2] = ref_row_left[2];
            }
            for (int x = width - border; x < width; x++) {
                size_t d = x * 3;
                row[d + 0] = ref_row_right[0];
                row[d + 1] = ref_row_right[1];
                row[d + 2] = ref_row_right[2];
            }
        }
    } else {
        for (int y = border; y < height - border; y++) {
            uint16_t* row = reinterpret_cast<uint16_t*>(rgb + y * stride);
            uint16_t* ref_row_left  = row + border * 3;
            uint16_t* ref_row_right = row + (width - 1 - border) * 3;
            for (int x = 0; x < border; x++) {
                size_t d = x * 3;
                row[d + 0] = ref_row_left[0];
                row[d + 1] = ref_row_left[1];
                row[d + 2] = ref_row_left[2];
            }
            for (int x = width - border; x < width; x++) {
                size_t d = x * 3;
                row[d + 0] = ref_row_right[0];
                row[d + 1] = ref_row_right[1];
                row[d + 2] = ref_row_right[2];
            }
        }
    }
}

template<typename T>
inline void fill_intermediate_borders_t(T* R, T* G, T* B,
                                        int width, int height, int border) {
    if (border <= 0 || width <= border * 2 || height <= border * 2) return;

    for (int y = 0; y < border; y++) {
        size_t di = static_cast<size_t>(y) * width;
        size_t ri = static_cast<size_t>(border) * width;
        for (int x = 0; x < width; x++) {
            R[di + x] = R[ri + x]; G[di + x] = G[ri + x]; B[di + x] = B[ri + x];
        }
    }
    for (int y = height - border; y < height; y++) {
        size_t di = static_cast<size_t>(y) * width;
        size_t ri = static_cast<size_t>(height - 1 - border) * width;
        for (int x = 0; x < width; x++) {
            R[di + x] = R[ri + x]; G[di + x] = G[ri + x]; B[di + x] = B[ri + x];
        }
    }

    DLOOP(y, border, height - border) {
        size_t ry = static_cast<size_t>(y) * width;
        size_t rl = ry + border;
        size_t rr = ry + (width - 1 - border);
        for (int x = 0; x < border; x++) {
            R[ry + x] = R[rl]; G[ry + x] = G[rl]; B[ry + x] = B[rl];
        }
        for (int x = width - border; x < width; x++) {
            R[ry + x] = R[rr]; G[ry + x] = G[rr]; B[ry + x] = B[rr];
        }
    }
}

template<typename T>
inline void fill_intermediate_borders(T* R, T* G, T* B,
                                      int width, int height, int border) {
    fill_intermediate_borders_t(R, G, B, width, height, border);
}

inline void fill_intermediate_borders_int(int* R, int* G, int* B,
                                          int width, int height, int border) {
    fill_intermediate_borders_t(R, G, B, width, height, border);
}

template<typename RawFunc>
inline void interpolate_rb_at_g_positions(float* R, float* B, const float* G,
                                          int x, int y, int width,
                                          const PatternOffsets& po,
                                          RawFunc&& raw) {
    using namespace pixel;
    size_t idx = static_cast<size_t>(y) * width + x;
    float g = G[idx];
    float r_cd_sum = 0.0f, b_cd_sum = 0.0f;
    int r_cnt = 0, b_cnt = 0;

    if (is_r_at(po, y-1, x)) { r_cd_sum += raw(x, y-1) - G[static_cast<size_t>(y-1)*width+x]; r_cnt++; }
    if (is_r_at(po, y+1, x)) { r_cd_sum += raw(x, y+1) - G[static_cast<size_t>(y+1)*width+x]; r_cnt++; }
    if (is_r_at(po, y, x-1)) { r_cd_sum += raw(x-1, y) - G[static_cast<size_t>(y)*width+(x-1)]; r_cnt++; }
    if (is_r_at(po, y, x+1)) { r_cd_sum += raw(x+1, y) - G[static_cast<size_t>(y)*width+(x+1)]; r_cnt++; }

    if (is_b_at(po, y-1, x)) { b_cd_sum += raw(x, y-1) - G[static_cast<size_t>(y-1)*width+x]; b_cnt++; }
    if (is_b_at(po, y+1, x)) { b_cd_sum += raw(x, y+1) - G[static_cast<size_t>(y+1)*width+x]; b_cnt++; }
    if (is_b_at(po, y, x-1)) { b_cd_sum += raw(x-1, y) - G[static_cast<size_t>(y)*width+(x-1)]; b_cnt++; }
    if (is_b_at(po, y, x+1)) { b_cd_sum += raw(x+1, y) - G[static_cast<size_t>(y)*width+(x+1)]; b_cnt++; }

    R[idx] = g + (r_cnt > 0 ? r_cd_sum / static_cast<float>(r_cnt) : 0.0f);
    B[idx] = g + (b_cnt > 0 ? b_cd_sum / static_cast<float>(b_cnt) : 0.0f);
}

template<typename RawFunc>
inline void interpolate_rb_at_rb_positions(float* R, float* B, const float* G,
                                           int x, int y, int width,
                                           const PatternOffsets& po,
                                           RawFunc&& raw) {
    using namespace pixel;
    size_t idx = static_cast<size_t>(y) * width + x;
    float g = G[idx];

    if (is_r_at(po, y, x)) {
        R[idx] = raw(x, y);
        float cd_sum = 0.0f; int cd_cnt = 0;
        int diag[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
        for (int k = 0; k < 4; k++) {
            int nx = x + diag[k][0], ny = y + diag[k][1];
            if (nx >= 0 && nx < width && ny >= 0 && is_b_at(po, ny, nx)) {
                cd_sum += raw(nx, ny) - G[static_cast<size_t>(ny)*width+nx]; cd_cnt++;
            }
        }
        B[idx] = g + (cd_cnt > 0 ? cd_sum / static_cast<float>(cd_cnt) : 0.0f);
    } else if (is_b_at(po, y, x)) {
        B[idx] = raw(x, y);
        float cd_sum = 0.0f; int cd_cnt = 0;
        int diag[4][2] = {{-1,-1},{1,-1},{-1,1},{1,1}};
        for (int k = 0; k < 4; k++) {
            int nx = x + diag[k][0], ny = y + diag[k][1];
            if (nx >= 0 && nx < width && ny >= 0 && is_r_at(po, ny, nx)) {
                cd_sum += raw(nx, ny) - G[static_cast<size_t>(ny)*width+nx]; cd_cnt++;
            }
        }
        R[idx] = g + (cd_cnt > 0 ? cd_sum / static_cast<float>(cd_cnt) : 0.0f);
    }
}

inline void write_float_planes_to_rgb(const float* R, const float* G, const float* B,
                                      uint8_t* rgb, int width, int height, int bit_depth) {
    using namespace pixel;
    int max_val = safe_max_val(bit_depth);
    DLOOP(y, 0, height) {
        for (int x = 0; x < width; x++) {
            size_t idx = static_cast<size_t>(y) * width + x;
            int r_out = std::max(0, std::min(static_cast<int>(R[idx] + 0.5f), max_val));
            int g_out = std::max(0, std::min(static_cast<int>(G[idx] + 0.5f), max_val));
            int b_out = std::max(0, std::min(static_cast<int>(B[idx] + 0.5f), max_val));
            set_rgb_raw(rgb, x, y, width, r_out, g_out, b_out, bit_depth);
        }
    }
}

} // namespace detail
} // namespace imop

#endif
