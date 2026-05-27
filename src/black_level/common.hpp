#ifndef BLACK_LEVEL_COMMON_HPP
#define BLACK_LEVEL_COMMON_HPP

#include <cstdint>
#include <algorithm>
#include "imop/types.hpp"

namespace black_level {
namespace detail {

inline int safe_max_val(int bit_depth) {
    if (bit_depth <= 0) return 255;
    if (bit_depth > 16) return 65535;
    return static_cast<int>((1u << bit_depth) - 1);
}

inline int read_bayer(const uint8_t* data, int x, int y, int width, int bit_depth) {
    if (bit_depth <= 8) return data[static_cast<size_t>(y) * width + x];
    const uint16_t* d16 = reinterpret_cast<const uint16_t*>(data);
    return d16[static_cast<size_t>(y) * width + x];
}

inline void write_bayer(uint8_t* data, int x, int y, int width, int bit_depth, int value) {
    if (bit_depth <= 8) data[static_cast<size_t>(y) * width + x] = static_cast<uint8_t>(value);
    else reinterpret_cast<uint16_t*>(data)[static_cast<size_t>(y) * width + x] = static_cast<uint16_t>(value);
}

// Bayer color index: 0=R, 1=Gr, 2=Gb, 3=B
inline int bayer_color(int row, int col, const imop::PatternOffsets& po) {
    if ((row & 1) == po.r_row && (col & 1) == po.r_col) return 0; // R
    if ((row & 1) == po.b_row && (col & 1) == po.b_col) return 3; // B
    // Gr = same row as R, same col as B; Gb = same row as B, same col as R
    if ((row & 1) == po.r_row) return 1; // Gr
    return 2; // Gb
}

} // namespace detail
} // namespace black_level

#endif
