#ifndef WHITE_BALANCE_COMMON_HPP
#define WHITE_BALANCE_COMMON_HPP

#include "white_balance/algorithms.hpp"
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <vector>

namespace white_balance {
namespace detail {

// Clamp value to [0, max_val]
inline int clamp_val(int v, int max_val) {
    return std::max(0, std::min(v, max_val));
}

// Compute safe max value for a given bit depth
inline int safe_max_val(int bit_depth) {
    if (bit_depth <= 0) return 255;
    if (bit_depth >= 16) return 65535;
    return (1 << bit_depth) - 1;
}

// Read a pixel channel value from 8-bit or 16-bit data
inline int read_pixel(const uint8_t* data, int x, int y, int width,
                      int channels, int bit_depth, int channel) {
    if (bit_depth <= 8) {
        return data[(static_cast<size_t>(y) * width + x) * channels + channel];
    } else {
        const uint16_t* data16 = reinterpret_cast<const uint16_t*>(data);
        return data16[(static_cast<size_t>(y) * width + x) * channels + channel];
    }
}

// Write a pixel channel value to 8-bit or 16-bit data
inline void write_pixel(uint8_t* data, int x, int y, int width,
                        int channels, int bit_depth, int channel, int value) {
    if (bit_depth <= 8) {
        data[(static_cast<size_t>(y) * width + x) * channels + channel] =
            static_cast<uint8_t>(value);
    } else {
        uint16_t* data16 = reinterpret_cast<uint16_t*>(data);
        data16[(static_cast<size_t>(y) * width + x) * channels + channel] =
            static_cast<uint16_t>(value);
    }
}

// Bytes per pixel per channel
inline int bpp(int bit_depth) {
    return bit_depth <= 8 ? 1 : 2;
}

} // namespace detail
} // namespace white_balance

#endif // WHITE_BALANCE_COMMON_HPP
