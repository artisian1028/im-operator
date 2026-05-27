#ifndef TONE_SRC_COMMON_HPP
#define TONE_SRC_COMMON_HPP

#include "tone/algorithms.hpp"
#include <cstdint>
#include <algorithm>
#include <cmath>

namespace tone {
namespace detail {

inline int clamp_val(int v, int max_val) {
    return std::max(0, std::min(v, max_val));
}

inline int safe_max_val(int bit_depth) {
    if (bit_depth <= 0) return 255;
    if (bit_depth >= 16) return 65535;
    return (1 << bit_depth) - 1;
}

inline int read_pixel(const uint8_t* data, int x, int y, int width,
                      int channels, int bit_depth, int channel) {
    if (bit_depth <= 8) {
        return data[(static_cast<size_t>(y) * width + x) * channels + channel];
    } else {
        const uint16_t* data16 = reinterpret_cast<const uint16_t*>(data);
        return data16[(static_cast<size_t>(y) * width + x) * channels + channel];
    }
}

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

} // namespace detail
} // namespace tone

#endif // TONE_SRC_COMMON_HPP
