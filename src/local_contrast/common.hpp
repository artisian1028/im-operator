#ifndef LOCAL_CONTRAST_COMMON_HPP
#define LOCAL_CONTRAST_COMMON_HPP

#include <cstdint>
#include <algorithm>
#include <cmath>

namespace local_contrast {
namespace detail {

inline int safe_max_val(int bit_depth) {
    if (bit_depth <= 0) return 255;
    if (bit_depth > 16) return 65535;
    return static_cast<int>((1u << bit_depth) - 1);
}

inline int read_pixel(const uint8_t* data, int x, int y, int width,
                       int /*channels*/, int bit_depth, int channel) {
    if (bit_depth <= 8)
        return data[(static_cast<size_t>(y) * width + x) * 3 + channel];
    const uint16_t* d16 = reinterpret_cast<const uint16_t*>(data);
    return d16[(static_cast<size_t>(y) * width + x) * 3 + channel];
}

inline void write_pixel(uint8_t* data, int x, int y, int width,
                         int /*channels*/, int bit_depth, int channel, int value) {
    if (bit_depth <= 8)
        data[(static_cast<size_t>(y) * width + x) * 3 + channel] = static_cast<uint8_t>(value);
    else
        reinterpret_cast<uint16_t*>(data)[(static_cast<size_t>(y) * width + x) * 3 + channel] = static_cast<uint16_t>(value);
}

inline int clamp_val(int v, int max_val) { return std::clamp(v, 0, max_val); }

} // namespace detail
} // namespace local_contrast

#endif
