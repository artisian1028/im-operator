#ifndef DENOISE_ALGORITHMS_COMMON_HPP
#define DENOISE_ALGORITHMS_COMMON_HPP

#include "denoise/algorithms.hpp"
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <vector>

namespace denoise {
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

// Read a pixel value from 8-bit or 16-bit data
inline int read_pixel(const uint8_t* data, int x, int y, int width,
                      int channels, int bit_depth, int channel) {
    if (bit_depth <= 8) {
        return data[(static_cast<size_t>(y) * width + x) * channels + channel];
    } else {
        const uint16_t* data16 = reinterpret_cast<const uint16_t*>(data);
        return data16[(static_cast<size_t>(y) * width + x) * channels + channel];
    }
}

// Write a pixel value to 8-bit or 16-bit data
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

// Read a raw Bayer pixel value (single channel)
inline int read_bayer(const uint8_t* bayer, int x, int y, int width, int bit_depth) {
    if (bit_depth <= 8) {
        return bayer[static_cast<size_t>(y) * width + x];
    } else {
        const uint16_t* bayer16 = reinterpret_cast<const uint16_t*>(bayer);
        return bayer16[static_cast<size_t>(y) * width + x];
    }
}

// Write a raw Bayer pixel value (single channel)
inline void write_bayer(uint8_t* bayer, int x, int y, int width, int bit_depth, int value) {
    if (bit_depth <= 8) {
        bayer[static_cast<size_t>(y) * width + x] = static_cast<uint8_t>(value);
    } else {
        uint16_t* bayer16 = reinterpret_cast<uint16_t*>(bayer);
        bayer16[static_cast<size_t>(y) * width + x] = static_cast<uint16_t>(value);
    }
}

// Bytes per pixel per channel
inline int bpp(int bit_depth) {
    return bit_depth <= 8 ? 1 : 2;
}

// Row stride in bytes
inline int row_stride(int width, int channels, int bit_depth) {
    return width * channels * bpp(bit_depth);
}

} // namespace detail
} // namespace denoise

#endif // DENOISE_ALGORITHMS_COMMON_HPP
