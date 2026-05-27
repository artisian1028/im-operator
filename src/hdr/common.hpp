#ifndef HDR_COMMON_HPP
#define HDR_COMMON_HPP

#include <cstdint>
#include <algorithm>
#include <cmath>

namespace hdr {
namespace detail {

// --- Clamp and range ---

inline int clamp_val(int v, int max_val) {
    if (v < 0) return 0;
    if (v > max_val) return max_val;
    return v;
}

inline float clamp_val_f(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

inline int safe_max_val(int bit_depth) {
    if (bit_depth <= 0) return 1;     // float: max_val=1 (no scaling used)
    if (bit_depth >= 32) return static_cast<int>((1u << 31) - 1);
    if (bit_depth >= 16) return 65535;
    return static_cast<int>((1u << bit_depth) - 1);
}

inline float safe_max_val_f(int bit_depth) {
    return static_cast<float>(safe_max_val(bit_depth));
}

// --- Pixel I/O ---

// Read a single pixel channel, normalized to [0,1].
// bit_depth=0: float32 raw values pass through (can be >1.0 for HDR).
// bit_depth>0: integer values normalized by dividing by max_val.
inline float read_pixel_f(const uint8_t* data, int x, int y, int width,
                           int /*channels*/, int bit_depth, int channel) {
    if (bit_depth == 0) {
        const float* f32 = reinterpret_cast<const float*>(data);
        return f32[(static_cast<size_t>(y) * width + x) * 3 + channel];
    }
    float maxv = safe_max_val_f(bit_depth);
    if (bit_depth <= 8) {
        return static_cast<float>(data[(static_cast<size_t>(y) * width + x) * 3 + channel]) / maxv;
    }
    const uint16_t* d16 = reinterpret_cast<const uint16_t*>(data);
    return static_cast<float>(d16[(static_cast<size_t>(y) * width + x) * 3 + channel]) / maxv;
}

// Integer read (for algorithms that still need int paths)
inline int read_pixel(const uint8_t* data, int x, int y, int width,
                       int /*channels*/, int bit_depth, int channel) {
    if (bit_depth == 0) {
        // Read float, convert to int by rounding scaled value
        const float* f32 = reinterpret_cast<const float*>(data);
        float v = f32[(static_cast<size_t>(y) * width + x) * 3 + channel];
        return static_cast<int>(v * 255.0f + 0.5f); // fallback: scale to 8-bit
    }
    if (bit_depth <= 8) {
        return static_cast<int>(data[(static_cast<size_t>(y) * width + x) * 3 + channel]);
    }
    const uint16_t* d16 = reinterpret_cast<const uint16_t*>(data);
    return static_cast<int>(d16[(static_cast<size_t>(y) * width + x) * 3 + channel]);
}

// Write a single pixel channel. bit_depth=0 means float32 per channel.
inline void write_pixel_f(uint8_t* data, int x, int y, int width,
                           int /*channels*/, int bit_depth, int channel, float value) {
    if (bit_depth == 0) {
        float* f32 = reinterpret_cast<float*>(data);
        f32[(static_cast<size_t>(y) * width + x) * 3 + channel] = value;
        return;
    }
    int max_val = safe_max_val(bit_depth);
    int iv = static_cast<int>(value * static_cast<float>(max_val) + 0.5f);
    if (iv < 0) iv = 0;
    if (iv > max_val) iv = max_val;
    if (bit_depth <= 8) {
        data[(static_cast<size_t>(y) * width + x) * 3 + channel] = static_cast<uint8_t>(iv);
    } else {
        uint16_t* d16 = reinterpret_cast<uint16_t*>(data);
        d16[(static_cast<size_t>(y) * width + x) * 3 + channel] = static_cast<uint16_t>(iv);
    }
}

// Integer write (for algorithms that still need int paths)
inline void write_pixel(uint8_t* data, int x, int y, int width,
                         int /*channels*/, int bit_depth, int channel, int value) {
    if (bit_depth == 0) {
        float* f32 = reinterpret_cast<float*>(data);
        f32[(static_cast<size_t>(y) * width + x) * 3 + channel] = static_cast<float>(value) / 255.0f;
        return;
    }
    if (bit_depth <= 8) {
        data[(static_cast<size_t>(y) * width + x) * 3 + channel] = static_cast<uint8_t>(value);
    } else {
        uint16_t* d16 = reinterpret_cast<uint16_t*>(data);
        d16[(static_cast<size_t>(y) * width + x) * 3 + channel] = static_cast<uint16_t>(value);
    }
}

// --- Shared math utilities ---

// BT.709 luminance from linear RGB
inline float rgb_to_luma(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Apply a power function safely (clamp to avoid NaN from negative base)
inline float safe_pow(float base, float exponent) {
    if (base <= 0.0f) return 0.0f;
    return std::pow(base, exponent);
}

} // namespace detail
} // namespace hdr

#endif // HDR_COMMON_HPP
