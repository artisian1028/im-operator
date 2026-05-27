#ifndef SATURATION_SRC_COMMON_HPP
#define SATURATION_SRC_COMMON_HPP

#include "saturation/algorithms.hpp"
#include <cstdint>
#include <algorithm>
#include <cmath>

namespace saturation {
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

// RGB -> HSL lightness (min+max)/2, saturation, hue
inline void rgb_to_hsl(float r, float g, float b,
                        float& h, float& s, float& l) {
    float mx = std::max({r, g, b});
    float mn = std::min({r, g, b});
    l = (mx + mn) * 0.5f;

    if (mx == mn) {
        h = 0.0f;
        s = 0.0f;
    } else {
        float d = mx - mn;
        s = (l > 0.5f) ? d / (2.0f - mx - mn) : d / (mx + mn);

        if (mx == r) {
            h = (g - b) / d + (g < b ? 6.0f : 0.0f);
        } else if (mx == g) {
            h = (b - r) / d + 2.0f;
        } else {
            h = (r - g) / d + 4.0f;
        }
        h /= 6.0f;
    }
}

// HSL -> RGB helper
inline float hue_to_rgb(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1.0f / 2.0f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}

inline void hsl_to_rgb(float h, float s, float l,
                        float& r, float& g, float& b) {
    if (s == 0.0f) {
        r = g = b = l;
    } else {
        float q = (l < 0.5f) ? l * (1.0f + s) : l + s - l * s;
        float p = 2.0f * l - q;
        r = hue_to_rgb(p, q, h + 1.0f / 3.0f);
        g = hue_to_rgb(p, q, h);
        b = hue_to_rgb(p, q, h - 1.0f / 3.0f);
    }
}

} // namespace detail
} // namespace saturation

#endif // SATURATION_SRC_COMMON_HPP
