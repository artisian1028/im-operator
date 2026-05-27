#ifndef IMOP_PIXEL_UTILS_HPP
#define IMOP_PIXEL_UTILS_HPP

#include "types.hpp"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstddef>

namespace imop {
namespace pixel {

constexpr int kMaxBitDepth = 16;

inline int safe_max_val(int bit_depth) {
    if (bit_depth <= 0) return 0;
    if (bit_depth > kMaxBitDepth) return static_cast<int>((1u << kMaxBitDepth) - 1);
    return static_cast<int>((1u << bit_depth) - 1);
}

inline size_t compute_packed_byte_size(int width, int height, int bit_depth) {
    return (static_cast<size_t>(width) * height * bit_depth + 7) / 8;
}

inline size_t compute_bayer_byte_size(int width, int height, int bit_depth, bool is_packed) {
    if (is_packed) return compute_packed_byte_size(width, height, bit_depth);
    return static_cast<size_t>(width) * height * (bit_depth <= 8 ? 1 : 2);
}

inline size_t compute_rgb_byte_size(int width, int height, int bit_depth) {
    size_t s = static_cast<size_t>(width) * height * 3;
    if (bit_depth > 8) s *= 2;
    return s;
}

inline bool is_r_at(const PatternOffsets& po, int row, int col) {
    return ((row & 1) == po.r_row) && ((col & 1) == po.r_col);
}

inline bool is_b_at(const PatternOffsets& po, int row, int col) {
    return ((row & 1) == po.b_row) && ((col & 1) == po.b_col);
}

inline bool is_g_at(const PatternOffsets& po, int row, int col) {
    return !is_r_at(po, row, col) && !is_b_at(po, row, col);
}

inline uint16_t read_u16(const uint8_t* data, size_t byte_offset) {
    uint16_t val;
    std::memcpy(&val, data + byte_offset, sizeof(val));
    return val;
}

inline void write_u16(uint8_t* data, size_t byte_offset, uint16_t val) {
    std::memcpy(data + byte_offset, &val, sizeof(val));
}

inline int get_packed_raw(const uint8_t* data, int x, int y, int width, int bit_depth,
                          size_t data_byte_size = 0, int height = 0) {
    if (bit_depth < 1 || bit_depth > kMaxBitDepth) return 0;
    if (x < 0 || y < 0 || x >= width || (height > 0 && y >= height)) return 0;
    size_t pixel_index = static_cast<size_t>(y) * width + x;

    if (data_byte_size == 0 && height > 0)
        data_byte_size = compute_packed_byte_size(width, height, bit_depth);

    int max_val = safe_max_val(bit_depth);

    if (bit_depth == 12) {
        size_t byte_offset = pixel_index / 2 * 3;
        if (data_byte_size > 0 && byte_offset + 2 >= data_byte_size) return 0;
        uint8_t b0 = data[byte_offset];
        uint8_t b1 = data[byte_offset + 1];
        uint8_t b2 = data[byte_offset + 2];
        if ((pixel_index & 1) == 0) {
            return static_cast<int>(((b0 << 4) | (b1 & 0xF)) & max_val);
        } else {
            return static_cast<int>(((b2 << 4) | (b1 >> 4)) & max_val);
        }
    }

    if (bit_depth == 10) {
        size_t byte_offset = pixel_index / 4 * 5;
        int sub = static_cast<int>(pixel_index & 3);
        if (data_byte_size > 0 && byte_offset + 4 >= data_byte_size) return 0;
        uint8_t b0 = data[byte_offset];
        uint8_t b1 = data[byte_offset + 1];
        uint8_t b2 = data[byte_offset + 2];
        uint8_t b3 = data[byte_offset + 3];
        uint8_t b4 = data[byte_offset + 4];
        switch (sub) {
            case 0: return static_cast<int>(((b0 << 2) | (b4 & 0x3)) & max_val);
            case 1: return static_cast<int>(((b1 << 2) | ((b4 >> 2) & 0x3)) & max_val);
            case 2: return static_cast<int>(((b2 << 2) | ((b4 >> 4) & 0x3)) & max_val);
            case 3: return static_cast<int>(((b3 << 2) | ((b4 >> 6) & 0x3)) & max_val);
            default: return 0;
        }
    }

    size_t bit_offset = pixel_index * bit_depth;
    size_t byte_offset = bit_offset / 8;
    int bit_shift = static_cast<int>(bit_offset % 8);
    int bits_needed = bit_depth + bit_shift;
    int bytes_to_read = (bits_needed + 7) / 8;
    if (bytes_to_read < 1) bytes_to_read = 1;

    if (data_byte_size > 0) {
        if (byte_offset >= data_byte_size) return 0;
        bytes_to_read = static_cast<int>(std::min(static_cast<size_t>(bytes_to_read),
                                                   data_byte_size - byte_offset));
    }

    uint32_t val = 0;
    for (int i = 0; i < bytes_to_read; i++) {
        val |= static_cast<uint32_t>(data[byte_offset + i]) << (8 * i);
    }
    return static_cast<int>((val >> bit_shift) & static_cast<uint32_t>(max_val));
}

inline int get_packed(const uint8_t* data, int x, int y, int width, int height, int bit_depth,
                      size_t data_byte_size = 0) {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    if (data_byte_size == 0)
        data_byte_size = compute_packed_byte_size(width, height, bit_depth);
    return get_packed_raw(data, x, y, width, bit_depth, data_byte_size);
}

inline uint16_t align_raw_value(uint16_t val, int bit_depth) {
    if (bit_depth > 8 && bit_depth < 16) {
        int max_val = safe_max_val(bit_depth);
        if (val > static_cast<uint16_t>(max_val)) {
            val >>= (16 - bit_depth);
        }
    }
    return val;
}

inline int get_raw(const uint8_t* data, int x, int y, int width, int bit_depth, bool is_packed = false,
                   size_t data_byte_size = 0, int height = 0) {
    if (bit_depth <= 8) {
        return static_cast<int>(data[static_cast<size_t>(y) * width + x]);
    } else if (is_packed) {
        return get_packed_raw(data, x, y, width, bit_depth, data_byte_size, height);
    } else {
        size_t byte_offset = (static_cast<size_t>(y) * width + x) * 2;
        uint16_t val = read_u16(data, byte_offset);
        return static_cast<int>(align_raw_value(val, bit_depth));
    }
}

inline int get_raw_8(const uint8_t* data, int x, int y, int width) {
    return static_cast<int>(data[static_cast<size_t>(y) * width + x]);
}

inline int get_raw_16(const uint8_t* data, int x, int y, int width, int bit_depth) {
    size_t byte_offset = (static_cast<size_t>(y) * width + x) * 2;
    uint16_t val = read_u16(data, byte_offset);
    return static_cast<int>(align_raw_value(val, bit_depth));
}

inline int get_clamped(const uint8_t* data, int x, int y, int width, int height, int bit_depth, bool is_packed = false,
                       size_t data_byte_size = 0) {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    if (bit_depth <= 8) {
        return static_cast<int>(data[static_cast<size_t>(y) * width + x]);
    } else if (is_packed) {
        return get_packed(data, x, y, width, height, bit_depth, data_byte_size);
    } else {
        size_t byte_offset = (static_cast<size_t>(y) * width + x) * 2;
        uint16_t val = read_u16(data, byte_offset);
        return static_cast<int>(align_raw_value(val, bit_depth));
    }
}

inline int get_clamped_8(const uint8_t* data, int x, int y, int width, int height) {
    x = (x < 0) ? 0 : (x >= width ? width - 1 : x);
    y = (y < 0) ? 0 : (y >= height ? height - 1 : y);
    return static_cast<int>(data[static_cast<size_t>(y) * width + x]);
}

inline int get_clamped_16(const uint8_t* data, int x, int y, int width, int height, int bit_depth) {
    x = (x < 0) ? 0 : (x >= width ? width - 1 : x);
    y = (y < 0) ? 0 : (y >= height ? height - 1 : y);
    size_t byte_offset = (static_cast<size_t>(y) * width + x) * 2;
    uint16_t val = read_u16(data, byte_offset);
    return static_cast<int>(align_raw_value(val, bit_depth));
}

inline void set_rgb(uint8_t* rgb, int x, int y, int width,
                    int r, int g, int b, int bit_depth, bool should_clamp) {
    if (should_clamp) {
        int max_val = safe_max_val(bit_depth);
        r = std::clamp(r, 0, max_val);
        g = std::clamp(g, 0, max_val);
        b = std::clamp(b, 0, max_val);
    }
    if (bit_depth <= 8) {
        size_t idx = (static_cast<size_t>(y) * width + x) * 3;
        rgb[idx + 0] = static_cast<uint8_t>(r);
        rgb[idx + 1] = static_cast<uint8_t>(g);
        rgb[idx + 2] = static_cast<uint8_t>(b);
    } else {
        size_t idx = (static_cast<size_t>(y) * width + x) * 3;
        uint16_t* rgb16 = reinterpret_cast<uint16_t*>(rgb);
        rgb16[idx + 0] = static_cast<uint16_t>(r);
        rgb16[idx + 1] = static_cast<uint16_t>(g);
        rgb16[idx + 2] = static_cast<uint16_t>(b);
    }
}

inline void set_rgb_raw(uint8_t* rgb, int x, int y, int width,
                        int r, int g, int b, int bit_depth) {
    if (bit_depth <= 8) {
        size_t idx = (static_cast<size_t>(y) * width + x) * 3;
        rgb[idx + 0] = static_cast<uint8_t>(r);
        rgb[idx + 1] = static_cast<uint8_t>(g);
        rgb[idx + 2] = static_cast<uint8_t>(b);
    } else {
        size_t idx = (static_cast<size_t>(y) * width + x) * 3;
        uint16_t* rgb16 = reinterpret_cast<uint16_t*>(rgb);
        rgb16[idx + 0] = static_cast<uint16_t>(r);
        rgb16[idx + 1] = static_cast<uint16_t>(g);
        rgb16[idx + 2] = static_cast<uint16_t>(b);
    }
}

inline void set_rgb_clamped(uint8_t* rgb, int x, int y, int width,
                            int r, int g, int b, int bit_depth) {
    set_rgb(rgb, x, y, width, r, g, b, bit_depth, true);
}

inline void set_rgb_8(uint8_t* rgb, int x, int y, int width,
                      int r, int g, int b) {
    size_t idx = (static_cast<size_t>(y) * width + x) * 3;
    rgb[idx + 0] = static_cast<uint8_t>(r);
    rgb[idx + 1] = static_cast<uint8_t>(g);
    rgb[idx + 2] = static_cast<uint8_t>(b);
}

inline void set_rgb_8_clamp(uint8_t* rgb, int x, int y, int width,
                            int r, int g, int b) {
    size_t idx = (static_cast<size_t>(y) * width + x) * 3;
    rgb[idx + 0] = static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
    rgb[idx + 1] = static_cast<uint8_t>(g < 0 ? 0 : (g > 255 ? 255 : g));
    rgb[idx + 2] = static_cast<uint8_t>(b < 0 ? 0 : (b > 255 ? 255 : b));
}

inline void set_rgb_16(uint8_t* rgb, int x, int y, int width,
                       int r, int g, int b) {
    size_t idx = (static_cast<size_t>(y) * width + x) * 3;
    uint16_t* rgb16 = reinterpret_cast<uint16_t*>(rgb);
    rgb16[idx + 0] = static_cast<uint16_t>(r);
    rgb16[idx + 1] = static_cast<uint16_t>(g);
    rgb16[idx + 2] = static_cast<uint16_t>(b);
}

} // namespace pixel
} // namespace imop

#endif // IMOP_PIXEL_UTILS_HPP
