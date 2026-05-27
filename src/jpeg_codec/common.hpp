#ifndef JPEG_CODEC_SRC_COMMON_HPP
#define JPEG_CODEC_SRC_COMMON_HPP

#include "jpeg_codec/algorithms.hpp"

#include <cstdint>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>
#include <array>

namespace jpeg_codec {
namespace detail {

// ============================================================
//  Pixel I/O helpers
// ============================================================

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

// ============================================================
//  JPEG Standard Quantization Tables (Annex K)
// ============================================================

// Standard luminance quantization table (quality 50)
constexpr int kStdLumaQTable50[64] = {
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99
};

// Standard chrominance quantization table (quality 50)
constexpr int kStdChromaQTable50[64] = {
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99
};

// ============================================================
//  Zigzag ordering (8x8 block)
// ============================================================

constexpr int kZigzag[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// ============================================================
//  DCT / IDCT (separable 1D DCT-II / DCT-III)
// ============================================================

namespace {

// 1D DCT Type-II (forward).  Input x[8], output X[8].
// JPEG scaling: X[u] = 0.5 * C(u) * sum_{i} x[i] * cos((2i+1)*u*pi/16)
// where C(0) = 1/sqrt(2), C(u>0) = 1
inline void dct_1d_fwd(const float x[8], float X[8]) {
    constexpr float scale = 0.5f;
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    for (int u = 0; u < 8; u++) {
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) {
            sum += x[i] * std::cos((2.0f * i + 1.0f) * u * 3.14159265358979323846f / 16.0f);
        }
        X[u] = scale * (u == 0 ? inv_sqrt2 : 1.0f) * sum;
    }
}

// 1D DCT Type-III (inverse).  Input X[8], output x[8].
// JPEG scaling: x[i] = 0.5 * sum_{u} C(u) * X[u] * cos((2i+1)*u*pi/16)
inline void dct_1d_inv(const float X[8], float x[8]) {
    constexpr float scale = 0.5f;
    constexpr float inv_sqrt2 = 0.70710678118654752440f;
    for (int i = 0; i < 8; i++) {
        float sum = 0.0f;
        for (int u = 0; u < 8; u++) {
            float alpha = (u == 0 ? inv_sqrt2 : 1.0f);
            sum += alpha * X[u] * std::cos((2.0f * i + 1.0f) * u * 3.14159265358979323846f / 16.0f);
        }
        x[i] = scale * sum;
    }
}

} // anonymous namespace

// Forward 2D DCT on 8x8 block
// Input:  block[64] in row-major order (block[y*8+x] = spatial sample)
// Output: block[64] in row-major order (block[v*8+u] = DCT coefficient at frequency (u,v))
inline void fdct_8x8(float block[64]) {
    float tmp[64];

    // 1D DCT on rows: block[y*8+x] -> tmp[y*8+u]
    for (int y = 0; y < 8; y++) {
        dct_1d_fwd(block + y * 8, tmp + y * 8);
    }

    // 1D DCT on columns: tmp[y*8+u] -> block[v*8+u]
    for (int u = 0; u < 8; u++) {
        float col[8], out[8];
        for (int y = 0; y < 8; y++) col[y] = tmp[y * 8 + u];
        dct_1d_fwd(col, out);
        for (int v = 0; v < 8; v++) block[v * 8 + u] = out[v];
    }
}

// Inverse 2D DCT on 8x8 block
// Input:  block[64] in row-major order (block[v*8+u] = DCT coefficient at frequency (u,v))
// Output: block[64] in row-major order (block[y*8+x] = spatial sample)
inline void idct_8x8(float block[64]) {
    float tmp[64];

    // 1D IDCT on columns: block[v*8+u] -> tmp[y*8+u]
    for (int u = 0; u < 8; u++) {
        float col[8], out[8];
        for (int v = 0; v < 8; v++) col[v] = block[v * 8 + u];
        dct_1d_inv(col, out);
        for (int y = 0; y < 8; y++) tmp[y * 8 + u] = out[y];
    }

    // 1D IDCT on rows: tmp[y*8+u] -> block[y*8+x]
    for (int y = 0; y < 8; y++) {
        dct_1d_inv(tmp + y * 8, block + y * 8);
    }
}

// ============================================================
//  Quantization table scaling by quality factor
// ============================================================

inline void scale_quant_table(int qtable[64], const int base[64], int quality) {
    // JPEG standard quality scaling
    int q = std::max(1, std::min(100, quality));
    int scale;
    if (q < 50) {
        scale = 5000 / q;
    } else {
        scale = 200 - q * 2;
    }
    for (int i = 0; i < 64; i++) {
        int v = (base[i] * scale + 50) / 100;
        qtable[i] = std::max(1, std::min(255, v));
    }
}

// ============================================================
//  RGB <-> YCbCr Color Conversion
// ============================================================

inline void rgb_to_ycbcr(int r, int g, int b, int& y, int& cb, int& cr) {
    // JPEG JFIF conversion (full range 0-255)
    y  = static_cast<int>( 0.299f * r + 0.587f * g + 0.114f * b);
    cb = static_cast<int>(-0.168736f * r - 0.331264f * g + 0.5f * b + 128.0f);
    cr = static_cast<int>( 0.5f * r - 0.418688f * g - 0.081312f * b + 128.0f);
}

inline void ycbcr_to_rgb(int y, int cb, int cr, int& r, int& g, int& b) {
    float yf  = static_cast<float>(y);
    float cbf = static_cast<float>(cb) - 128.0f;
    float crf = static_cast<float>(cr) - 128.0f;

    r = static_cast<int>(yf + 1.402f * crf + 0.5f);
    g = static_cast<int>(yf - 0.344136f * cbf - 0.714136f * crf + 0.5f);
    b = static_cast<int>(yf + 1.772f * cbf + 0.5f);
}

// ============================================================
//  Bitstream Writer (for JPEG output)
// ============================================================

class BitWriter {
public:
    BitWriter(uint8_t* buffer, size_t capacity)
        : buf_(buffer), cap_(capacity), pos_(0), bit_buf_(0), bit_count_(0) {}

    void write_byte(uint8_t b) {
        if (pos_ < cap_) buf_[pos_++] = b;
    }

    void write_bytes(const uint8_t* data, size_t len) {
        for (size_t i = 0; i < len && pos_ < cap_; i++) {
            buf_[pos_++] = data[i];
        }
    }

    void write_uint16_be(uint16_t v) {
        write_byte(static_cast<uint8_t>(v >> 8));
        write_byte(static_cast<uint8_t>(v & 0xFF));
    }

    void write_bits(int bits, int num_bits) {
        for (int i = num_bits - 1; i >= 0; i--) {
            bit_buf_ = (bit_buf_ << 1) | ((bits >> i) & 1);
            bit_count_++;
            if (bit_count_ == 8) {
                write_byte(static_cast<uint8_t>(bit_buf_));
                if (bit_buf_ == 0xFF) write_byte(0x00); // byte stuff
                bit_buf_ = 0;
                bit_count_ = 0;
            }
        }
    }

    void flush_bits() {
        if (bit_count_ > 0) {
            bit_buf_ <<= (8 - bit_count_);
            write_byte(static_cast<uint8_t>(bit_buf_));
            if (bit_buf_ == 0xFF) write_byte(0x00); // byte stuff
            bit_buf_ = 0;
            bit_count_ = 0;
        }
    }

    size_t position() const { return pos_; }

private:
    uint8_t* buf_;
    size_t cap_;
    size_t pos_;
    int bit_buf_;
    int bit_count_;
};

// ============================================================
//  Bitstream Reader (for JPEG input)
// ============================================================

class BitReader {
public:
    BitReader(const uint8_t* data, size_t size)
        : data_(data), size_(size), pos_(0), bit_buf_(0), bit_count_(0) {}

    // Read a raw byte (for header parsing, no stuffing)
    uint8_t read_byte() {
        if (pos_ >= size_) return 0;
        return data_[pos_++];
    }

    uint16_t read_uint16_be() {
        uint16_t hi = read_byte();
        uint16_t lo = read_byte();
        return static_cast<uint16_t>((hi << 8) | lo);
    }

    void read_bytes(uint8_t* dst, size_t len) {
        for (size_t i = 0; i < len && pos_ < size_; i++) {
            dst[i] = read_byte();
        }
    }

    // Read bits from scan data, handling JPEG byte stuffing (0xFF 0x00 -> 0xFF)
    // and end-of-scan marker detection (0xFF followed by any non-0x00 byte).
    int read_bits(int num_bits) {
        int result = 0;
        for (int i = 0; i < num_bits; i++) {
            if (bit_count_ == 0) {
                if (pos_ >= size_) return 0;

                // If we are at a marker (0xFF followed by a non-stuffed byte),
                // we have reached the end of the scan data.  Return 0 so that
                // the Huffman decoder will fail to match and report an error
                // instead of silently consuming marker bytes as image data.
                if (data_[pos_] == 0xFF && pos_ + 1 < size_ && data_[pos_ + 1] != 0x00) {
                    return 0;
                }

                bit_buf_ = data_[pos_++];
                bit_count_ = 8;
                // JPEG byte stuffing: 0xFF followed by 0x00 represents a single 0xFF data byte
                if (bit_buf_ == 0xFF && pos_ < size_ && data_[pos_] == 0x00) {
                    pos_++; // consume stuffed zero
                }
            }
            result = (result << 1) | ((bit_buf_ >> 7) & 1);
            bit_buf_ <<= 1;
            bit_count_--;
        }
        return result;
    }

    bool eof() const { return pos_ >= size_; }
    size_t position() const { return pos_; }

    void skip(size_t n) {
        pos_ = std::min(pos_ + n, size_);
        bit_count_ = 0;
    }

    void skip_to_marker() {
        bit_count_ = 0;
        while (pos_ < size_) {
            if (data_[pos_] == 0xFF && pos_ + 1 < size_ && data_[pos_ + 1] != 0x00) {
                pos_ += 2;
                return;
            }
            pos_++;
        }
    }

    uint8_t peek_byte() const {
        if (pos_ >= size_) return 0;
        return data_[pos_];
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;
    int bit_buf_;
    int bit_count_;
};

// ============================================================
//  JPEG Marker Constants
// ============================================================

constexpr uint8_t kMarkerSOI  = 0xD8; // Start of Image
constexpr uint8_t kMarkerEOI  = 0xD9; // End of Image
constexpr uint8_t kMarkerSOS  = 0xDA; // Start of Scan
constexpr uint8_t kMarkerSOF0 = 0xC0; // Start of Frame (Baseline DCT)
constexpr uint8_t kMarkerDQT  = 0xDB; // Define Quantization Table
constexpr uint8_t kMarkerDHT  = 0xC4; // Define Huffman Table
constexpr uint8_t kMarkerAPP0 = 0xE0; // Application data (JFIF)
constexpr uint8_t kMarkerCOM  = 0xFE; // Comment
constexpr uint8_t kMarkerDNL  = 0xDC; // Define Number of Lines

// ============================================================
//  Standard JPEG Huffman Tables (Annex K)
// ============================================================

// Standard DC luminance Huffman table
struct HuffmanTable {
    // bits[i] = number of codes of length i+1
    uint8_t bits[16];
    // huffval[i] = values assigned to each code, sorted by code length
    uint8_t huffval[256];
    // Lookup table: huffsize[value] = code length in bits
    // huffcode[value] = Huffman code bits
    int huffsize[256];
    int huffcode[256];
};

// Generate lookup tables from bits/huffval specification
inline void build_huffman_lookup(HuffmanTable& table) {
    // Zero-initialise lookup arrays so that symbol values not present
    // in the table have huffsize=0 and can never match a valid code
    // length (1–16).  Without this, leftover stack values could cause
    // non-deterministic phantom matches on corrupted or misaligned input.
    std::memset(table.huffsize, 0, sizeof(table.huffsize));
    std::memset(table.huffcode, 0, sizeof(table.huffcode));

    int k = 0;
    int code = 0;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < table.bits[i]; j++) {
            uint8_t val = table.huffval[k];
            table.huffsize[val] = i + 1;
            table.huffcode[val] = code;
            k++;
            code++;
        }
        code <<= 1;
    }
}

// Standard DC Luminance Huffman table
constexpr uint8_t kStdDCLumaBits[16] = {
    0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};
constexpr uint8_t kStdDCLumaValues[12] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B
};

// Standard DC Chrominance Huffman table
constexpr uint8_t kStdDCChromaBits[16] = {
    0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00
};
constexpr uint8_t kStdDCChromaValues[12] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B
};

// Standard AC Luminance Huffman table
constexpr uint8_t kStdACLumaBits[16] = {
    0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03,
    0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D
};
constexpr uint8_t kStdACLumaValues[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61,
    0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52,
    0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25,
    0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64,
    0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x83,
    0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
    0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
    0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3,
    0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8,
    0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA
};

// Standard AC Chrominance Huffman table
constexpr uint8_t kStdACChromaBits[16] = {
    0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04,
    0x07, 0x05, 0x04, 0x04, 0x00, 0x01, 0x02, 0x77
};
constexpr uint8_t kStdACChromaValues[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61,
    0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09, 0x23, 0x33,
    0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25, 0xF1, 0x17, 0x18,
    0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44,
    0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63,
    0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A,
    0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4,
    0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA,
    0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7,
    0xE8, 0xE9, 0xEA, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA
};

// Initialize the four standard Huffman tables
inline void init_std_huffman_tables(HuffmanTable& dc_luma, HuffmanTable& dc_chroma,
                                      HuffmanTable& ac_luma, HuffmanTable& ac_chroma) {
    std::memcpy(dc_luma.bits, kStdDCLumaBits, 16);
    std::memcpy(dc_luma.huffval, kStdDCLumaValues, 12);
    build_huffman_lookup(dc_luma);

    std::memcpy(dc_chroma.bits, kStdDCChromaBits, 16);
    std::memcpy(dc_chroma.huffval, kStdDCChromaValues, 12);
    build_huffman_lookup(dc_chroma);

    std::memcpy(ac_luma.bits, kStdACLumaBits, 16);
    std::memcpy(ac_luma.huffval, kStdACLumaValues, 162);
    build_huffman_lookup(ac_luma);

    std::memcpy(ac_chroma.bits, kStdACChromaBits, 16);
    std::memcpy(ac_chroma.huffval, kStdACChromaValues, 162);
    build_huffman_lookup(ac_chroma);
}

// Write Huffman table specification to bitstream
inline void write_huffman_table(BitWriter& bw, uint8_t table_class, uint8_t table_id,
                                  const uint8_t* bits, const uint8_t* huffval) {
    int count = 0;
    for (int i = 0; i < 16; i++) count += bits[i];

    bw.write_byte(0xFF);
    bw.write_byte(kMarkerDHT);
    // Length = 2 (TcTh) + 16 (bits) + count (values) + 2 (length field)
    bw.write_uint16_be(static_cast<uint16_t>(3 + 16 + count));
    bw.write_byte(static_cast<uint8_t>((table_class << 4) | table_id));
    bw.write_bytes(bits, 16);
    bw.write_bytes(huffval, count);
}

// ============================================================
//  MCU size for chroma subsampling
// ============================================================

inline int get_mcu_width(int subsample) {
    return (subsample == 0) ? 8 : 16;  // 4:4:4 -> 8, 4:2:0/4:2:2 -> 16
}

inline int get_mcu_height(int subsample) {
    return (subsample == 1) ? 16 : 8;  // 4:2:0 -> 16, others -> 8
}

inline int get_h_sampling(int subsample, int component) {
    // component: 0=Y, 1=Cb, 2=Cr
    if (component == 0) return (subsample == 0) ? 1 : 2;
    return (subsample == 0) ? 1 : ((subsample == 2) ? 1 : 1);
}

inline int get_v_sampling(int subsample, int component) {
    if (component == 0) return (subsample == 1) ? 2 : 1;
    return (subsample == 1) ? 1 : 1;
}

} // namespace detail
} // namespace jpeg_codec

#endif // JPEG_CODEC_SRC_COMMON_HPP
