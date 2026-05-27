#include "imop/analyzer.hpp"
#include "imop/pixel_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace imop {

// Check if a given number of pixels matches a known camera resolution.
// Returns the pixel count if matched, 0 otherwise.
static size_t match_common_resolution(size_t pixels) {
    static const size_t common_sizes[] = {
        640*480, 720*480, 720*576, 800*600, 1024*768,
        1280*720, 1280*800, 1280*1024, 1440*900, 1920*1080,
        1920*1200, 2048*1536, 2560*1440, 2560*1600, 3840*2160,
        4096*2160, 4096*3072, 5120*2880, 5320*4600, 7680*4320,
        // Additional common sensor resolutions
        1006*758, 4024*3032,
    };
    for (size_t s : common_sizes) { if (pixels == s) return s; }
    return 0;
}

// For 16-bit (non-packed) data, examine uint16 values to guess bit depth.
static int guess_bit_from_16bit_data(const uint8_t* data, size_t byte_size) {
    size_t count16 = byte_size / 2;
    size_t sample_count = std::min(count16, static_cast<size_t>(100000));
    int max_val = 0;
    for (size_t i = 0; i < sample_count; i++) {
        uint16_t val;
        std::memcpy(&val, data + i * 2, sizeof(val));
        if (val > max_val) max_val = val;
    }
    if (max_val <= 255) return 8;
    if (max_val <= 1023) return 10;
    if (max_val <= 4095) return 12;
    if (max_val <= 16383) return 14;
    return 16;
}

// Heuristic: does the data look like packed raw pixels?
// Packed data, when misinterpreted as uint16, produces values with
// many high-byte bits set (since pixel bits span byte boundaries).
// Non-packed 8-bit data has hi_byte_ratio < 0.1 typically.
static bool looks_like_packed(const uint8_t* data, size_t byte_size) {
    if (byte_size < 8) return false;
    size_t sample_count = std::min(byte_size / 2, static_cast<size_t>(10000));
    int high_byte_count = 0, zero_count = 0, ff_count = 0;
    int prev_high = 0;
    for (size_t i = 0; i < sample_count; i++) {
        uint16_t val;
        std::memcpy(&val, data + i * 2, sizeof(val));
        if ((val >> 8) != 0) high_byte_count++;
        if (val == 0) zero_count++;
        if (val == 0xFFFF) ff_count++;
        // Track transitions: packed data alternates between
        // "reasonable" and "garbage" uint16 values when decoded wrong
        if ((val >> 8) != 0) prev_high++; else prev_high = 0;
    }
    double hi_ratio = static_cast<double>(high_byte_count) / sample_count;
    double zero_ratio = static_cast<double>(zero_count) / sample_count;

    // Packed data seen as uint16: very high hi-byte ratio (>90%)
    // and very low zero ratio (no pixel is truly zero in packed form)
    // 8-bit data: hi_ratio near 0
    // 16-bit data with values < 256: hi_ratio near 0
    return (hi_ratio > 0.85 && zero_ratio < 0.01);
}

int detect_bit_depth(const uint8_t* data, size_t byte_size) {
    if (!data || byte_size == 0) return 0;

    // Step 1: check if byte_size matches a known 8-bit resolution
    if (match_common_resolution(byte_size) != 0) return 8;

    // Step 2: check if byte_size/2 matches a known resolution (16-bit storage)
    if (byte_size % 2 == 0 && match_common_resolution(byte_size / 2) != 0) {
        return guess_bit_from_16bit_data(data, byte_size);
    }

    // Step 3: try packed interpretations.
    // For a known resolution with N pixels and bit_depth D:
    //   packed size = ceil(N * D / 8)
    // Try D = 10, 12, 14, 16 and see if any match.
    if (looks_like_packed(data, byte_size)) {
        for (int d : {10, 12, 14, 16}) {
            // byte_size * 8 = N * d (approximately, with up to d-1 bits of rounding)
            // N = floor(byte_size * 8 / d)
            size_t bits_total = byte_size * 8;
            size_t pixel_count = bits_total / d;
            // Reverse check: does this pixel count correspond to a known resolution?
            if (match_common_resolution(pixel_count) != 0) return d;
            // Also try pixel_count+1 (rounding)
            if (match_common_resolution(pixel_count + 1) != 0) return d;
        }
    }

    // Step 4: generic heuristic for unpacked 16-bit data
    if (byte_size % 2 == 0 && byte_size >= 4) {
        size_t count16 = byte_size / 2;
        size_t sample_count = std::min(count16, static_cast<size_t>(10000));
        int high_byte_nonzero = 0;
        int val16_max = 0;
        for (size_t i = 0; i < sample_count; i++) {
            uint16_t val;
            std::memcpy(&val, data + i * 2, sizeof(val));
            if ((val >> 8) > 0) high_byte_nonzero++;
            if (val > val16_max) val16_max = val;
        }
        double ratio = static_cast<double>(high_byte_nonzero) / sample_count;
        if (ratio > 0.3 && val16_max > 255) return 16;
    }

    return 8;
}

std::vector<std::pair<int, int>> suggest_dimensions(size_t pixel_count) {
    std::vector<std::pair<int, int>> results;
    if (pixel_count == 0) return results;

    static const int common_widths[] = {640, 720, 800, 1024, 1280, 1440, 1920, 2048, 2560, 3840, 4096, 5120, 7680};
    static const int common_heights[] = {480, 576, 600, 720, 768, 800, 1024, 1080, 1200, 1440, 1536, 2048, 2160, 3072, 4320};

    for (int w : common_widths) {
        if (pixel_count % w == 0) {
            int h = static_cast<int>(pixel_count / w);
            for (int ch : common_heights) {
                if (h == ch) { results.push_back({w, h}); break; }
            }
        }
    }

    if (results.empty()) {
        int sqrt_n = static_cast<int>(std::sqrt(static_cast<double>(pixel_count)));
        for (int w = sqrt_n - 100; w <= sqrt_n + 100; w++) {
            if (w > 0 && pixel_count % w == 0) {
                int h = static_cast<int>(pixel_count / w);
                if (w >= 64 && h >= 64) {
                    results.push_back({w, h});
                    if (results.size() >= 5) break;
                }
            }
        }
    }

    return results;
}

DataInfo analyze_data(const uint8_t* data, size_t byte_size) {
    DataInfo info;
    if (!data || byte_size == 0) return info;

    info.detected_bit_depth = detect_bit_depth(data, byte_size);
    info.is_likely_16bit = (info.detected_bit_depth > 8);

    // Determine pixel_count based on storage format.
    // Unpacked 8-bit: pixel_count = byte_size
    // Unpacked 16-bit: pixel_count = byte_size / 2
    // Packed: pixel_count = floor(byte_size * 8 / bit_depth) or ceil variant
    if (info.detected_bit_depth > 8) {
        size_t pixel_count_16 = byte_size / 2;
        if (match_common_resolution(pixel_count_16) != 0) {
            // Unpacked (16-bit per pixel storage)
            info.pixel_count = static_cast<int>(pixel_count_16);
            info.min_value = 65535;
            for (size_t i = 0; i < pixel_count_16; i++) {
                uint16_t v;
                std::memcpy(&v, data + i * 2, sizeof(v));
                if (v < info.min_value) info.min_value = v;
                if (v > info.max_value) info.max_value = v;
            }
        } else {
            info.is_packed = true;
            size_t bits_total = byte_size * 8;
            info.pixel_count = static_cast<int>(bits_total / info.detected_bit_depth);
            // Rounding correction
            if (match_common_resolution(info.pixel_count) == 0 &&
                match_common_resolution(info.pixel_count + 1) != 0) {
                info.pixel_count++;
            }
            // Min/max: decode a sample of packed pixels
            info.min_value = pixel::safe_max_val(info.detected_bit_depth);
            int sample = std::min(info.pixel_count, 10000);
            for (int i = 0; i < sample; i++) {
                int bo = (i * info.detected_bit_depth) / 8;
                int bs = (i * info.detected_bit_depth) % 8;
                if (static_cast<size_t>(bo) + 1 < byte_size) {
                    int v = (data[bo] | (data[bo + 1] << 8)) >> bs;
                    v &= static_cast<int>((1u << info.detected_bit_depth) - 1);
                    if (v < info.min_value) info.min_value = v;
                    if (v > info.max_value) info.max_value = v;
                }
            }
        }
    } else {
        info.pixel_count = static_cast<int>(byte_size);
        info.min_value = 255;
        for (size_t i = 0; i < byte_size; i++) {
            int v = data[i];
            if (v < info.min_value) info.min_value = v;
            if (v > info.max_value) info.max_value = v;
        }
    }

    info.possible_dimensions = suggest_dimensions(info.pixel_count);
    if (!info.possible_dimensions.empty()) {
        info.suggested_width = info.possible_dimensions[0].first;
        info.suggested_height = info.possible_dimensions[0].second;
    }

    return info;
}

BayerPattern guess_pattern(const uint8_t* data, int width, int height,
                            int bit_depth, bool is_packed) {
    if (!data || width < 4 || height < 4) return BayerPattern::RGGB;

    size_t data_byte_size = is_packed
        ? pixel::compute_packed_byte_size(width, height, bit_depth)
        : static_cast<size_t>(width) * height * (bit_depth <= 8 ? 1 : 2);

    auto get_val = [&](int x, int y) -> int {
        if (x < 0 || x >= width || y < 0 || y >= height) return 0;
        if (bit_depth <= 8) return static_cast<int>(data[static_cast<size_t>(y) * width + x]);
        if (is_packed) return pixel::get_packed_raw(data, x, y, width, bit_depth, data_byte_size);
        uint16_t val;
        std::memcpy(&val, data + (static_cast<size_t>(y) * width + x) * 2, sizeof(val));
        return static_cast<int>(pixel::align_raw_value(val, bit_depth));
    };

    int sample_count = std::min(20000, static_cast<int>(static_cast<long long>(width) * height));
    int data_min = get_val(0, 0);
    int data_max = data_min;
    for (int i = 0; i < sample_count; i++) {
        int idx = (i * 10007) % static_cast<int>(static_cast<long long>(width) * height);
        int v = get_val(idx % width, idx / width);
        if (v < data_min) data_min = v;
        if (v > data_max) data_max = v;
    }
    int data_range = std::max(1, data_max - data_min);

    int max_blocks = 5000;
    int step = std::max(1, static_cast<int>(std::sqrt(
        static_cast<double>(width) * height / max_blocks)));
    step = (step / 2) * 2;
    if (step < 2) step = 2;

    double confidence_threshold = data_range * 0.02;
    if (confidence_threshold < 1.0) confidence_threshold = 1.0;

    int vote_RGBG_family = 0;
    int vote_GRGB_family = 0;

    for (int y = 0; y < height - 1; y += step) {
        for (int x = 0; x < width - 1; x += step) {
            double tl = get_val(x, y);
            double tr = get_val(x + 1, y);
            double bl = get_val(x, y + 1);
            double br = get_val(x + 1, y + 1);

            double diag1_diff = std::abs(tl - br);
            double diag2_diff = std::abs(tr - bl);

            double diff_between_diagonals = std::abs(diag1_diff - diag2_diff);
            if (diff_between_diagonals < confidence_threshold) continue;

            double local_range = std::max({tl, tr, bl, br}) - std::min({tl, tr, bl, br});
            double edge_limit = data_range * 0.15;
            if (local_range > edge_limit) continue;

            double horiz_diff = std::abs(tl - tr) + std::abs(bl - br);
            double vert_diff  = std::abs(tl - bl) + std::abs(tr - br);
            double edge_ratio = std::max(horiz_diff, vert_diff) / std::max(1.0, std::min(horiz_diff, vert_diff));
            if (edge_ratio > 4.0) continue;

            if (diag2_diff < diag1_diff) {
                vote_RGBG_family++;
            } else {
                vote_GRGB_family++;
            }
        }
    }

    int global_step = std::max(1, std::min(width, height) / 256);
    global_step |= 1;

    double sum_ee = 0, sum_oo = 0, sum_eo = 0, sum_oe = 0;
    int cnt_ee = 0, cnt_oo = 0, cnt_eo = 0, cnt_oe = 0;

    for (int y = 0; y < height; y += global_step) {
        for (int x = 0; x < width; x += global_step) {
            int v = get_val(x, y);
            if ((y & 1) == 0 && (x & 1) == 0)      { sum_ee += v; cnt_ee++; }
            else if ((y & 1) == 1 && (x & 1) == 1)  { sum_oo += v; cnt_oo++; }
            else if ((y & 1) == 0 && (x & 1) == 1)  { sum_eo += v; cnt_eo++; }
            else                                     { sum_oe += v; cnt_oe++; }
        }
    }

    BayerPattern result;
    int vote_family_margin = std::abs(vote_RGBG_family - vote_GRGB_family);
    bool family_is_RGBG = (vote_RGBG_family >= vote_GRGB_family);

    if (vote_family_margin <= 2 && vote_RGBG_family + vote_GRGB_family < 10) {
        result = BayerPattern::RGGB;
    } else if (family_is_RGBG) {
        double mean_ee = cnt_ee > 0 ? sum_ee / cnt_ee : 0;
        double mean_oo = cnt_oo > 0 ? sum_oo / cnt_oo : 0;
        double ratio = (mean_ee > 0) ? mean_oo / mean_ee : 0;
        result = (ratio > 1.15) ? BayerPattern::BGGR : BayerPattern::RGGB;
    } else {
        double mean_eo = cnt_eo > 0 ? sum_eo / cnt_eo : 0;
        double mean_oe = cnt_oe > 0 ? sum_oe / cnt_oe : 0;
        double ratio = (mean_eo > 0) ? mean_oe / mean_eo : 0;
        result = (ratio > 1.15) ? BayerPattern::GBRG : BayerPattern::GRBG;
    }

    return result;
}

} // namespace imop
