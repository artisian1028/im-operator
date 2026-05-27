#ifndef IMOP_TYPES_HPP
#define IMOP_TYPES_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <utility>

namespace imop {

enum class BayerPattern {
    RGGB,
    BGGR,
    GRBG,
    GBRG
};

enum class DemosaicAlgorithm {
    SUPER_FAST,
    HQLI,
    MG,
    L7,
    DFPD,
    AHD,
    AMAZE,
    RCD,
    PRISM
};

enum class DemosaicError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidPattern,
    ImageTooSmall,
    InternalError
};

inline const char* demosaic_error_message(DemosaicError err) {
    switch (err) {
        case DemosaicError::Ok: return "Success";
        case DemosaicError::NullInput: return "Null input/output pointer";
        case DemosaicError::InvalidDimensions: return "Invalid image dimensions";
        case DemosaicError::InvalidBitDepth: return "Invalid bit depth (must be 1-16)";
        case DemosaicError::InvalidPattern: return "Invalid Bayer pattern";
        case DemosaicError::ImageTooSmall: return "Image too small for algorithm";
        case DemosaicError::InternalError: return "Internal processing error";
        default: return "Unknown error";
    }
}

inline bool operator!(DemosaicError err) {
    return err != DemosaicError::Ok;
}

inline bool ok(DemosaicError err) {
    return err == DemosaicError::Ok;
}

struct ImageBuffer {
    std::vector<uint8_t> data;
    int width = 0;
    int height = 0;
    int channels = 1;
    int bit_depth = 8;
    bool is_packed = false;

    size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }
    uint8_t* ptr() { return data.data(); }
    const uint8_t* ptr() const { return data.data(); }
};

struct DataInfo {
    int detected_bit_depth = 0;
    int suggested_width = 0;
    int suggested_height = 0;
    int pixel_count = 0;
    int max_value = 0;
    int min_value = 0;
    bool is_likely_16bit = false;
    bool is_packed = false;
    std::vector<std::pair<int, int>> possible_dimensions;
};

struct PatternOffsets {
    int r_row, r_col;
    int b_row, b_col;

    static PatternOffsets from_pattern(BayerPattern pattern) {
        switch (pattern) {
            case BayerPattern::RGGB: return {0, 0, 1, 1};
            case BayerPattern::BGGR: return {1, 1, 0, 0};
            case BayerPattern::GRBG: return {0, 1, 1, 0};
            case BayerPattern::GBRG: return {1, 0, 0, 1};
            default: return {0, 0, 1, 1};
        }
    }
};

inline bool is_valid_bayer_pattern(BayerPattern p) {
    switch (p) {
        case BayerPattern::RGGB:
        case BayerPattern::BGGR:
        case BayerPattern::GRBG:
        case BayerPattern::GBRG:
            return true;
        default:
            return false;
    }
}

inline bool is_valid_bit_depth(int bit_depth) {
    return bit_depth >= 1 && bit_depth <= 16;
}

inline bool is_valid_dimensions(int width, int height) {
    return width > 0 && height > 0;
}

} // namespace imop

#endif // IMOP_TYPES_HPP
