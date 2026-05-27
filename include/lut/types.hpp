#ifndef LUT_TYPES_HPP
#define LUT_TYPES_HPP

#include <cstdint>
#include <vector>
#include <string>

namespace lut {

enum class LUTAlgorithm {
    CUBE_FILE,       // Import .cube format 3D LUT file
    CUSTOM_3D,       // User-supplied 3D LUT data (float*)
    SEPIA,           // Built-in sepia tone
    COOL,            // Built-in cool/blue cast
    WARM,            // Built-in warm/amber cast
    HIGH_CONTRAST,   // Built-in S-curve contrast boost
    LOW_CONTRAST,    // Built-in flat/bleached look
    INVERT,          // Built-in color inversion
    VINTAGE_FADE     // Built-in vintage faded look
};

enum class LUTError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    LUTSizeMismatch,
    FileNotFound,
    FileParseError,
    ImageTooSmall,
    InternalError
};

inline const char* lut_error_message(LUTError err) {
    switch (err) {
        case LUTError::Ok:               return "Success";
        case LUTError::NullInput:        return "Null input/output pointer";
        case LUTError::InvalidDimensions: return "Invalid image dimensions";
        case LUTError::InvalidBitDepth:   return "Invalid bit depth (must be 1-16)";
        case LUTError::InvalidChannels:   return "Invalid channel count (must be 3)";
        case LUTError::LUTSizeMismatch:   return "LUT size does not match expected dimensions";
        case LUTError::FileNotFound:      return "LUT file not found";
        case LUTError::FileParseError:    return "Error parsing LUT file";
        case LUTError::ImageTooSmall:     return "Image too small for algorithm";
        case LUTError::InternalError:    return "Internal processing error";
        default:                          return "Unknown error";
    }
}

inline bool operator!(LUTError err) {
    return err != LUTError::Ok;
}

inline bool ok(LUTError err) {
    return err == LUTError::Ok;
}

inline bool is_valid_bit_depth(int bit_depth) {
    return bit_depth >= 1 && bit_depth <= 16;
}

inline bool is_valid_dimensions(int width, int height) {
    return width > 0 && height > 0;
}

// 3D LUT structure
// Samples are stored in R-major order: lut[ri][gi][bi] = {r, g, b}
// size: number of samples per dimension (e.g. 33 for a 33³ LUT)
// data: flattened 3D array, size = size³ * 3 floats
struct LUT3D {
    int size = 0;
    std::vector<float> data; // size³ * 3 elements, R-major

    bool empty() const { return data.empty(); }
    size_t total_samples() const { return static_cast<size_t>(size) * size * size; }
};

} // namespace lut

#endif // LUT_TYPES_HPP
