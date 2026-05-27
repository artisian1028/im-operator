#ifndef BLACK_LEVEL_TYPES_HPP
#define BLACK_LEVEL_TYPES_HPP

#include <cstdint>

namespace black_level {

enum class BlackLevelAlgorithm {
    PER_CHANNEL,   // R, Gr, Gb, B each have their own offset
    GLOBAL         // Single offset for all pixels
};

enum class BlackLevelError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ImageTooSmall,
    InternalError
};

inline const char* black_level_error_message(BlackLevelError err) {
    switch (err) {
        case BlackLevelError::Ok:               return "Success";
        case BlackLevelError::NullInput:        return "Null input/output pointer";
        case BlackLevelError::InvalidDimensions: return "Invalid image dimensions";
        case BlackLevelError::InvalidBitDepth:   return "Invalid bit depth (must be 1-16)";
        case BlackLevelError::InvalidChannels:   return "Invalid channel count (must be 1 for Bayer)";
        case BlackLevelError::ImageTooSmall:     return "Image too small for algorithm";
        case BlackLevelError::InternalError:    return "Internal processing error";
        default:                                 return "Unknown error";
    }
}

inline bool operator!(BlackLevelError err) { return err != BlackLevelError::Ok; }
inline bool ok(BlackLevelError err) { return err == BlackLevelError::Ok; }
inline bool is_valid_bit_depth(int bd) { return bd >= 1 && bd <= 16; }
inline bool is_valid_dimensions(int w, int h) { return w > 0 && h > 0; }

struct BlackLevelParams {
    float r_offset = 0.0f;
    float gr_offset = 0.0f;
    float gb_offset = 0.0f;
    float b_offset = 0.0f;
};

} // namespace black_level

#endif
