#ifndef SATURATION_TYPES_HPP
#define SATURATION_TYPES_HPP

#include <cstdint>

namespace saturation {

enum class SaturationAlgorithm {
    HSL,            // Convert to HSL, scale S, convert back
    VIBRANCE,       // Intelligent saturation: protects skin tones, boosts muted colors
    CHANNEL_MIXER,  // Cross-channel saturation blending
    SELECTIVE       // Per-channel saturation (R, G, B independently)
};

enum class SaturationError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ImageTooSmall,
    InternalError
};

inline const char* saturation_error_message(SaturationError err) {
    switch (err) {
        case SaturationError::Ok:               return "Success";
        case SaturationError::NullInput:        return "Null input/output pointer";
        case SaturationError::InvalidDimensions: return "Invalid image dimensions";
        case SaturationError::InvalidBitDepth:   return "Invalid bit depth (must be 1-16)";
        case SaturationError::InvalidChannels:   return "Invalid channel count (must be 3)";
        case SaturationError::ImageTooSmall:     return "Image too small for algorithm";
        case SaturationError::InternalError:    return "Internal processing error";
        default:                                 return "Unknown error";
    }
}

inline bool operator!(SaturationError err) { return err != SaturationError::Ok; }
inline bool ok(SaturationError err) { return err == SaturationError::Ok; }

inline bool is_valid_bit_depth(int bit_depth) {
    return bit_depth >= 1 && bit_depth <= 16;
}
inline bool is_valid_dimensions(int width, int height) {
    return width > 0 && height > 0;
}

struct SaturationParams {
    float saturation = 1.0f;  // global saturation [0, 3], 1.0 = identity
    float r_sat = 1.0f;       // R channel saturation for SELECTIVE
    float g_sat = 1.0f;       // G channel saturation for SELECTIVE
    float b_sat = 1.0f;       // B channel saturation for SELECTIVE
    float vibrance = 1.0f;    // vibrance strength [0, 3] for VIBRANCE
};

} // namespace saturation

#endif // SATURATION_TYPES_HPP
