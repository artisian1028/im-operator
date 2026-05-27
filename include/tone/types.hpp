#ifndef TONE_TYPES_HPP
#define TONE_TYPES_HPP

#include <cstdint>

namespace tone {

enum class ToneAlgorithm {
    GAMMA,              // Power-law gamma correction
    S_CURVE,            // S-curve contrast with shadow/highlight control
    LEVELS,             // Levels adjustment (black/mid/white clipping)
    CURVES_3POINT,      // Three-point cubic Bezier tone curve
    SHADOWS_HIGHLIGHTS  // Independent shadow lift + highlight recovery
};

enum class ToneError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ImageTooSmall,
    InternalError
};

inline const char* tone_error_message(ToneError err) {
    switch (err) {
        case ToneError::Ok:               return "Success";
        case ToneError::NullInput:        return "Null input/output pointer";
        case ToneError::InvalidDimensions: return "Invalid image dimensions";
        case ToneError::InvalidBitDepth:   return "Invalid bit depth (must be 1-16)";
        case ToneError::InvalidChannels:   return "Invalid channel count (must be 3)";
        case ToneError::ImageTooSmall:     return "Image too small for algorithm";
        case ToneError::InternalError:    return "Internal processing error";
        default:                           return "Unknown error";
    }
}

inline bool operator!(ToneError err) { return err != ToneError::Ok; }
inline bool ok(ToneError err) { return err == ToneError::Ok; }

inline bool is_valid_bit_depth(int bit_depth) {
    return bit_depth >= 1 && bit_depth <= 16;
}
inline bool is_valid_dimensions(int width, int height) {
    return width > 0 && height > 0;
}

// Tone adjustment parameters
struct ToneParams {
    float gamma = 1.0f;       // gamma: [0.1, 10.0]   (1.0 = identity)
    float contrast = 0.0f;    // contrast: [-1, 2]     (0 = identity)
    float shadows = 0.0f;     // shadow lift: [-1, 1]
    float highlights = 0.0f;  // highlight recovery: [-1, 1]
    float black_point = 0.0f; // levels black clip: [0, 1]
    float white_point = 1.0f; // levels white clip: [0, 1]
    float mid_point = 0.5f;   // levels mid-point: [0, 1]
};

} // namespace tone

#endif // TONE_TYPES_HPP
