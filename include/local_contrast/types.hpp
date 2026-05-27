#ifndef LOCAL_CONTRAST_TYPES_HPP
#define LOCAL_CONTRAST_TYPES_HPP

#include <cstdint>

namespace local_contrast {

enum class LocalContrastAlgorithm {
    UNSHARP,     // Large-radius unsharp mask (Lightroom Clarity)
    BILATERAL    // Bilateral decomposition, enhance detail layer
};

enum class LocalContrastError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ImageTooSmall,
    InternalError
};

inline const char* local_contrast_error_message(LocalContrastError err) {
    switch (err) {
        case LocalContrastError::Ok:               return "Success";
        case LocalContrastError::NullInput:        return "Null input/output pointer";
        case LocalContrastError::InvalidDimensions: return "Invalid image dimensions";
        case LocalContrastError::InvalidBitDepth:   return "Invalid bit depth (must be 1-16)";
        case LocalContrastError::InvalidChannels:   return "Invalid channel count (must be 3)";
        case LocalContrastError::ImageTooSmall:     return "Image too small for algorithm";
        case LocalContrastError::InternalError:    return "Internal processing error";
        default:                                    return "Unknown error";
    }
}

inline bool operator!(LocalContrastError err) { return err != LocalContrastError::Ok; }
inline bool ok(LocalContrastError err) { return err == LocalContrastError::Ok; }
inline bool is_valid_bit_depth(int bd) { return bd >= 1 && bd <= 16; }
inline bool is_valid_dimensions(int w, int h) { return w > 0 && h > 0; }

struct LocalContrastParams {
    float amount = 0.0f;    // strength [0, 2], 0 = off, 1 = default Clarity
    float radius = 20.0f;   // blur sigma in pixels [3, 50]
    float threshold = 0.0f; // detail edge threshold [0, 0.5]
};

} // namespace local_contrast

#endif
