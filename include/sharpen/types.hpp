#ifndef SHARPEN_TYPES_HPP
#define SHARPEN_TYPES_HPP

#include <cstdint>

namespace sharpen {

enum class SharpenAlgorithm {
    UNSHARP_MASK,    // Classic unsharp mask: sharpen = original + amount * (original - blurred)
    LAPLACIAN,       // 3x3/5x5 Laplacian edge enhancement
    HIGH_PASS,       // High-pass filter overlay for fine detail enhancement
    ADAPTIVE         // Content-adaptive sharpening (edge-aware)
};

enum class SharpenError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ImageTooSmall,
    InternalError
};

inline const char* sharpen_error_message(SharpenError err) {
    switch (err) {
        case SharpenError::Ok:               return "Success";
        case SharpenError::NullInput:        return "Null input/output pointer";
        case SharpenError::InvalidDimensions: return "Invalid image dimensions";
        case SharpenError::InvalidBitDepth:   return "Invalid bit depth (must be 1-16)";
        case SharpenError::InvalidChannels:   return "Invalid channel count (must be 3)";
        case SharpenError::ImageTooSmall:     return "Image too small for algorithm";
        case SharpenError::InternalError:    return "Internal processing error";
        default:                              return "Unknown error";
    }
}

inline bool operator!(SharpenError err) {
    return err != SharpenError::Ok;
}

inline bool ok(SharpenError err) {
    return err == SharpenError::Ok;
}

inline bool is_valid_bit_depth(int bit_depth) {
    return bit_depth >= 1 && bit_depth <= 16;
}

inline bool is_valid_dimensions(int width, int height) {
    return width > 0 && height > 0;
}

// Sharpen parameters
struct SharpenParams {
    float amount = 1.0f;   // sharpening strength [0, 3]
    float radius = 1.0f;   // blur radius for unsharp mask / laplacian sigma
    float threshold = 0.0f; // edge threshold for adaptive mode
};

} // namespace sharpen

#endif // SHARPEN_TYPES_HPP
