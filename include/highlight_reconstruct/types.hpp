#ifndef HIGHLIGHT_RECONSTRUCT_TYPES_HPP
#define HIGHLIGHT_RECONSTRUCT_TYPES_HPP

#include <cstdint>

namespace highlight_reconstruct {

enum class HighlightReconstructAlgorithm {
    CHANNEL_GUIDED,   // Use unclipped channels' ratios to estimate clipped ones
    GRADIENT_BASED    // Propagate gradient from clip boundaries inward
};

enum class HighlightReconstructError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ImageTooSmall,
    InternalError
};

inline const char* highlight_reconstruct_error_message(HighlightReconstructError err) {
    switch (err) {
        case HighlightReconstructError::Ok:               return "Success";
        case HighlightReconstructError::NullInput:        return "Null input/output pointer";
        case HighlightReconstructError::InvalidDimensions: return "Invalid image dimensions";
        case HighlightReconstructError::InvalidBitDepth:   return "Invalid bit depth (must be 1-16)";
        case HighlightReconstructError::InvalidChannels:   return "Invalid channel count (must be 3)";
        case HighlightReconstructError::ImageTooSmall:     return "Image too small for algorithm";
        case HighlightReconstructError::InternalError:    return "Internal processing error";
        default:                                           return "Unknown error";
    }
}

inline bool operator!(HighlightReconstructError err) { return err != HighlightReconstructError::Ok; }
inline bool ok(HighlightReconstructError err) { return err == HighlightReconstructError::Ok; }
inline bool is_valid_bit_depth(int bd) { return bd >= 1 && bd <= 16; }
inline bool is_valid_dimensions(int w, int h) { return w > 0 && h > 0; }

struct HighlightReconstructParams {
    float threshold = 0.95f;  // clip detection threshold relative to max_val
};

} // namespace highlight_reconstruct

#endif
