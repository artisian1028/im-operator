#ifndef DENOISE_TYPES_HPP
#define DENOISE_TYPES_HPP

#include <cstdint>

namespace denoise {

enum class DenoiseAlgorithm {
    GAUSSIAN,       // Gaussian blur (separable, fast)
    MEDIAN,         // Median filter (salt & pepper noise)
    BILATERAL,      // Bilateral filter (edge-preserving)
    NLM,            // Non-Local Means (patch-based)
    WAVELET,        // Wavelet thresholding (frequency-domain)
    BAYER_DENOISE   // Bayer-domain denoise (raw CFA data)
};

enum class DenoiseError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ImageTooSmall,
    InternalError
};

inline const char* denoise_error_message(DenoiseError err) {
    switch (err) {
        case DenoiseError::Ok:               return "Success";
        case DenoiseError::NullInput:        return "Null input/output pointer";
        case DenoiseError::InvalidDimensions: return "Invalid image dimensions";
        case DenoiseError::InvalidBitDepth:   return "Invalid bit depth (must be 1-16)";
        case DenoiseError::InvalidChannels:   return "Invalid channel count (must be 1 or 3)";
        case DenoiseError::ImageTooSmall:     return "Image too small for algorithm";
        case DenoiseError::InternalError:    return "Internal processing error";
        default:                              return "Unknown error";
    }
}

inline bool operator!(DenoiseError err) {
    return err != DenoiseError::Ok;
}

inline bool ok(DenoiseError err) {
    return err == DenoiseError::Ok;
}

inline bool is_valid_bit_depth(int bit_depth) {
    return bit_depth >= 1 && bit_depth <= 16;
}

inline bool is_valid_dimensions(int width, int height) {
    return width > 0 && height > 0;
}

} // namespace denoise

#endif // DENOISE_TYPES_HPP
