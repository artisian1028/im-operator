#ifndef HDR_TYPES_HPP
#define HDR_TYPES_HPP

#include <cstdint>

namespace hdr {

enum class HdrAlgorithm {
    REINHARD,          // Classic Reinhard global: L/(1+L)
    REINHARD_EXT,      // Reinhard extended: key value + white point
    FILMIC_ACES,       // ACES RRT + ODT (Narkowicz approximation)
    HABLE,             // Uncharted 2 filmic curve
    DRAGO,             // Adaptive logarithmic tone mapping
    ADAPTIVE_LOCAL,    // Bilateral decomposition local tone mapping
    EXPONENTIAL,       // 1 - exp(-k * L)
    LOGARITHMIC,       // log(1 + k*L) / log(1 + k)
    LINEAR_TO_PQ,      // Linear → ST.2084 PQ
    PQ_TO_LINEAR,      // ST.2084 PQ → Linear
    LINEAR_TO_HLG,     // Linear → BT.2100 HLG OETF
    HLG_TO_LINEAR      // BT.2100 HLG EOTF → Linear
};

enum class HdrError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ImageTooSmall,
    CudaNotAvailable,
    InternalError
};

inline const char* hdr_error_message(HdrError err) {
    switch (err) {
        case HdrError::Ok:               return "Success";
        case HdrError::NullInput:        return "Null input/output pointer";
        case HdrError::InvalidDimensions: return "Invalid image dimensions";
        case HdrError::InvalidBitDepth:   return "Invalid bit depth (must be 1-32)";
        case HdrError::InvalidChannels:   return "Invalid channel count (must be 3)";
        case HdrError::ImageTooSmall:     return "Image too small for algorithm";
        case HdrError::CudaNotAvailable:  return "CUDA not available";
        case HdrError::InternalError:    return "Internal processing error";
        default:                          return "Unknown error";
    }
}

inline bool operator!(HdrError err) {
    return err != HdrError::Ok;
}

inline bool ok(HdrError err) {
    return err == HdrError::Ok;
}

inline bool is_valid_bit_depth(int bit_depth) {
    // 0 = float32 per channel, 1-32 = integer bit depth
    return bit_depth == 0 || (bit_depth >= 1 && bit_depth <= 32);
}

inline bool is_valid_dimensions(int width, int height) {
    return width > 0 && height > 0;
}

// HDR tone mapping parameters
struct HdrParams {
    float exposure = 0.0f;      // EV adjustment [-8, 8]
    float gamma = 2.2f;         // output gamma [0.5, 4.0]
    float saturation = 1.0f;    // post-tone-mapping saturation [0, 2]
    float key = 0.18f;          // middle gray key value [0.01, 1.0]
    float white_point = 1.0f;   // Reinhard white point [0.5, 20.0]
    float strength = 1.0f;      // general strength [0, 2]
};

// Luminance coefficients (BT.709 primaries)
constexpr float kLumaR = 0.2126f;
constexpr float kLumaG = 0.7152f;
constexpr float kLumaB = 0.0722f;

} // namespace hdr

#endif // HDR_TYPES_HPP
