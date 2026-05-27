#ifndef WHITE_BALANCE_TYPES_HPP
#define WHITE_BALANCE_TYPES_HPP

#include <cstdint>

namespace white_balance {

enum class WhiteBalanceAlgorithm {
    GRAY_WORLD,       // Average R/G/B should be equal; compute gains
    WHITE_PATCH,       // Brightest pixel is the white point (Max RGB)
    SHADE_OF_GRAY,     // Minkowski-norm generalization of Gray World
    MANUAL             // User-supplied RGB gains
};

enum class WhiteBalanceError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ImageTooSmall,
    InternalError
};

inline const char* white_balance_error_message(WhiteBalanceError err) {
    switch (err) {
        case WhiteBalanceError::Ok:               return "Success";
        case WhiteBalanceError::NullInput:        return "Null input/output pointer";
        case WhiteBalanceError::InvalidDimensions: return "Invalid image dimensions";
        case WhiteBalanceError::InvalidBitDepth:   return "Invalid bit depth (must be 1-16)";
        case WhiteBalanceError::InvalidChannels:   return "Invalid channel count (must be 3)";
        case WhiteBalanceError::ImageTooSmall:     return "Image too small for algorithm";
        case WhiteBalanceError::InternalError:    return "Internal processing error";
        default:                                   return "Unknown error";
    }
}

inline bool operator!(WhiteBalanceError err) {
    return err != WhiteBalanceError::Ok;
}

inline bool ok(WhiteBalanceError err) {
    return err == WhiteBalanceError::Ok;
}

inline bool is_valid_bit_depth(int bit_depth) {
    return bit_depth >= 1 && bit_depth <= 16;
}

inline bool is_valid_dimensions(int width, int height) {
    return width > 0 && height > 0;
}

struct WBCoefficients {
    float r_gain = 1.0f;
    float g_gain = 1.0f;
    float b_gain = 1.0f;
};

} // namespace white_balance

#endif // WHITE_BALANCE_TYPES_HPP
