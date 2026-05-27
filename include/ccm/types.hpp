#ifndef CCM_TYPES_HPP
#define CCM_TYPES_HPP

#include <cstdint>

namespace ccm {

enum class CCMAlgorithm {
    LINEAR_3X3,       // Standard 3x3 linear color correction matrix
    LINEAR_4X3,       // 4x3 matrix (RGB + offset row), supports bias term
    POLYNOMIAL_3X9,   // 2nd-order polynomial: [R G B RG RB GB R² G² B²] (3x9)
    MANUAL            // User-supplied matrix
};

enum class CCMError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ImageTooSmall,
    InternalError
};

inline const char* ccm_error_message(CCMError err) {
    switch (err) {
        case CCMError::Ok:               return "Success";
        case CCMError::NullInput:        return "Null input/output pointer";
        case CCMError::InvalidDimensions: return "Invalid image dimensions";
        case CCMError::InvalidBitDepth:   return "Invalid bit depth (must be 1-16)";
        case CCMError::InvalidChannels:   return "Invalid channel count (must be 3)";
        case CCMError::ImageTooSmall:     return "Image too small for algorithm";
        case CCMError::InternalError:    return "Internal processing error";
        default:                          return "Unknown error";
    }
}

inline bool operator!(CCMError err) {
    return err != CCMError::Ok;
}

inline bool ok(CCMError err) {
    return err == CCMError::Ok;
}

inline bool is_valid_bit_depth(int bit_depth) {
    return bit_depth >= 1 && bit_depth <= 16;
}

inline bool is_valid_dimensions(int width, int height) {
    return width > 0 && height > 0;
}

// 3x3 linear color correction matrix (row-major: out = M * in)
// out_r = m[0]*r + m[1]*g + m[2]*b
// out_g = m[3]*r + m[4]*g + m[5]*b
// out_b = m[6]*r + m[7]*g + m[8]*b
struct CCMatrix3x3 {
    CCMAlgorithm matrix_type = CCMAlgorithm::LINEAR_3X3;
    float m[9] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };
};

// 3x4 matrix with bias (offset) row
// out_r = m[0]*r + m[1]*g + m[2]*b + m[3]
// out_g = m[4]*r + m[5]*g + m[6]*b + m[7]
// out_b = m[8]*r + m[9]*g + m[10]*b + m[11]
struct CCMatrix3x4 {
    CCMAlgorithm matrix_type = CCMAlgorithm::LINEAR_4X3;
    float m[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
};

// 3x9 polynomial matrix (2nd-order expansion)
// Features: [R, G, B, RG, RB, GB, R^2, G^2, B^2]
// out_r = M_r . features, out_g = M_g . features, out_b = M_b . features
struct CCMatrix3x9 {
    CCMAlgorithm matrix_type = CCMAlgorithm::POLYNOMIAL_3X9;
    float m[27] = {
        // Row 0 (R output): coefficients for [R, G, B, RG, RB, GB, R^2, G^2, B^2]
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        // Row 1 (G output):
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        // Row 2 (B output):
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
};

} // namespace ccm

#endif // CCM_TYPES_HPP
