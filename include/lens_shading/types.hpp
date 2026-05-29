#ifndef LENS_SHADING_TYPES_HPP
#define LENS_SHADING_TYPES_HPP

#include <cstdint>

namespace lens_shading {

enum class LensShadingAlgorithm {
    POLYNOMIAL,   // Radial polynomial gain: 1 + a2*r^2 + a4*r^4 + a6*r^6
    FLAT_FIELD    // Gain map from a flat-field (uniform white) reference image
};

enum class LensShadingError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ImageTooSmall,
    InternalError
};

inline const char* lens_shading_error_message(LensShadingError err) {
    switch (err) {
        case LensShadingError::Ok:               return "Success";
        case LensShadingError::NullInput:        return "Null input/output pointer";
        case LensShadingError::InvalidDimensions: return "Invalid image dimensions";
        case LensShadingError::InvalidBitDepth:   return "Invalid bit depth (must be 1-16)";
        case LensShadingError::InvalidChannels:   return "Invalid channel count (must be 1 for Bayer)";
        case LensShadingError::ImageTooSmall:     return "Image too small for algorithm";
        case LensShadingError::InternalError:    return "Internal processing error";
        default:                                  return "Unknown error";
    }
}

inline bool operator!(LensShadingError err) { return err != LensShadingError::Ok; }
inline bool ok(LensShadingError err) { return err == LensShadingError::Ok; }
inline bool is_valid_bit_depth(int bd) { return bd >= 1 && bd <= 16; }
inline bool is_valid_dimensions(int w, int h) { return w > 0 && h > 0; }

// Polynomial coefficients per channel: gain(r) = 1 + a2*r² + a4*r⁴ + a6*r⁶
// r is normalized distance from optical center: r ∈ [0, 1]
struct ShadingPolynomial {
    float a2 = 0.0f;
    float a4 = 0.0f;
    float a6 = 0.0f;
};

struct LensShadingParams {
    ShadingPolynomial r_coef;   // R channel polynomial
    ShadingPolynomial gr_coef;  // Gr channel polynomial
    ShadingPolynomial gb_coef;  // Gb channel polynomial
    ShadingPolynomial b_coef;   // B channel polynomial
    float center_x = 0.5f;      // optical center x (normalized 0-1)
    float center_y = 0.5f;      // optical center y (normalized 0-1)
    // Flat-field inputs
    const uint8_t* flat_field = nullptr;  // flat-field image (same format as input)
    int flat_field_width = 0;
    int flat_field_height = 0;
};

} // namespace lens_shading

#endif
