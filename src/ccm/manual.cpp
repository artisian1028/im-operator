#include "common.hpp"

namespace ccm {

// Manual: delegates to the appropriate core based on matrix type.
// The caller passes a CCMatrix3x3/3x4/3x9 pointer via void* matrix. If nullptr,
// applies identity (no-op).
CCMError process_manual_ccm(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int bit_depth, const void* matrix) {
    CCMError err = validate_ccm_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != CCMError::Ok) return err;

    // If no matrix provided, apply identity via 3x3
    if (!matrix) {
        return process_linear_3x3(input, output, width, height, channels,
                                   bit_depth, nullptr);
    }

    // Read matrix type from the first field (all CCMatrix structs have CCMAlgorithm as first member)
    CCMAlgorithm matrix_type = *static_cast<const CCMAlgorithm*>(matrix);

    switch (matrix_type) {
        case CCMAlgorithm::LINEAR_3X3:
            return process_linear_3x3(input, output, width, height, channels,
                                       bit_depth, matrix);
        case CCMAlgorithm::LINEAR_4X3:
            return process_linear_4x3(input, output, width, height, channels,
                                       bit_depth, matrix);
        case CCMAlgorithm::POLYNOMIAL_3X9:
            return process_polynomial_3x9(input, output, width, height, channels,
                                           bit_depth, matrix);
        default:
            return CCMError::InternalError;
    }
}

// --- Predefined standard matrices ---

CCMatrix3x3 srgb_to_xyz_d65() {
    // sRGB linear -> CIE XYZ (D65)
    CCMatrix3x3 m;
    m.m[0] = 0.4124564f;  m.m[1] = 0.3575761f;  m.m[2] = 0.1804375f;
    m.m[3] = 0.2126729f;  m.m[4] = 0.7151522f;  m.m[5] = 0.0721750f;
    m.m[6] = 0.0193339f;  m.m[7] = 0.1191920f;  m.m[8] = 0.9503041f;
    return m;
}

CCMatrix3x3 xyz_to_srgb_d65() {
    // CIE XYZ (D65) -> sRGB linear
    CCMatrix3x3 m;
    m.m[0] =  3.2404542f;  m.m[1] = -1.5371385f;  m.m[2] = -0.4985314f;
    m.m[3] = -0.9692660f;  m.m[4] =  1.8760108f;  m.m[5] =  0.0415560f;
    m.m[6] =  0.0556434f;  m.m[7] = -0.2040259f;  m.m[8] =  1.0572252f;
    return m;
}

CCMatrix3x3 srgb_to_bt709() {
    // sRGB and BT.709 share the same primaries; identity
    CCMatrix3x3 m;
    m.m[0] = 1.0f; m.m[1] = 0.0f; m.m[2] = 0.0f;
    m.m[3] = 0.0f; m.m[4] = 1.0f; m.m[5] = 0.0f;
    m.m[6] = 0.0f; m.m[7] = 0.0f; m.m[8] = 1.0f;
    return m;
}

CCMatrix3x3 bt709_to_srgb() {
    // Identity (same primaries)
    CCMatrix3x3 m;
    m.m[0] = 1.0f; m.m[1] = 0.0f; m.m[2] = 0.0f;
    m.m[3] = 0.0f; m.m[4] = 1.0f; m.m[5] = 0.0f;
    m.m[6] = 0.0f; m.m[7] = 0.0f; m.m[8] = 1.0f;
    return m;
}

CCMatrix3x3 identity_3x3() {
    CCMatrix3x3 m;
    m.m[0] = 1.0f; m.m[1] = 0.0f; m.m[2] = 0.0f;
    m.m[3] = 0.0f; m.m[4] = 1.0f; m.m[5] = 0.0f;
    m.m[6] = 0.0f; m.m[7] = 0.0f; m.m[8] = 1.0f;
    return m;
}

CCMatrix3x3 saturation_matrix(float sat) {
    // Standard REC.709 luminance weights
    const float lr = 0.2126f;
    const float lg = 0.7152f;
    const float lb = 0.0722f;

    CCMatrix3x3 m;
    m.m[0] = lr + sat * (1.0f - lr);
    m.m[1] = lg - sat * lg;
    m.m[2] = lb - sat * lb;

    m.m[3] = lr - sat * lr;
    m.m[4] = lg + sat * (1.0f - lg);
    m.m[5] = lb - sat * lb;

    m.m[6] = lr - sat * lr;
    m.m[7] = lg - sat * lg;
    m.m[8] = lb + sat * (1.0f - lb);

    return m;
}

} // namespace ccm
