#include "common.hpp"

namespace lut {

LUTError process_custom_3d(const uint8_t* input, uint8_t* output,
                            int width, int height, int channels,
                            int bit_depth, const void* lut_data, int lut_size) {
    LUTError err = validate_lut_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != LUTError::Ok) return err;

    const LUT3D* lut = static_cast<const LUT3D*>(lut_data);
    if (!lut || lut->empty()) {
        // Fallback to identity
        LUT3D identity = build_identity_lut(lut_size);
        return apply_lut(identity, input, output, width, height, bit_depth);
    }

    return apply_lut(*lut, input, output, width, height, bit_depth);
}

} // namespace lut
