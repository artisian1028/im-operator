#include "common.hpp"

namespace lut {

LUTError process_style_sepia(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth, const void* /*lut_data*/, int lut_size) {
    LUTError err = validate_lut_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != LUTError::Ok) return err;

    LUT3D lut = build_sepia_lut(lut_size);
    return apply_lut(lut, input, output, width, height, bit_depth);
}

LUTError process_style_cool(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int bit_depth, const void* /*lut_data*/, int lut_size) {
    LUTError err = validate_lut_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != LUTError::Ok) return err;

    LUT3D lut = build_cool_lut(lut_size);
    return apply_lut(lut, input, output, width, height, bit_depth);
}

LUTError process_style_warm(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int bit_depth, const void* /*lut_data*/, int lut_size) {
    LUTError err = validate_lut_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != LUTError::Ok) return err;

    LUT3D lut = build_warm_lut(lut_size);
    return apply_lut(lut, input, output, width, height, bit_depth);
}

LUTError process_style_high_contrast(const uint8_t* input, uint8_t* output,
                                      int width, int height, int channels,
                                      int bit_depth, const void* /*lut_data*/, int lut_size) {
    LUTError err = validate_lut_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != LUTError::Ok) return err;

    LUT3D lut = build_high_contrast_lut(lut_size);
    return apply_lut(lut, input, output, width, height, bit_depth);
}

LUTError process_style_low_contrast(const uint8_t* input, uint8_t* output,
                                     int width, int height, int channels,
                                     int bit_depth, const void* /*lut_data*/, int lut_size) {
    LUTError err = validate_lut_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != LUTError::Ok) return err;

    LUT3D lut = build_low_contrast_lut(lut_size);
    return apply_lut(lut, input, output, width, height, bit_depth);
}

LUTError process_style_invert(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, const void* /*lut_data*/, int lut_size) {
    LUTError err = validate_lut_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != LUTError::Ok) return err;

    LUT3D lut = build_invert_lut(lut_size);
    return apply_lut(lut, input, output, width, height, bit_depth);
}

LUTError process_style_vintage_fade(const uint8_t* input, uint8_t* output,
                                     int width, int height, int channels,
                                     int bit_depth, const void* /*lut_data*/, int lut_size) {
    LUTError err = validate_lut_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != LUTError::Ok) return err;

    LUT3D lut = build_vintage_fade_lut(lut_size);
    return apply_lut(lut, input, output, width, height, bit_depth);
}

} // namespace lut
