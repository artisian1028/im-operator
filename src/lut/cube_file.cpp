#include "common.hpp"
#include <cstring>

namespace lut {

LUTError process_cube_file(const uint8_t* input, uint8_t* output,
                            int width, int height, int channels,
                            int bit_depth, const void* lut_data, int /*lut_size*/) {
    LUTError err = validate_lut_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != LUTError::Ok) return err;

    const char* filepath = static_cast<const char*>(lut_data);
    if (!filepath) return LUTError::FileNotFound;

    LUT3D lut = load_cube_file(filepath);
    if (lut.empty()) return LUTError::FileParseError;

    return apply_lut(lut, input, output, width, height, bit_depth);
}

} // namespace lut
