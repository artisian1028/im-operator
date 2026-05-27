#include "common.hpp"

namespace ccm {

// Linear 3x4: out = M * [r g b 1]^T  (with bias/offset term)
// out_r = m[0]*r + m[1]*g + m[2]*b + m[3]
// out_g = m[4]*r + m[5]*g + m[6]*b + m[7]
// out_b = m[8]*r + m[9]*g + m[10]*b + m[11]
CCMError process_linear_4x3(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int bit_depth, const void* matrix) {
    CCMError err = validate_ccm_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != CCMError::Ok) return err;

    const CCMatrix3x4* mat = static_cast<const CCMatrix3x4*>(matrix);
    CCMatrix3x4 identity;
    if (!mat) {
        mat = &identity;
    }

    const float* m = mat->m;
    int max_val = detail::safe_max_val(bit_depth);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = static_cast<float>(detail::read_pixel(input, x, y, width, channels, bit_depth, 0));
            float g = static_cast<float>(detail::read_pixel(input, x, y, width, channels, bit_depth, 1));
            float b = static_cast<float>(detail::read_pixel(input, x, y, width, channels, bit_depth, 2));

            float ro = m[0]*r + m[1]*g + m[2]*b + m[3];
            float go = m[4]*r + m[5]*g + m[6]*b + m[7];
            float bo = m[8]*r + m[9]*g + m[10]*b + m[11];

            detail::write_pixel(output, x, y, width, channels, bit_depth, 0,
                                detail::clamp_val(static_cast<int>(ro + 0.5f), max_val));
            detail::write_pixel(output, x, y, width, channels, bit_depth, 1,
                                detail::clamp_val(static_cast<int>(go + 0.5f), max_val));
            detail::write_pixel(output, x, y, width, channels, bit_depth, 2,
                                detail::clamp_val(static_cast<int>(bo + 0.5f), max_val));
        }
    }

    return CCMError::Ok;
}

} // namespace ccm
