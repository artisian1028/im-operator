#include "common.hpp"

namespace ccm {

// Polynomial 3x9: second-order expansion
// Features: [R, G, B, R*G, R*B, G*B, R², G², B²]
// out_r = sum(M_r[i] * features[i]), same for G and B
CCMError process_polynomial_3x9(const uint8_t* input, uint8_t* output,
                                 int width, int height, int channels,
                                 int bit_depth, const void* matrix) {
    CCMError err = validate_ccm_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != CCMError::Ok) return err;

    const CCMatrix3x9* mat = static_cast<const CCMatrix3x9*>(matrix);
    CCMatrix3x9 identity;
    if (!mat) {
        mat = &identity;
    }

    const float* m = mat->m;
    int max_val = detail::safe_max_val(bit_depth);
    float mv = static_cast<float>(max_val);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float r = static_cast<float>(detail::read_pixel(input, x, y, width, channels, bit_depth, 0));
            float g = static_cast<float>(detail::read_pixel(input, x, y, width, channels, bit_depth, 1));
            float b = static_cast<float>(detail::read_pixel(input, x, y, width, channels, bit_depth, 2));

            // Normalize to [0, 1] for polynomial stability
            float rn = r / mv;
            float gn = g / mv;
            float bn = b / mv;

            float feat[9] = {
                rn, gn, bn,
                rn * gn, rn * bn, gn * bn,
                rn * rn, gn * gn, bn * bn
            };

            // Row 0 (R output)
            float ro = 0.0f;
            for (int i = 0; i < 9; i++) ro += m[i] * feat[i];
            // Row 1 (G output)
            float go = 0.0f;
            for (int i = 0; i < 9; i++) go += m[9 + i] * feat[i];
            // Row 2 (B output)
            float bo = 0.0f;
            for (int i = 0; i < 9; i++) bo += m[18 + i] * feat[i];

            detail::write_pixel(output, x, y, width, channels, bit_depth, 0,
                                detail::clamp_val(static_cast<int>(ro * mv + 0.5f), max_val));
            detail::write_pixel(output, x, y, width, channels, bit_depth, 1,
                                detail::clamp_val(static_cast<int>(go * mv + 0.5f), max_val));
            detail::write_pixel(output, x, y, width, channels, bit_depth, 2,
                                detail::clamp_val(static_cast<int>(bo * mv + 0.5f), max_val));
        }
    }

    return CCMError::Ok;
}

} // namespace ccm
