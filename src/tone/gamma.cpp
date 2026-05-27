#include "common.hpp"
#include <cmath>

namespace tone {

// Gamma correction: out = in ^ (1/gamma)
//   gamma < 1: darken midtones
//   gamma = 1: identity
//   gamma > 1: brighten midtones
ToneError process_gamma(const uint8_t* input, uint8_t* output,
                         int width, int height, int channels,
                         int bit_depth, const ToneParams& params) {
    ToneError err = validate_tone_inputs(input, output, width, height,
                                          channels, bit_depth);
    if (err != ToneError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);
    float mv = static_cast<float>(max_val);
    float g = std::max(0.1f, std::min(10.0f, params.gamma));
    float inv_gamma = 1.0f / g;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                int val = detail::read_pixel(input, x, y, width, channels, bit_depth, c);
                float norm = static_cast<float>(val) / mv;
                float corrected = std::pow(norm, inv_gamma);
                detail::write_pixel(output, x, y, width, channels, bit_depth, c,
                                    detail::clamp_val(static_cast<int>(corrected * mv + 0.5f), max_val));
            }
        }
    }

    return ToneError::Ok;
}

} // namespace tone
