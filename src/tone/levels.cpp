#include "common.hpp"
#include <cmath>

namespace tone {

// Levels adjustment: remap input range [black_point, white_point] to [0, 1],
// then apply gamma power at the midpoint.
//
//   out = ((in - black) / (white - black)) ^ (1/gamma_mid)
//   where gamma_mid = log(mid) / log(0.5)
ToneError process_levels(const uint8_t* input, uint8_t* output,
                          int width, int height, int channels,
                          int bit_depth, const ToneParams& params) {
    ToneError err = validate_tone_inputs(input, output, width, height,
                                          channels, bit_depth);
    if (err != ToneError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);
    float mv = static_cast<float>(max_val);

    // levels params
    float black = std::max(0.0f, std::min(0.99f, params.black_point));
    float white = std::max(0.01f, std::min(1.0f, params.white_point));
    float mid   = std::max(0.01f, std::min(0.99f, params.mid_point));

    if (white <= black) white = black + 0.01f;

    // Compute gamma equivalent from mid-point
    // After linear stretch, mid_in should map to mid/mid_val equivalent.
    // Effective gamma: (0.5)^g = mid_linearized => g = log(0.5) / log(mid_linearized)
    float range = white - black;
    float mid_norm = (mid - black) / range;
    float gamma;
    if (mid_norm > 0.0f && mid_norm < 1.0f) {
        gamma = std::log(0.5f) / std::log(mid_norm);
    } else {
        gamma = 1.0f;  // identity gamma for edge cases
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                int val = detail::read_pixel(input, x, y, width, channels, bit_depth, c);
                float norm = static_cast<float>(val) / mv;

                // Linear stretch
                float stretched = (norm - black) / range;
                stretched = std::max(0.0f, std::min(1.0f, stretched));

                // Gamma
                float corrected = std::pow(stretched, 1.0f / gamma);

                detail::write_pixel(output, x, y, width, channels, bit_depth, c,
                                    detail::clamp_val(static_cast<int>(corrected * mv + 0.5f), max_val));
            }
        }
    }

    return ToneError::Ok;
}

} // namespace tone
