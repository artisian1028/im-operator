#include "common.hpp"
#include "lens_shading/algorithms.hpp"

namespace lens_shading {

// Radial polynomial lens shading correction.
//
// For each pixel at distance r from optical center:
//   gain(r) = 1 + a2*r² + a4*r⁴ + a6*r⁶
//   pixel' = clamp(pixel * gain(r))
//
// Each Bayer channel (R/Gr/Gb/B) has its own polynomial coefficients.
LensShadingError process_polynomial(uint8_t* data,
                                     int width, int height,
                                     BayerPattern pattern,
                                     int bit_depth,
                                     const LensShadingParams& params) {
    auto po = imop::PatternOffsets::from_pattern(pattern);
    int max_val = detail::safe_max_val(bit_depth);

    const ShadingPolynomial* coefs[4] = {
        &params.r_coef, &params.gr_coef, &params.gb_coef, &params.b_coef
    };

    float cx = params.center_x * static_cast<float>(width);
    float cy = params.center_y * static_cast<float>(height);
    float max_r = std::sqrt(cx*cx + cy*cy); // corner distance as reference

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float dx = static_cast<float>(x) - cx;
            float dy = static_cast<float>(y) - cy;
            float r = std::sqrt(dx*dx + dy*dy) / max_r; // normalize to [0, 1]

            int c = detail::bayer_color(y, x, po);
            const auto& coef = coefs[c];

            float r2 = r * r;
            float r4 = r2 * r2;
            float r6 = r4 * r2;
            float gain = 1.0f + coef->a2 * r2 + coef->a4 * r4 + coef->a6 * r6;

            int v = detail::read_bayer(data, x, y, width, bit_depth);
            int new_v = static_cast<int>(static_cast<float>(v) * gain + 0.5f);
            if (new_v < 0) new_v = 0;
            if (new_v > max_val) new_v = max_val;
            detail::write_bayer(data, x, y, width, bit_depth, new_v);
        }
    }

    return LensShadingError::Ok;
}

} // namespace lens_shading
