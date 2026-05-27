#include "common.hpp"
#include <cmath>

namespace tone {

// 3-Point Cubic Bezier tone curve.
//
// Control points:
//   P0 = (0, 0 + shadows)     — black point
//   P1 = (mid, mid + contrast_mid) — mid-tone anchor
//   P2 = (1, 1 + highlights)  — white point
//
// Uses a 4-control-point cubic Bezier with P0,P1,P2,P3 (P3 at (1,1)).
// The curve is evaluated as a function t -> x, then y is looked up.
//
// Simplified: use two quadratic segments meeting at the mid-point
// for a smooth S-like tone curve.
ToneError process_curves_3point(const uint8_t* input, uint8_t* output,
                                 int width, int height, int channels,
                                 int bit_depth, const ToneParams& params) {
    ToneError err = validate_tone_inputs(input, output, width, height,
                                          channels, bit_depth);
    if (err != ToneError::Ok) return err;

    int max_val = detail::safe_max_val(bit_depth);
    float mv = static_cast<float>(max_val);

    float shadows = std::max(-1.0f, std::min(1.0f, params.shadows));
    float contrast = std::max(-1.0f, std::min(2.0f, params.contrast));
    float highlights = std::max(-1.0f, std::min(1.0f, params.highlights));

    // Three control points (x in [0,1], y in [0,1]):
    // P0 = (0, shadows * 0.1f)        — lift/sink blacks
    // P1 = (0.5, 0.5 + contrast * 0.15f)  — mid-tone push
    // P2 = (1, 1.0f + highlights * 0.1f)  — lift/sink whites

    float y0 = std::max(0.0f, std::min(1.0f, shadows * 0.1f));
    float y1 = std::max(0.0f, std::min(1.0f, 0.5f + contrast * 0.15f));
    float y2 = std::max(0.0f, std::min(1.0f, 1.0f + highlights * 0.1f));

    // Quadratic Bezier on [0, 0.5]: control points P0, Q0, P1
    // where Q0 is chosen so the curve passes through (0.25, lerp(y0, y1, 0.5)).
    float q0x = 0.25f, q0y = 0.5f * (y0 + y1);

    // Quadratic Bezier on [0.5, 1]: control points P1, Q1, P2
    float q1x = 0.75f, q1y = 0.5f * (y1 + y2);

    // Evaluate quadratic Bezier at parameter t for endpoints A, C, control B
    // B(t) = (1-t)^2*A + 2*(1-t)*t*B + t^2*C
    auto quad_bezier = [](float t, float ax, float ay,
                                   float bx, float by,
                                   float cx, float cy) -> float {
        float t1 = 1.0f - t;
        // We only need y, but x drives t; we invert x->t via linear mapping.
        float x = t1 * t1 * ax + 2.0f * t1 * t * bx + t * t * cx;
        (void)x;
        return t1 * t1 * ay + 2.0f * t1 * t * by + t * t * cy;
    };

    // For simplicity, sample the curve into a 256-entry LUT
    float curve_lut[256];
    for (int i = 0; i < 256; i++) {
        float norm = static_cast<float>(i) / 255.0f;
        float result;

        if (norm <= 0.5f) {
            float t = norm * 2.0f; // map [0,0.5] -> [0,1]
            result = quad_bezier(t, 0.0f, y0, q0x, q0y, 0.5f, y1);
        } else {
            float t = (norm - 0.5f) * 2.0f; // map [0.5,1] -> [0,1]
            result = quad_bezier(t, 0.5f, y1, q1x, q1y, 1.0f, y2);
        }

        curve_lut[i] = std::max(0.0f, std::min(1.0f, result));
    }

    // Apply LUT with linear interpolation
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                int val = detail::read_pixel(input, x, y, width, channels, bit_depth, c);

                float norm;
                int idx;
                float frac;
                if (bit_depth <= 8) {
                    idx = val;
                    frac = 0.0f;
                    if (idx > 254) { idx = 254; frac = 1.0f; }
                    norm = curve_lut[idx] * (1.0f - frac) + curve_lut[idx + 1] * frac;
                } else {
                    norm = static_cast<float>(val) / mv;
                    float fi = norm * 255.0f;
                    idx = static_cast<int>(fi);
                    frac = fi - static_cast<float>(idx);
                    if (idx > 254) { idx = 254; frac = 1.0f; }
                    norm = curve_lut[idx] * (1.0f - frac) + curve_lut[idx + 1] * frac;
                }

                detail::write_pixel(output, x, y, width, channels, bit_depth, c,
                                    detail::clamp_val(static_cast<int>(norm * mv + 0.5f), max_val));
            }
        }
    }

    return ToneError::Ok;
}

} // namespace tone
