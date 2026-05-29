#include "common.hpp"
#include "lens_shading/algorithms.hpp"
#include <vector>
#include <cstring>

namespace lens_shading {

// Flat-field lens shading correction.
//
// A flat-field image is captured by photographing a uniformly-lit white surface
// with the same lens/aperture. The gain at each pixel is:
//   gain(x,y) = target / flat_field(x,y)
// where target is the maximum (center) value of the flat field.
//
// For efficiency, we downsample the gain map to a smaller grid and bilinear-interpolate.
LensShadingError process_flat_field(uint8_t* data,
                                     int width, int height,
                                     BayerPattern pattern,
                                     int bit_depth,
                                     const LensShadingParams& params) {
    if (!params.flat_field) return LensShadingError::NullInput;
    if (params.flat_field_width <= 0 || params.flat_field_height <= 0)
        return LensShadingError::InvalidDimensions;

    auto po = imop::PatternOffsets::from_pattern(pattern);
    int max_val = detail::safe_max_val(bit_depth);
    int ffw = params.flat_field_width;
    int ffh = params.flat_field_height;

    // Compute per-channel target (center average of flat field, per Bayer channel)
    float targets[4] = {0};
    int counts[4] = {0};
    int cx0 = ffw * 3 / 10, cx1 = ffw * 7 / 10;
    int cy0 = ffh * 3 / 10, cy1 = ffh * 7 / 10;

    for (int y = cy0; y < cy1; y += 2) {
        for (int x = cx0; x < cx1; x += 2) {
            int c = detail::bayer_color(y, x, po);
            int v = detail::read_bayer(params.flat_field, x, y, ffw, bit_depth);
            targets[c] += static_cast<float>(v);
            counts[c]++;
        }
    }
    for (int c = 0; c < 4; c++) {
        targets[c] = (counts[c] > 0) ? targets[c] / static_cast<float>(counts[c]) : static_cast<float>(max_val);
    }

    // Downsample gain map: use 32x24 grid for efficiency
    const int gw = 32, gh = 24;
    std::vector<float> gain_map(gw * gh * 4, 1.0f);

    for (int gy = 0; gy < gh; gy++) {
        for (int gx = 0; gx < gw; gx++) {
            int fx0 = gx * ffw / gw;
            int fx1 = (gx + 1) * ffw / gw;
            int fy0 = gy * ffh / gh;
            int fy1 = (gy + 1) * ffh / gh;

            float sum[4] = {0};
            int cnt[4] = {0};

            for (int y = fy0; y < fy1; y += 2) {
                for (int x = fx0; x < fx1; x += 2) {
                    int c = detail::bayer_color(y, x, po);
                    int v = detail::read_bayer(params.flat_field, x, y, ffw, bit_depth);
                    sum[c] += static_cast<float>(v);
                    cnt[c]++;
                }
            }

            for (int c = 0; c < 4; c++) {
                float mean = (cnt[c] > 0) ? sum[c] / static_cast<float>(cnt[c]) : targets[c];
                gain_map[(gy * gw + gx) * 4 + c] = targets[c] / std::max(1e-6f, mean);
            }
        }
    }

    // Apply gain map with bilinear interpolation
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int c = detail::bayer_color(y, x, po);

            // Map pixel to gain grid coordinates
            float gx_f = static_cast<float>(x) * static_cast<float>(gw - 1) / static_cast<float>(width - 1);
            float gy_f = static_cast<float>(y) * static_cast<float>(gh - 1) / static_cast<float>(height - 1);
            int gx_i = std::clamp(static_cast<int>(gx_f), 0, gw - 2);
            int gy_i = std::clamp(static_cast<int>(gy_f), 0, gh - 2);
            float fx = gx_f - static_cast<float>(gx_i);
            float fy = gy_f - static_cast<float>(gy_i);

            float g00 = gain_map[(gy_i * gw + gx_i) * 4 + c];
            float g10 = gain_map[(gy_i * gw + gx_i + 1) * 4 + c];
            float g01 = gain_map[((gy_i + 1) * gw + gx_i) * 4 + c];
            float g11 = gain_map[((gy_i + 1) * gw + gx_i + 1) * 4 + c];

            float gain = g00 * (1.0f - fx) * (1.0f - fy)
                       + g10 * fx * (1.0f - fy)
                       + g01 * (1.0f - fx) * fy
                       + g11 * fx * fy;

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
