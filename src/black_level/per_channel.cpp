#include "common.hpp"
#include "black_level/algorithms.hpp"

namespace black_level {

BlackLevelError process_per_channel(uint8_t* data,
                                     int width, int height,
                                     BayerPattern pattern,
                                     int bit_depth,
                                     const BlackLevelParams& params) {
    auto po = imop::PatternOffsets::from_pattern(pattern);
    int max_val = detail::safe_max_val(bit_depth);

    // Clamp offsets to valid range
    float offsets[4] = {
        std::max(0.0f, params.r_offset),
        std::max(0.0f, params.gr_offset),
        std::max(0.0f, params.gb_offset),
        std::max(0.0f, params.b_offset)
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int v = detail::read_bayer(data, x, y, width, bit_depth);
            int c = detail::bayer_color(y, x, po);
            int new_v = static_cast<int>(static_cast<float>(v) - offsets[c]);
            if (new_v < 0) new_v = 0;
            if (new_v > max_val) new_v = max_val;
            detail::write_bayer(data, x, y, width, bit_depth, new_v);
        }
    }
    return BlackLevelError::Ok;
}

} // namespace black_level
