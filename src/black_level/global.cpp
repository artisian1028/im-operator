#include "common.hpp"
#include "black_level/algorithms.hpp"

namespace black_level {

BlackLevelError process_global(uint8_t* data,
                                int width, int height,
                                BayerPattern /*pattern*/,
                                int bit_depth,
                                const BlackLevelParams& params) {
    int max_val = detail::safe_max_val(bit_depth);
    float offset = std::max(0.0f, params.r_offset); // use r_offset as global

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int v = detail::read_bayer(data, x, y, width, bit_depth);
            int new_v = static_cast<int>(static_cast<float>(v) - offset);
            if (new_v < 0) new_v = 0;
            if (new_v > max_val) new_v = max_val;
            detail::write_bayer(data, x, y, width, bit_depth, new_v);
        }
    }
    return BlackLevelError::Ok;
}

} // namespace black_level
