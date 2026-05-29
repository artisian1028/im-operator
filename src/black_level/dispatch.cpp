#include "black_level/algorithms.hpp"
#include <string>
#include <array>

namespace black_level {

BlackLevelError validate_black_level_inputs(const uint8_t* data,
                                             int width, int height,
                                             int channels, int bit_depth) {
    if (!data) return BlackLevelError::NullInput;
    if (!is_valid_dimensions(width, height)) return BlackLevelError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return BlackLevelError::InvalidBitDepth;
    if (channels != 1) return BlackLevelError::InvalidChannels;
    return BlackLevelError::Ok;
}

std::string algorithm_name(BlackLevelAlgorithm algo) {
    switch (algo) {
        case BlackLevelAlgorithm::PER_CHANNEL: return "Per-Channel Black Level";
        case BlackLevelAlgorithm::GLOBAL:      return "Global Black Level";
        default:                                return "Unknown";
    }
}

int algorithm_window_size(BlackLevelAlgorithm algo) {
    (void)algo; return 0;
}

using BlackLevelFunc = BlackLevelError(*)(uint8_t*, int, int, BayerPattern, int, const BlackLevelParams&);

struct Entry { BlackLevelAlgorithm algo; BlackLevelFunc func; };

static const std::array<Entry, 2> kRegistry = {{
    {BlackLevelAlgorithm::PER_CHANNEL, process_per_channel},
    {BlackLevelAlgorithm::GLOBAL,      process_global}
}};
static_assert(kRegistry.size() == 2, "Registry must match enum count");

static BlackLevelFunc find(BlackLevelAlgorithm algo) {
    for (const auto& e : kRegistry) if (e.algo == algo) return e.func;
    return nullptr;
}

BlackLevelError process_black_level(uint8_t* data,
                                     int width, int height,
                                     BayerPattern pattern,
                                     BlackLevelAlgorithm algorithm,
                                     int bit_depth,
                                     const BlackLevelParams& params) {
    auto err = validate_black_level_inputs(data, width, height, 1, bit_depth);
    if (err != BlackLevelError::Ok) return err;

    if (has_cuda()) {
        BlackLevelError cuda_err = process_black_level_cuda(data, width, height,
                                                              pattern, algorithm,
                                                              bit_depth, params);
        if (cuda_err == BlackLevelError::Ok) return cuda_err;
    }

    auto f = find(algorithm);
    if (!f) return BlackLevelError::InternalError;
    return f(data, width, height, pattern, bit_depth, params);
}

} // namespace black_level
