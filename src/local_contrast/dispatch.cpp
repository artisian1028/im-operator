#include "local_contrast/algorithms.hpp"
#include <string>
#include <array>

namespace local_contrast {

LocalContrastError validate_local_contrast_inputs(const uint8_t* input, uint8_t* output,
                                                   int width, int height,
                                                   int channels, int bit_depth) {
    if (!input || !output) return LocalContrastError::NullInput;
    if (!is_valid_dimensions(width, height)) return LocalContrastError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return LocalContrastError::InvalidBitDepth;
    if (channels != 3) return LocalContrastError::InvalidChannels;
    return LocalContrastError::Ok;
}

std::string algorithm_name(LocalContrastAlgorithm algo) {
    switch (algo) {
        case LocalContrastAlgorithm::UNSHARP:   return "Unsharp Clarity";
        case LocalContrastAlgorithm::BILATERAL: return "Bilateral Clarity";
        default: return "Unknown";
    }
}

int algorithm_window_size(LocalContrastAlgorithm algo) {
    return 11;
}

using LFunc = LocalContrastError(*)(const uint8_t*, uint8_t*, int, int, int, int, const LocalContrastParams&);

struct Entry { LocalContrastAlgorithm algo; LFunc func; };

static const std::array<Entry, 2> kRegistry = {{
    {LocalContrastAlgorithm::UNSHARP,   process_unsharp},
    {LocalContrastAlgorithm::BILATERAL, process_bilateral}
}};
static_assert(kRegistry.size() == 2, "Registry must match enum count");

static LFunc find(LocalContrastAlgorithm algo) {
    for (const auto& e : kRegistry) if (e.algo == algo) return e.func;
    return nullptr;
}

LocalContrastError process_local_contrast(const uint8_t* input, uint8_t* output,
                                           int width, int height, int channels,
                                           LocalContrastAlgorithm algorithm,
                                           int bit_depth,
                                           const LocalContrastParams& params) {
    auto err = validate_local_contrast_inputs(input, output, width, height, channels, bit_depth);
    if (err != LocalContrastError::Ok) return err;

    if (has_cuda()) {
        LocalContrastError cuda_err = process_local_contrast_cuda(input, output, width, height,
                                                                    channels, bit_depth, params);
        if (cuda_err == LocalContrastError::Ok) return cuda_err;
    }

    auto f = find(algorithm);
    if (!f) return LocalContrastError::InternalError;
    return f(input, output, width, height, channels, bit_depth, params);
}

} // namespace local_contrast
