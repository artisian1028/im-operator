#include "highlight_reconstruct/algorithms.hpp"
#include <string>
#include <array>

namespace highlight_reconstruct {

HighlightReconstructError validate_highlight_reconstruct_inputs(const uint8_t* input,
                                                                  uint8_t* output,
                                                                  int width, int height,
                                                                  int channels, int bit_depth) {
    if (!input || !output) return HighlightReconstructError::NullInput;
    if (!is_valid_dimensions(width, height)) return HighlightReconstructError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return HighlightReconstructError::InvalidBitDepth;
    if (channels != 3) return HighlightReconstructError::InvalidChannels;
    return HighlightReconstructError::Ok;
}

std::string algorithm_name(HighlightReconstructAlgorithm algo) {
    switch (algo) {
        case HighlightReconstructAlgorithm::CHANNEL_GUIDED: return "Channel-Guided Highlight Reconstruct";
        case HighlightReconstructAlgorithm::GRADIENT_BASED: return "Gradient-Based Highlight Reconstruct";
        default: return "Unknown";
    }
}

int algorithm_window_size(HighlightReconstructAlgorithm algo) {
    return (algo == HighlightReconstructAlgorithm::GRADIENT_BASED) ? 5 : 1;
}

using HFunc = HighlightReconstructError(*)(const uint8_t*, uint8_t*, int, int, int, int, const HighlightReconstructParams&);

struct Entry { HighlightReconstructAlgorithm algo; HFunc func; };

static const std::array<Entry, 2> kRegistry = {{
    {HighlightReconstructAlgorithm::CHANNEL_GUIDED, process_channel_guided},
    {HighlightReconstructAlgorithm::GRADIENT_BASED, process_gradient_based}
}};
static_assert(kRegistry.size() == 2, "Registry must match enum count");

static HFunc find(HighlightReconstructAlgorithm algo) {
    for (const auto& e : kRegistry) if (e.algo == algo) return e.func;
    return nullptr;
}

HighlightReconstructError process_highlight_reconstruct(const uint8_t* input, uint8_t* output,
                                                         int width, int height, int channels,
                                                         HighlightReconstructAlgorithm algorithm,
                                                         int bit_depth,
                                                         const HighlightReconstructParams& params) {
    auto err = validate_highlight_reconstruct_inputs(input, output, width, height, channels, bit_depth);
    if (err != HighlightReconstructError::Ok) return err;
    auto f = find(algorithm);
    if (!f) return HighlightReconstructError::InternalError;
    return f(input, output, width, height, channels, bit_depth, params);
}

} // namespace highlight_reconstruct
