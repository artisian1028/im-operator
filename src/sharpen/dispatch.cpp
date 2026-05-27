#include "sharpen/algorithms.hpp"
#include <string>
#include <array>

namespace sharpen {

// --- Input validation ---

SharpenError validate_sharpen_inputs(const uint8_t* input, uint8_t* output,
                                      int width, int height, int channels,
                                      int bit_depth) {
    if (!input || !output) return SharpenError::NullInput;
    if (!is_valid_dimensions(width, height)) return SharpenError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return SharpenError::InvalidBitDepth;
    if (channels != 3) return SharpenError::InvalidChannels;
    return SharpenError::Ok;
}

// --- Metadata ---

std::string algorithm_name(SharpenAlgorithm algo) {
    switch (algo) {
        case SharpenAlgorithm::UNSHARP_MASK: return "Unsharp Mask";
        case SharpenAlgorithm::LAPLACIAN:    return "Laplacian Edge Enhancement";
        case SharpenAlgorithm::HIGH_PASS:    return "High-Pass Filter Overlay";
        case SharpenAlgorithm::ADAPTIVE:     return "Adaptive (edge-aware)";
        default:                              return "Unknown";
    }
}

int algorithm_window_size(SharpenAlgorithm algo) {
    switch (algo) {
        case SharpenAlgorithm::UNSHARP_MASK: return 5;
        case SharpenAlgorithm::LAPLACIAN:    return 5;
        case SharpenAlgorithm::HIGH_PASS:    return 3;
        case SharpenAlgorithm::ADAPTIVE:     return 5;
        default:                              return 3;
    }
}

// --- Registry ---

using SharpenFunc = SharpenError(*)(const uint8_t*, uint8_t*, int, int, int, int, const SharpenParams&);

struct AlgorithmEntry {
    SharpenAlgorithm algorithm;
    SharpenFunc func;
};

static const std::array<AlgorithmEntry, 4> kSharpenRegistry = {{
    {SharpenAlgorithm::UNSHARP_MASK, process_unsharp_mask},
    {SharpenAlgorithm::LAPLACIAN,    process_laplacian},
    {SharpenAlgorithm::HIGH_PASS,    process_high_pass},
    {SharpenAlgorithm::ADAPTIVE,     process_adaptive}
}};

static_assert(kSharpenRegistry.size() == 4,
              "kSharpenRegistry size must match SharpenAlgorithm enum count");

static SharpenFunc find_sharpen_func(SharpenAlgorithm algorithm) {
    for (const auto& entry : kSharpenRegistry) {
        if (entry.algorithm == algorithm) return entry.func;
    }
    return nullptr;
}

// --- Main dispatch ---

SharpenError process_sharpen(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              SharpenAlgorithm algorithm, int bit_depth,
                              const SharpenParams& params) {
    SharpenError err = validate_sharpen_inputs(input, output, width, height,
                                                channels, bit_depth);
    if (err != SharpenError::Ok) return err;

    SharpenFunc func = find_sharpen_func(algorithm);
    if (!func) {
        return SharpenError::InternalError;
    }

    return func(input, output, width, height, channels, bit_depth, params);
}

} // namespace sharpen
