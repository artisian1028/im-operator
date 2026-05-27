#include "saturation/algorithms.hpp"
#include <string>
#include <array>

namespace saturation {

// --- Input validation ---

SaturationError validate_saturation_inputs(const uint8_t* input, uint8_t* output,
                                            int width, int height, int channels,
                                            int bit_depth) {
    if (!input || !output) return SaturationError::NullInput;
    if (!is_valid_dimensions(width, height)) return SaturationError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return SaturationError::InvalidBitDepth;
    if (channels != 3) return SaturationError::InvalidChannels;
    return SaturationError::Ok;
}

// --- Metadata ---

std::string algorithm_name(SaturationAlgorithm algo) {
    switch (algo) {
        case SaturationAlgorithm::HSL:           return "HSL Saturation";
        case SaturationAlgorithm::VIBRANCE:      return "Vibrance (intelligent)";
        case SaturationAlgorithm::CHANNEL_MIXER: return "Channel Mixer";
        case SaturationAlgorithm::SELECTIVE:     return "Selective (per-channel)";
        default:                                  return "Unknown";
    }
}

int algorithm_window_size(SaturationAlgorithm algo) {
    (void)algo;
    return 1;
}

// --- Registry ---

using SatFunc = SaturationError(*)(const uint8_t*, uint8_t*, int, int, int, int, const SaturationParams&);

struct AlgorithmEntry {
    SaturationAlgorithm algorithm;
    SatFunc func;
};

static const std::array<AlgorithmEntry, 4> kSaturationRegistry = {{
    {SaturationAlgorithm::HSL,           process_hsl},
    {SaturationAlgorithm::VIBRANCE,      process_vibrance},
    {SaturationAlgorithm::CHANNEL_MIXER, process_channel_mixer},
    {SaturationAlgorithm::SELECTIVE,     process_selective}
}};

static_assert(kSaturationRegistry.size() == 4,
              "kSaturationRegistry size must match SaturationAlgorithm enum count");

static SatFunc find_sat_func(SaturationAlgorithm algorithm) {
    for (const auto& entry : kSaturationRegistry) {
        if (entry.algorithm == algorithm) return entry.func;
    }
    return nullptr;
}

// --- Main dispatch ---

SaturationError process_saturation(const uint8_t* input, uint8_t* output,
                                    int width, int height, int channels,
                                    SaturationAlgorithm algorithm,
                                    int bit_depth, const SaturationParams& params) {
    SaturationError err = validate_saturation_inputs(input, output, width, height,
                                                       channels, bit_depth);
    if (err != SaturationError::Ok) return err;

    SatFunc func = find_sat_func(algorithm);
    if (!func) {
        return SaturationError::InternalError;
    }

    return func(input, output, width, height, channels, bit_depth, params);
}

} // namespace saturation
