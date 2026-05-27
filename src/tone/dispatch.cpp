#include "tone/algorithms.hpp"
#include <string>
#include <array>

namespace tone {

// --- Input validation ---

ToneError validate_tone_inputs(const uint8_t* input, uint8_t* output,
                                int width, int height, int channels,
                                int bit_depth) {
    if (!input || !output) return ToneError::NullInput;
    if (!is_valid_dimensions(width, height)) return ToneError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return ToneError::InvalidBitDepth;
    if (channels != 3) return ToneError::InvalidChannels;
    return ToneError::Ok;
}

// --- Metadata ---

std::string algorithm_name(ToneAlgorithm algo) {
    switch (algo) {
        case ToneAlgorithm::GAMMA:              return "Gamma Correction";
        case ToneAlgorithm::S_CURVE:            return "S-Curve Contrast";
        case ToneAlgorithm::LEVELS:             return "Levels Adjustment";
        case ToneAlgorithm::CURVES_3POINT:      return "3-Point Curves";
        case ToneAlgorithm::SHADOWS_HIGHLIGHTS: return "Shadows / Highlights";
        default:                                 return "Unknown";
    }
}

int algorithm_window_size(ToneAlgorithm algo) {
    (void)algo;
    return 1;
}

// --- Registry ---

using ToneFunc = ToneError(*)(const uint8_t*, uint8_t*, int, int, int, int, const ToneParams&);

struct AlgorithmEntry {
    ToneAlgorithm algorithm;
    ToneFunc func;
};

static const std::array<AlgorithmEntry, 5> kToneRegistry = {{
    {ToneAlgorithm::GAMMA,              process_gamma},
    {ToneAlgorithm::S_CURVE,            process_s_curve},
    {ToneAlgorithm::LEVELS,             process_levels},
    {ToneAlgorithm::CURVES_3POINT,      process_curves_3point},
    {ToneAlgorithm::SHADOWS_HIGHLIGHTS, process_shadows_highlights}
}};

static_assert(kToneRegistry.size() == 5,
              "kToneRegistry size must match ToneAlgorithm enum count");

static ToneFunc find_tone_func(ToneAlgorithm algorithm) {
    for (const auto& entry : kToneRegistry) {
        if (entry.algorithm == algorithm) return entry.func;
    }
    return nullptr;
}

// --- Main dispatch ---

ToneError process_tone(const uint8_t* input, uint8_t* output,
                        int width, int height, int channels,
                        ToneAlgorithm algorithm, int bit_depth,
                        const ToneParams& params) {
    ToneError err = validate_tone_inputs(input, output, width, height,
                                          channels, bit_depth);
    if (err != ToneError::Ok) return err;

    ToneFunc func = find_tone_func(algorithm);
    if (!func) {
        return ToneError::InternalError;
    }

    return func(input, output, width, height, channels, bit_depth, params);
}

} // namespace tone
