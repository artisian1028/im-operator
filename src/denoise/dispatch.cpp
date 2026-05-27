#include "denoise/algorithms.hpp"
#include <string>
#include <array>

namespace denoise {

// --- Input validation ---

DenoiseError validate_denoise_inputs(const uint8_t* input, uint8_t* output,
                                     int width, int height, int channels,
                                     int bit_depth) {
    if (!input || !output) return DenoiseError::NullInput;
    if (!is_valid_dimensions(width, height)) return DenoiseError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return DenoiseError::InvalidBitDepth;
    if (channels != 1 && channels != 3) return DenoiseError::InvalidChannels;
    return DenoiseError::Ok;
}

// --- Metadata ---

std::string algorithm_name(DenoiseAlgorithm algo) {
    switch (algo) {
        case DenoiseAlgorithm::GAUSSIAN:       return "Gaussian (separable blur)";
        case DenoiseAlgorithm::MEDIAN:         return "Median (3x3)";
        case DenoiseAlgorithm::BILATERAL:      return "Bilateral (edge-preserving)";
        case DenoiseAlgorithm::NLM:            return "NLM (Non-Local Means)";
        case DenoiseAlgorithm::WAVELET:        return "Wavelet (Soft Threshold)";
        case DenoiseAlgorithm::BAYER_DENOISE:  return "Bayer-domain denoise";
        default:                                return "Unknown";
    }
}

int algorithm_window_size(DenoiseAlgorithm algo) {
    switch (algo) {
        case DenoiseAlgorithm::GAUSSIAN:       return 5;
        case DenoiseAlgorithm::MEDIAN:         return 3;
        case DenoiseAlgorithm::BILATERAL:      return 5;
        case DenoiseAlgorithm::NLM:            return 7;
        case DenoiseAlgorithm::WAVELET:        return 8;
        case DenoiseAlgorithm::BAYER_DENOISE:  return 5;
        default:                                return 3;
    }
}

// --- Registry ---

using DenoiseFunc = DenoiseError(*)(const uint8_t*, uint8_t*, int, int, int, int, float);

struct AlgorithmEntry {
    DenoiseAlgorithm algorithm;
    DenoiseFunc func;
};

static const std::array<AlgorithmEntry, 6> kDenoiseRegistry = {{
    {DenoiseAlgorithm::GAUSSIAN,       process_gaussian},
    {DenoiseAlgorithm::MEDIAN,         process_median},
    {DenoiseAlgorithm::BILATERAL,       process_bilateral},
    {DenoiseAlgorithm::NLM,             process_nlm},
    {DenoiseAlgorithm::WAVELET,         process_wavelet},
    {DenoiseAlgorithm::BAYER_DENOISE,  process_bayer_denoise}
}};

static_assert(kDenoiseRegistry.size() == 6,
              "kDenoiseRegistry size must match DenoiseAlgorithm enum count");

static DenoiseFunc find_denoise_func(DenoiseAlgorithm algorithm) {
    for (const auto& entry : kDenoiseRegistry) {
        if (entry.algorithm == algorithm) return entry.func;
    }
    return nullptr;
}

// --- Main dispatch ---

DenoiseError process_denoise(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             DenoiseAlgorithm algorithm, int bit_depth,
                             float strength) {
    DenoiseError err = validate_denoise_inputs(input, output, width, height,
                                                channels, bit_depth);
    if (err != DenoiseError::Ok) return err;

    DenoiseFunc func = find_denoise_func(algorithm);
    if (!func) {
        return DenoiseError::InternalError;
    }

    return func(input, output, width, height, channels, bit_depth, strength);
}

} // namespace denoise
