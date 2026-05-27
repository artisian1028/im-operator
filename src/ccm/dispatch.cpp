#include "ccm/algorithms.hpp"
#include <string>
#include <array>
#include <algorithm>

namespace ccm {

// --- Input validation ---

CCMError validate_ccm_inputs(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth) {
    if (!input || !output) return CCMError::NullInput;
    if (!is_valid_dimensions(width, height)) return CCMError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return CCMError::InvalidBitDepth;
    if (channels != 3) return CCMError::InvalidChannels;
    return CCMError::Ok;
}

// --- Metadata ---

std::string algorithm_name(CCMAlgorithm algo) {
    switch (algo) {
        case CCMAlgorithm::LINEAR_3X3:    return "Linear 3x3 CCM";
        case CCMAlgorithm::LINEAR_4X3:    return "Linear 3x4 CCM (with bias)";
        case CCMAlgorithm::POLYNOMIAL_3X9: return "Polynomial 3x9 CCM (2nd-order)";
        case CCMAlgorithm::MANUAL:        return "Manual (user-supplied matrix)";
        default:                           return "Unknown";
    }
}

int algorithm_window_size(CCMAlgorithm algo) {
    switch (algo) {
        case CCMAlgorithm::LINEAR_3X3:    return 1;
        case CCMAlgorithm::LINEAR_4X3:    return 1;
        case CCMAlgorithm::POLYNOMIAL_3X9: return 1;
        case CCMAlgorithm::MANUAL:        return 1;
        default:                           return 1;
    }
}

// --- Registry ---

using CCMFunc = CCMError(*)(const uint8_t*, uint8_t*, int, int, int, int, const void*);

struct AlgorithmEntry {
    CCMAlgorithm algorithm;
    CCMFunc func;
};

static const std::array<AlgorithmEntry, 4> kCCMRegistry = {{
    {CCMAlgorithm::LINEAR_3X3,    process_linear_3x3},
    {CCMAlgorithm::LINEAR_4X3,    process_linear_4x3},
    {CCMAlgorithm::POLYNOMIAL_3X9, process_polynomial_3x9},
    {CCMAlgorithm::MANUAL,        process_manual_ccm}
}};

static_assert(kCCMRegistry.size() == 4,
              "kCCMRegistry size must match CCMAlgorithm enum count");

static CCMFunc find_ccm_func(CCMAlgorithm algorithm) {
    for (const auto& entry : kCCMRegistry) {
        if (entry.algorithm == algorithm) return entry.func;
    }
    return nullptr;
}

// --- Main dispatch ---

CCMError process_ccm(const uint8_t* input, uint8_t* output,
                      int width, int height, int channels,
                      CCMAlgorithm algorithm, int bit_depth,
                      const void* matrix) {
    CCMError err = validate_ccm_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != CCMError::Ok) return err;

    CCMFunc func = find_ccm_func(algorithm);
    if (!func) {
        return CCMError::InternalError;
    }

    return func(input, output, width, height, channels, bit_depth, matrix);
}

} // namespace ccm
