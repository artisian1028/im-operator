#include "white_balance/algorithms.hpp"
#include <string>
#include <array>

namespace white_balance {

// --- Input validation ---

WhiteBalanceError validate_white_balance_inputs(const uint8_t* input, uint8_t* output,
                                                  int width, int height, int channels,
                                                  int bit_depth) {
    if (!input || !output) return WhiteBalanceError::NullInput;
    if (!is_valid_dimensions(width, height)) return WhiteBalanceError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return WhiteBalanceError::InvalidBitDepth;
    if (channels != 3) return WhiteBalanceError::InvalidChannels;
    return WhiteBalanceError::Ok;
}

// --- Metadata ---

std::string algorithm_name(WhiteBalanceAlgorithm algo) {
    switch (algo) {
        case WhiteBalanceAlgorithm::GRAY_WORLD:      return "Gray World (average assumption)";
        case WhiteBalanceAlgorithm::WHITE_PATCH:     return "White Patch (max RGB)";
        case WhiteBalanceAlgorithm::SHADE_OF_GRAY:   return "Shade of Gray (Minkowski norm)";
        case WhiteBalanceAlgorithm::MANUAL:          return "Manual (user-supplied gains)";
        default:                                      return "Unknown";
    }
}

int algorithm_window_size(WhiteBalanceAlgorithm algo) {
    switch (algo) {
        case WhiteBalanceAlgorithm::GRAY_WORLD:      return 1;
        case WhiteBalanceAlgorithm::WHITE_PATCH:     return 1;
        case WhiteBalanceAlgorithm::SHADE_OF_GRAY:   return 1;
        case WhiteBalanceAlgorithm::MANUAL:          return 1;
        default:                                      return 1;
    }
}

// --- Registry ---

using WBFunc = WhiteBalanceError(*)(const uint8_t*, uint8_t*, int, int, int, int, float, const WBCoefficients&);

struct AlgorithmEntry {
    WhiteBalanceAlgorithm algorithm;
    WBFunc func;
};

static const std::array<AlgorithmEntry, 4> kWBalanceRegistry = {{
    {WhiteBalanceAlgorithm::GRAY_WORLD,      process_gray_world},
    {WhiteBalanceAlgorithm::WHITE_PATCH,     process_white_patch},
    {WhiteBalanceAlgorithm::SHADE_OF_GRAY,   process_shade_of_gray},
    {WhiteBalanceAlgorithm::MANUAL,          process_manual_wb}
}};

static_assert(kWBalanceRegistry.size() == 4,
              "kWBalanceRegistry size must match WhiteBalanceAlgorithm enum count");

static WBFunc find_wb_func(WhiteBalanceAlgorithm algorithm) {
    for (const auto& entry : kWBalanceRegistry) {
        if (entry.algorithm == algorithm) return entry.func;
    }
    return nullptr;
}

// --- Main dispatch ---

WhiteBalanceError process_white_balance(const uint8_t* input, uint8_t* output,
                                          int width, int height, int channels,
                                          WhiteBalanceAlgorithm algorithm,
                                          int bit_depth, float p,
                                          const WBCoefficients& manual_gains) {
    WhiteBalanceError err = validate_white_balance_inputs(input, output, width, height,
                                                            channels, bit_depth);
    if (err != WhiteBalanceError::Ok) return err;

    if (has_cuda() && algorithm == WhiteBalanceAlgorithm::MANUAL) {
        WhiteBalanceError cuda_err = process_white_balance_cuda(input, output, width, height,
                                                                 channels, bit_depth,
                                                                 manual_gains.r_gain, manual_gains.g_gain,
                                                                 manual_gains.b_gain);
        if (cuda_err == WhiteBalanceError::Ok) return cuda_err;
    }

    WBFunc func = find_wb_func(algorithm);
    if (!func) {
        return WhiteBalanceError::InternalError;
    }

    return func(input, output, width, height, channels, bit_depth, p, manual_gains);
}

} // namespace white_balance
