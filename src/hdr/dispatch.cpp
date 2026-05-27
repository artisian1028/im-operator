#include "hdr/algorithms.hpp"
#include <string>
#include <array>

namespace hdr {

// --- Input validation ---

HdrError validate_hdr_inputs(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth) {
    if (!input || !output) return HdrError::NullInput;
    if (!is_valid_dimensions(width, height)) return HdrError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return HdrError::InvalidBitDepth;
    if (channels != 3) return HdrError::InvalidChannels;
    return HdrError::Ok;
}

// --- Metadata ---

std::string algorithm_name(HdrAlgorithm algo) {
    switch (algo) {
        case HdrAlgorithm::REINHARD:       return "Reinhard";
        case HdrAlgorithm::REINHARD_EXT:   return "Reinhard Extended";
        case HdrAlgorithm::FILMIC_ACES:    return "Filmic ACES";
        case HdrAlgorithm::HABLE:          return "Hable (Uncharted 2)";
        case HdrAlgorithm::DRAGO:          return "Drago Adaptive Log";
        case HdrAlgorithm::ADAPTIVE_LOCAL: return "Adaptive Local";
        case HdrAlgorithm::EXPONENTIAL:    return "Exponential";
        case HdrAlgorithm::LOGARITHMIC:    return "Logarithmic";
        case HdrAlgorithm::LINEAR_TO_PQ:   return "Linear -> PQ (ST.2084)";
        case HdrAlgorithm::PQ_TO_LINEAR:   return "PQ (ST.2084) -> Linear";
        case HdrAlgorithm::LINEAR_TO_HLG:  return "Linear -> HLG (BT.2100)";
        case HdrAlgorithm::HLG_TO_LINEAR:  return "HLG (BT.2100) -> Linear";
        default:                            return "Unknown";
    }
}

int algorithm_window_size(HdrAlgorithm algo) {
    if (algo == HdrAlgorithm::ADAPTIVE_LOCAL) return 5;
    return 1;
}

// --- Registry ---

using HdrFunc = HdrError(*)(const uint8_t*, uint8_t*, int, int, int, int, const HdrParams&);

struct AlgorithmEntry {
    HdrAlgorithm algorithm;
    HdrFunc func;
};

static const std::array<AlgorithmEntry, 12> kHdrRegistry = {{
    {HdrAlgorithm::REINHARD,       process_reinhard},
    {HdrAlgorithm::REINHARD_EXT,   process_reinhard_ext},
    {HdrAlgorithm::FILMIC_ACES,    process_filmic_aces},
    {HdrAlgorithm::HABLE,          process_hable},
    {HdrAlgorithm::DRAGO,          process_drago},
    {HdrAlgorithm::ADAPTIVE_LOCAL, process_adaptive_local},
    {HdrAlgorithm::EXPONENTIAL,    process_exponential},
    {HdrAlgorithm::LOGARITHMIC,    process_logarithmic},
    {HdrAlgorithm::LINEAR_TO_PQ,   process_linear_to_pq},
    {HdrAlgorithm::PQ_TO_LINEAR,   process_pq_to_linear},
    {HdrAlgorithm::LINEAR_TO_HLG,  process_linear_to_hlg},
    {HdrAlgorithm::HLG_TO_LINEAR,  process_hlg_to_linear}
}};

static_assert(kHdrRegistry.size() == 12,
              "kHdrRegistry size must match HdrAlgorithm enum count");

static HdrFunc find_hdr_func(HdrAlgorithm algorithm) {
    for (const auto& entry : kHdrRegistry) {
        if (entry.algorithm == algorithm) return entry.func;
    }
    return nullptr;
}

// --- Main dispatch ---

HdrError process_hdr(const uint8_t* input, uint8_t* output,
                      int width, int height, int channels,
                      HdrAlgorithm algorithm, int bit_depth,
                      const HdrParams& params) {
    HdrError err = validate_hdr_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != HdrError::Ok) return err;

    // Try GPU acceleration first; transparent fallback to CPU
    if (has_cuda()) {
        HdrError cuda_err = process_hdr_cuda(input, output, width, height,
                                               channels, algorithm, bit_depth, params);
        if (cuda_err == HdrError::Ok) return cuda_err;
    }

    HdrFunc func = find_hdr_func(algorithm);
    if (!func) return HdrError::InternalError;

    return func(input, output, width, height, channels, bit_depth, params);
}

} // namespace hdr
