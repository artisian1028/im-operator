#include "imop/algorithms.hpp"
#include "imop/pixel_utils.hpp"
#include <string>
#include <thread>
#include <array>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace imop {

DemosaicError validate_demosaic_inputs(const uint8_t* bayer_data, uint8_t* rgb_data,
                                     int width, int height, int bit_depth) {
    if (!bayer_data || !rgb_data) return DemosaicError::NullInput;
    if (!is_valid_dimensions(width, height)) return DemosaicError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return DemosaicError::InvalidBitDepth;
    return DemosaicError::Ok;
}

std::string algorithm_name(DemosaicAlgorithm algo) {
    switch (algo) {
        case DemosaicAlgorithm::SUPER_FAST: return "SUPER_FAST (Nearest)";
        case DemosaicAlgorithm::HQLI: return "HQLI (5x5)";
        case DemosaicAlgorithm::MG:   return "MG (Malvar-He-Cutler 5x5)";
        case DemosaicAlgorithm::L7:   return "L7 (7x7)";
        case DemosaicAlgorithm::DFPD: return "DFPD (11x11)";
        case DemosaicAlgorithm::AHD:  return "AHD (Adaptive Homogeneity-Directed)";
        case DemosaicAlgorithm::AMAZE: return "AMAZE (Adaptive Gradient)";
        case DemosaicAlgorithm::RCD:  return "RCD (Ratio Corrected 9x9)";
        case DemosaicAlgorithm::PRISM: return "PRISM (Polar Ratio Interpolation Spectral Merging)";
        default: return "Unknown";
    }
}

int algorithm_window_size(DemosaicAlgorithm algo) {
    switch (algo) {
        case DemosaicAlgorithm::SUPER_FAST: return 1;
        case DemosaicAlgorithm::HQLI: return 5;
        case DemosaicAlgorithm::MG:   return 5;
        case DemosaicAlgorithm::L7:   return 7;
        case DemosaicAlgorithm::DFPD: return 11;
        case DemosaicAlgorithm::AHD:  return 5;
        case DemosaicAlgorithm::AMAZE: return 5;
        case DemosaicAlgorithm::RCD:  return 9;
        case DemosaicAlgorithm::PRISM: return 9;
        default: return 5;
    }
}

int compute_hardware_concurrency() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return static_cast<int>(std::thread::hardware_concurrency());
#endif
}

using DemosaicFunc = DemosaicError(*)(const uint8_t*, uint8_t*, int, int, BayerPattern, int, bool);

struct AlgorithmEntry {
    DemosaicAlgorithm algorithm;
    DemosaicFunc func;
};

static const std::array<AlgorithmEntry, 9> kDemosaicRegistry = {{
    {DemosaicAlgorithm::SUPER_FAST, process_super_fast},
    {DemosaicAlgorithm::HQLI,       process_hqli},
    {DemosaicAlgorithm::MG,         process_mg},
    {DemosaicAlgorithm::L7,         process_l7},
    {DemosaicAlgorithm::DFPD,       process_dfpd},
    {DemosaicAlgorithm::AHD,        process_ahd},
    {DemosaicAlgorithm::AMAZE,      process_amaze},
    {DemosaicAlgorithm::RCD,        process_rcd},
    {DemosaicAlgorithm::PRISM,      process_prism}
}};

static_assert(kDemosaicRegistry.size() == 9,
              "kDemosaicRegistry size must match DemosaicAlgorithm enum count");

static DemosaicFunc find_demosaic_func(DemosaicAlgorithm algorithm) {
    for (const auto& entry : kDemosaicRegistry) {
        if (entry.algorithm == algorithm) return entry.func;
    }
    return nullptr;
}

DemosaicError demosaic(const uint8_t* bayer_data, uint8_t* rgb_data,
                             int width, int height, BayerPattern pattern,
                             DemosaicAlgorithm algorithm, int bit_depth, bool is_packed) {
    DemosaicError err = validate_demosaic_inputs(bayer_data, rgb_data, width, height, bit_depth);
    if (err != DemosaicError::Ok) return err;
    if (!is_valid_bayer_pattern(pattern)) return DemosaicError::InvalidPattern;

    // Try GPU acceleration first; transparent fallback to CPU on any failure
    // (including unsupported formats like packed data, InternalError, etc).
    if (has_cuda()) {
        DemosaicError cuda_err = demosaic_cuda(bayer_data, rgb_data, width, height,
                                                      pattern, algorithm, bit_depth, is_packed);
        if (cuda_err == DemosaicError::Ok) {
            return cuda_err;
        }
    }

    DemosaicFunc func = find_demosaic_func(algorithm);
    if (!func) {
        return DemosaicError::InternalError;
    }

    DemosaicError result = func(bayer_data, rgb_data, width, height, pattern, bit_depth, is_packed);
    return result;
}

DemosaicError demosaic_cpu(const uint8_t* bayer_data, uint8_t* rgb_data,
                                 int width, int height, BayerPattern pattern,
                                 DemosaicAlgorithm algorithm, int bit_depth, bool is_packed) {
    DemosaicError err = validate_demosaic_inputs(bayer_data, rgb_data, width, height, bit_depth);
    if (err != DemosaicError::Ok) return err;
    if (!is_valid_bayer_pattern(pattern)) return DemosaicError::InvalidPattern;

    DemosaicFunc func = find_demosaic_func(algorithm);
    if (!func) {
        return DemosaicError::InternalError;
    }

    return func(bayer_data, rgb_data, width, height, pattern, bit_depth, is_packed);
}

} // namespace imop
