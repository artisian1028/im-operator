#include "defect_correct/algorithms.hpp"
#include <string>
#include <array>

namespace defect_correct {

DefectCorrectError validate_defect_correct_inputs(const uint8_t* data,
                                                   int width, int height,
                                                   int channels, int bit_depth) {
    if (!data) return DefectCorrectError::NullInput;
    if (!is_valid_dimensions(width, height)) return DefectCorrectError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return DefectCorrectError::InvalidBitDepth;
    if (channels != 1) return DefectCorrectError::InvalidChannels;
    return DefectCorrectError::Ok;
}

std::string algorithm_name(DefectCorrectAlgorithm algo) {
    switch (algo) {
        case DefectCorrectAlgorithm::ADAPTIVE: return "Adaptive Defect Correction";
        case DefectCorrectAlgorithm::MAP_BASED: return "Map-Based Defect Correction";
        default: return "Unknown";
    }
}

int algorithm_window_size(DefectCorrectAlgorithm algo) {
    return (algo == DefectCorrectAlgorithm::ADAPTIVE) ? 3 : 0;
}

using DefectFunc = DefectCorrectError(*)(uint8_t*, int, int, BayerPattern, int, const DefectCorrectParams&);

struct Entry { DefectCorrectAlgorithm algo; DefectFunc func; };

static const std::array<Entry, 2> kRegistry = {{
    {DefectCorrectAlgorithm::ADAPTIVE,  process_adaptive},
    {DefectCorrectAlgorithm::MAP_BASED, process_map_based}
}};
static_assert(kRegistry.size() == 2, "Registry must match enum count");

static DefectFunc find(DefectCorrectAlgorithm algo) {
    for (const auto& e : kRegistry) if (e.algo == algo) return e.func;
    return nullptr;
}

DefectCorrectError process_defect_correct(uint8_t* data,
                                           int width, int height,
                                           BayerPattern pattern,
                                           DefectCorrectAlgorithm algorithm,
                                           int bit_depth,
                                           const DefectCorrectParams& params) {
    auto err = validate_defect_correct_inputs(data, width, height, 1, bit_depth);
    if (err != DefectCorrectError::Ok) return err;

    if (has_cuda()) {
        DefectCorrectError cuda_err = process_defect_correct_cuda(data, width, height,
                                                                    pattern, algorithm,
                                                                    bit_depth, params);
        if (cuda_err == DefectCorrectError::Ok) return cuda_err;
    }

    auto f = find(algorithm);
    if (!f) return DefectCorrectError::InternalError;
    return f(data, width, height, pattern, bit_depth, params);
}

} // namespace defect_correct
