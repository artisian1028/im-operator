#include "lens_shading/algorithms.hpp"
#include <string>
#include <array>

namespace lens_shading {

LensShadingError validate_lens_shading_inputs(const uint8_t* data,
                                               int width, int height,
                                               int channels, int bit_depth) {
    if (!data) return LensShadingError::NullInput;
    if (!is_valid_dimensions(width, height)) return LensShadingError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return LensShadingError::InvalidBitDepth;
    if (channels != 1) return LensShadingError::InvalidChannels;
    return LensShadingError::Ok;
}

std::string algorithm_name(LensShadingAlgorithm algo) {
    switch (algo) {
        case LensShadingAlgorithm::POLYNOMIAL: return "Polynomial Lens Shading Correction";
        case LensShadingAlgorithm::FLAT_FIELD: return "Flat-Field Lens Shading Correction";
        default: return "Unknown";
    }
}

int algorithm_window_size(LensShadingAlgorithm) { return 0; }

using LFunc = LensShadingError(*)(uint8_t*, int, int, BayerPattern, int, const LensShadingParams&);
struct Entry { LensShadingAlgorithm algo; LFunc func; };

static const std::array<Entry, 2> kRegistry = {{
    {LensShadingAlgorithm::POLYNOMIAL, process_polynomial},
    {LensShadingAlgorithm::FLAT_FIELD, process_flat_field}
}};
static_assert(kRegistry.size() == 2, "Registry must match enum count");

static LFunc find(LensShadingAlgorithm algo) {
    for (const auto& e : kRegistry) if (e.algo == algo) return e.func;
    return nullptr;
}

LensShadingError process_lens_shading(uint8_t* data,
                                       int width, int height,
                                       BayerPattern pattern,
                                       LensShadingAlgorithm algorithm,
                                       int bit_depth,
                                       const LensShadingParams& params) {
    auto err = validate_lens_shading_inputs(data, width, height, 1, bit_depth);
    if (err != LensShadingError::Ok) return err;

    if (has_cuda()) {
        LensShadingError cuda_err = process_lens_shading_cuda(data, width, height,
                                                               pattern, algorithm,
                                                               bit_depth, params);
        if (cuda_err == LensShadingError::Ok) return cuda_err;
    }

    auto f = find(algorithm);
    if (!f) return LensShadingError::InternalError;
    return f(data, width, height, pattern, bit_depth, params);
}

} // namespace lens_shading
