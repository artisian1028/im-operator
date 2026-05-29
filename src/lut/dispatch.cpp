#include "lut/algorithms.hpp"
#include <string>
#include <array>

namespace lut {

// --- Input validation ---

LUTError validate_lut_inputs(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth) {
    if (!input || !output) return LUTError::NullInput;
    if (!is_valid_dimensions(width, height)) return LUTError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return LUTError::InvalidBitDepth;
    if (channels != 3) return LUTError::InvalidChannels;
    return LUTError::Ok;
}

// --- Metadata ---

std::string algorithm_name(LUTAlgorithm algo) {
    switch (algo) {
        case LUTAlgorithm::CUBE_FILE:      return "CUBE File LUT (.cube import)";
        case LUTAlgorithm::CUSTOM_3D:      return "Custom 3D LUT";
        case LUTAlgorithm::SEPIA:          return "Sepia Tone";
        case LUTAlgorithm::COOL:           return "Cool (blue cast)";
        case LUTAlgorithm::WARM:           return "Warm (amber cast)";
        case LUTAlgorithm::HIGH_CONTRAST:  return "High Contrast (S-curve)";
        case LUTAlgorithm::LOW_CONTRAST:   return "Low Contrast (faded)";
        case LUTAlgorithm::INVERT:         return "Color Inversion";
        case LUTAlgorithm::VINTAGE_FADE:   return "Vintage Fade";
        default:                            return "Unknown";
    }
}

int algorithm_window_size(LUTAlgorithm algo) {
    (void)algo;
    return 1;
}

// --- Registry ---

using LUTFunc = LUTError(*)(const uint8_t*, uint8_t*, int, int, int, int, const void*, int);

struct AlgorithmEntry {
    LUTAlgorithm algorithm;
    LUTFunc func;
};

static const std::array<AlgorithmEntry, 9> kLUTRegistry = {{
    {LUTAlgorithm::CUBE_FILE,       process_cube_file},
    {LUTAlgorithm::CUSTOM_3D,       process_custom_3d},
    {LUTAlgorithm::SEPIA,           process_style_sepia},
    {LUTAlgorithm::COOL,            process_style_cool},
    {LUTAlgorithm::WARM,            process_style_warm},
    {LUTAlgorithm::HIGH_CONTRAST,   process_style_high_contrast},
    {LUTAlgorithm::LOW_CONTRAST,    process_style_low_contrast},
    {LUTAlgorithm::INVERT,          process_style_invert},
    {LUTAlgorithm::VINTAGE_FADE,    process_style_vintage_fade}
}};

static_assert(kLUTRegistry.size() == 9,
              "kLUTRegistry size must match LUTAlgorithm enum count");

static LUTFunc find_lut_func(LUTAlgorithm algorithm) {
    for (const auto& entry : kLUTRegistry) {
        if (entry.algorithm == algorithm) return entry.func;
    }
    return nullptr;
}

// --- Main dispatch ---

LUTError process_lut(const uint8_t* input, uint8_t* output,
                      int width, int height, int channels,
                      LUTAlgorithm algorithm, int bit_depth,
                      const void* lut_data, int lut_size) {
    LUTError err = validate_lut_inputs(input, output, width, height,
                                        channels, bit_depth);
    if (err != LUTError::Ok) return err;

    // GPU path only supports CUSTOM_3D (lut_data is a LUT3D*).
    // CUBE_FILE passes a const char* filepath — casting to LUT3D* is UB.
    if (has_cuda() && algorithm == LUTAlgorithm::CUSTOM_3D) {
        LUTError cuda_err = process_lut_cuda(input, output, width, height, channels,
                                              bit_depth, *static_cast<const LUT3D*>(lut_data));
        if (cuda_err == LUTError::Ok) return cuda_err;
    }

    LUTFunc func = find_lut_func(algorithm);
    if (!func) {
        return LUTError::InternalError;
    }

    return func(input, output, width, height, channels, bit_depth, lut_data, lut_size);
}

} // namespace lut
