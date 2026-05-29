#include "color_temp/algorithms.hpp"
#include <string>
#include <array>
#include <cmath>

namespace color_temp {

// --- Input validation ---

ColorTempError validate_color_temp_inputs(const uint8_t* input, uint8_t* output,
                                            int width, int height, int channels,
                                            int bit_depth) {
    if (!input || !output) return ColorTempError::NullInput;
    if (!is_valid_dimensions(width, height)) return ColorTempError::InvalidDimensions;
    if (!is_valid_bit_depth(bit_depth)) return ColorTempError::InvalidBitDepth;
    if (channels != 3) return ColorTempError::InvalidChannels;
    return ColorTempError::Ok;
}

// --- Metadata ---

std::string algorithm_name(ColorTempAlgorithm algo) {
    switch (algo) {
        case ColorTempAlgorithm::KELVIN:         return "Kelvin Temperature";
        case ColorTempAlgorithm::PRESET:         return "Illuminant Preset";
        case ColorTempAlgorithm::MANUAL:         return "Manual RGB Multipliers";
        case ColorTempAlgorithm::WHITE_BALANCE:  return "Auto White Balance";
        default:                                  return "Unknown";
    }
}

int algorithm_window_size(ColorTempAlgorithm algo) {
    (void)algo;
    return 1;
}

// --- Registry ---

using CTFunc = ColorTempError(*)(const uint8_t*, uint8_t*, int, int, int, int,
                                   int, IlluminantPreset, float, float);

struct AlgorithmEntry {
    ColorTempAlgorithm algorithm;
    CTFunc func;
};

static const std::array<AlgorithmEntry, 4> kColorTempRegistry = {{
    {ColorTempAlgorithm::KELVIN,         process_kelvin},
    {ColorTempAlgorithm::PRESET,         process_preset},
    {ColorTempAlgorithm::MANUAL,         process_manual_temp},
    {ColorTempAlgorithm::WHITE_BALANCE,  process_white_balance_auto}
}};

static_assert(kColorTempRegistry.size() == 4,
              "kColorTempRegistry size must match ColorTempAlgorithm enum count");

static CTFunc find_ct_func(ColorTempAlgorithm algorithm) {
    for (const auto& entry : kColorTempRegistry) {
        if (entry.algorithm == algorithm) return entry.func;
    }
    return nullptr;
}

// --- Main dispatch ---

ColorTempError process_color_temp(const uint8_t* input, uint8_t* output,
                                    int width, int height, int channels,
                                    ColorTempAlgorithm algorithm,
                                    int bit_depth, int kelvin,
                                    IlluminantPreset preset,
                                    float r_gain, float b_gain) {
    ColorTempError err = validate_color_temp_inputs(input, output, width, height,
                                                      channels, bit_depth);
    if (err != ColorTempError::Ok) return err;

    if (has_cuda()) {
        float rm = 1.0f, gm = 1.0f, bm = 1.0f;
        if (algorithm == ColorTempAlgorithm::MANUAL) {
            rm = r_gain; bm = b_gain;
        } else if (algorithm == ColorTempAlgorithm::KELVIN) {
            kelvin_to_rgb_multipliers(kelvin, rm, bm);
        } else if (algorithm == ColorTempAlgorithm::PRESET) {
            illuminant_to_rgb_multipliers(preset, rm, bm);
        }
        if (algorithm != ColorTempAlgorithm::WHITE_BALANCE) {
            ColorTempError cuda_err = process_color_temp_cuda(input, output, width, height,
                                                               channels, bit_depth, rm, gm, bm);
            if (cuda_err == ColorTempError::Ok) return cuda_err;
        }
    }

    CTFunc func = find_ct_func(algorithm);
    if (!func) {
        return ColorTempError::InternalError;
    }

    return func(input, output, width, height, channels, bit_depth,
                kelvin, preset, r_gain, b_gain);
}

} // namespace color_temp
