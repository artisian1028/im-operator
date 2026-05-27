#ifndef DENOISE_ALGORITHMS_HPP
#define DENOISE_ALGORITHMS_HPP

#include "types.hpp"
#include <cstdint>
#include <string>

namespace denoise {

// Metadata helpers
std::string algorithm_name(DenoiseAlgorithm algo);
int algorithm_window_size(DenoiseAlgorithm algo);

// Input validation shared across algorithms
DenoiseError validate_denoise_inputs(const uint8_t* input, uint8_t* output,
                                     int width, int height, int channels,
                                     int bit_depth);

// Main dispatch: input RGB (channels=3) or grayscale (channels=1) -> output same format
// strength: [0.0, 2.0], 1.0 = default, controls filter intensity
DenoiseError process_denoise(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             DenoiseAlgorithm algorithm, int bit_depth = 8,
                             float strength = 1.0f);

// Individual algorithm functions
DenoiseError process_gaussian(const uint8_t* input, uint8_t* output,
                              int width, int height, int channels,
                              int bit_depth, float strength);

DenoiseError process_median(const uint8_t* input, uint8_t* output,
                            int width, int height, int channels,
                            int bit_depth, float strength);

DenoiseError process_bilateral(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, float strength);

DenoiseError process_nlm(const uint8_t* input, uint8_t* output,
                         int width, int height, int channels,
                         int bit_depth, float strength);

DenoiseError process_bayer_denoise(const uint8_t* input, uint8_t* output,
                                    int width, int height, int channels,
                                    int bit_depth, float strength);

DenoiseError process_wavelet(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int bit_depth, float strength);

} // namespace denoise

#endif // DENOISE_ALGORITHMS_HPP
