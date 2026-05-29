#ifndef SATURATION_ALGORITHMS_HPP
#define SATURATION_ALGORITHMS_HPP

#include "types.hpp"
#include <cstdint>
#include <string>

namespace saturation {

// Metadata helpers
std::string algorithm_name(SaturationAlgorithm algo);
int algorithm_window_size(SaturationAlgorithm algo);

// Input validation
SaturationError validate_saturation_inputs(const uint8_t* input, uint8_t* output,
                                            int width, int height, int channels,
                                            int bit_depth);

// Main dispatch: input RGB (channels=3) -> output RGB (channels=3)
SaturationError process_saturation(const uint8_t* input, uint8_t* output,
                                    int width, int height, int channels,
                                    SaturationAlgorithm algorithm,
                                    int bit_depth = 8,
                                    const SaturationParams& params = {});

// Individual algorithm functions
SaturationError process_hsl(const uint8_t* input, uint8_t* output,
                             int width, int height, int channels,
                             int bit_depth, const SaturationParams& params);

SaturationError process_vibrance(const uint8_t* input, uint8_t* output,
                                  int width, int height, int channels,
                                  int bit_depth, const SaturationParams& params);

SaturationError process_channel_mixer(const uint8_t* input, uint8_t* output,
                                       int width, int height, int channels,
                                       int bit_depth, const SaturationParams& params);

SaturationError process_selective(const uint8_t* input, uint8_t* output,
                                   int width, int height, int channels,
                                   int bit_depth, const SaturationParams& params);

// CUDA support
bool has_cuda();
SaturationError process_saturation_cuda(const uint8_t* input, uint8_t* output,
                                         int width, int height, int channels,
                                         int bit_depth, float sat, float vib);

} // namespace saturation

#endif // SATURATION_ALGORITHMS_HPP
