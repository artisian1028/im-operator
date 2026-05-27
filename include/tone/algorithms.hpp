#ifndef TONE_ALGORITHMS_HPP
#define TONE_ALGORITHMS_HPP

#include "types.hpp"
#include <cstdint>
#include <string>

namespace tone {

// Metadata helpers
std::string algorithm_name(ToneAlgorithm algo);
int algorithm_window_size(ToneAlgorithm algo);

// Input validation
ToneError validate_tone_inputs(const uint8_t* input, uint8_t* output,
                                int width, int height, int channels,
                                int bit_depth);

// Main dispatch: input RGB (channels=3) -> output RGB (channels=3)
ToneError process_tone(const uint8_t* input, uint8_t* output,
                        int width, int height, int channels,
                        ToneAlgorithm algorithm, int bit_depth = 8,
                        const ToneParams& params = {});

// Individual algorithm functions
ToneError process_gamma(const uint8_t* input, uint8_t* output,
                         int width, int height, int channels,
                         int bit_depth, const ToneParams& params);

ToneError process_s_curve(const uint8_t* input, uint8_t* output,
                           int width, int height, int channels,
                           int bit_depth, const ToneParams& params);

ToneError process_levels(const uint8_t* input, uint8_t* output,
                          int width, int height, int channels,
                          int bit_depth, const ToneParams& params);

ToneError process_curves_3point(const uint8_t* input, uint8_t* output,
                                 int width, int height, int channels,
                                 int bit_depth, const ToneParams& params);

ToneError process_shadows_highlights(const uint8_t* input, uint8_t* output,
                                      int width, int height, int channels,
                                      int bit_depth, const ToneParams& params);

} // namespace tone

#endif // TONE_ALGORITHMS_HPP
