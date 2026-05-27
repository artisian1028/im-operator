#ifndef WHITE_BALANCE_ALGORITHMS_HPP
#define WHITE_BALANCE_ALGORITHMS_HPP

#include "types.hpp"
#include <cstdint>
#include <string>

namespace white_balance {

// Metadata helpers
std::string algorithm_name(WhiteBalanceAlgorithm algo);
int algorithm_window_size(WhiteBalanceAlgorithm algo);

// Input validation shared across algorithms
WhiteBalanceError validate_white_balance_inputs(const uint8_t* input, uint8_t* output,
                                                  int width, int height, int channels,
                                                  int bit_depth);

// Main dispatch: input RGB (channels=3) -> output RGB (channels=3)
// Each algorithm estimates illuminant gains from the input and applies them.
// p: Minkowski norm parameter (used by SHADE_OF_GRAY), default 6.0
// manual_gains: RGB coefficients for MANUAL algorithm
WhiteBalanceError process_white_balance(const uint8_t* input, uint8_t* output,
                                          int width, int height, int channels,
                                          WhiteBalanceAlgorithm algorithm,
                                          int bit_depth = 8,
                                          float p = 6.0f,
                                          const WBCoefficients& manual_gains = {});

// Individual algorithm functions
WhiteBalanceError process_gray_world(const uint8_t* input, uint8_t* output,
                                      int width, int height, int channels,
                                      int bit_depth, float p,
                                      const WBCoefficients& manual_gains);

WhiteBalanceError process_white_patch(const uint8_t* input, uint8_t* output,
                                       int width, int height, int channels,
                                       int bit_depth, float p,
                                       const WBCoefficients& manual_gains);

WhiteBalanceError process_shade_of_gray(const uint8_t* input, uint8_t* output,
                                         int width, int height, int channels,
                                         int bit_depth, float p,
                                         const WBCoefficients& manual_gains);

WhiteBalanceError process_manual_wb(const uint8_t* input, uint8_t* output,
                                     int width, int height, int channels,
                                     int bit_depth, float p,
                                     const WBCoefficients& manual_gains);

// Estimate gains without applying correction (analysis-only)
WBCoefficients compute_white_balance_gains(const uint8_t* input,
                                             int width, int height,
                                             int bit_depth,
                                             WhiteBalanceAlgorithm algorithm,
                                             float p = 6.0f);

} // namespace white_balance

#endif // WHITE_BALANCE_ALGORITHMS_HPP
