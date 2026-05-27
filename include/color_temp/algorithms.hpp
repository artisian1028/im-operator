#ifndef COLOR_TEMP_ALGORITHMS_HPP
#define COLOR_TEMP_ALGORITHMS_HPP

#include "types.hpp"
#include <cstdint>
#include <string>

namespace color_temp {

// Metadata helpers
std::string algorithm_name(ColorTempAlgorithm algo);
int algorithm_window_size(ColorTempAlgorithm algo);

// Input validation
ColorTempError validate_color_temp_inputs(const uint8_t* input, uint8_t* output,
                                            int width, int height, int channels,
                                            int bit_depth);

// Main dispatch: input RGB (channels=3) -> output RGB (channels=3)
// kelvin:    color temperature in Kelvin (1000-40000), used by KELVIN algorithm
// preset:    named illuminant, used by PRESET algorithm
// r_gain/b_gain: manual multipliers, used by MANUAL algorithm (G=1.0)
ColorTempError process_color_temp(const uint8_t* input, uint8_t* output,
                                    int width, int height, int channels,
                                    ColorTempAlgorithm algorithm,
                                    int bit_depth = 8,
                                    int kelvin = 6500,
                                    IlluminantPreset preset = IlluminantPreset::CLOUDY,
                                    float r_gain = 1.0f,
                                    float b_gain = 1.0f);

// Individual algorithm functions
// All take kelvin+preset+r_gain+b_gain for dispatch flexibility
ColorTempError process_kelvin(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, int kelvin,
                               IlluminantPreset preset,
                               float r_gain, float b_gain);

ColorTempError process_preset(const uint8_t* input, uint8_t* output,
                               int width, int height, int channels,
                               int bit_depth, int kelvin,
                               IlluminantPreset preset,
                               float r_gain, float b_gain);

ColorTempError process_manual_temp(const uint8_t* input, uint8_t* output,
                                    int width, int height, int channels,
                                    int bit_depth, int kelvin,
                                    IlluminantPreset preset,
                                    float r_gain, float b_gain);

ColorTempError process_white_balance_auto(const uint8_t* input, uint8_t* output,
                                            int width, int height, int channels,
                                            int bit_depth, int kelvin,
                                            IlluminantPreset preset,
                                            float r_gain, float b_gain);

// --- Utility functions ---

// Convert Kelvin to linear RGB multipliers (blackbody radiator approximation)
// Returns (r_mult, g_mult, b_mult) where g_mult is always 1.0
void kelvin_to_rgb_multipliers(int kelvin, float& r_mult, float& b_mult);

// Get RGB multipliers for a named illuminant preset
// Returns (r_mult, g_mult=1.0, b_mult)
void illuminant_to_rgb_multipliers(IlluminantPreset preset, float& r_mult, float& b_mult);

// Get the nominal Kelvin value for a named illuminant
int illuminant_kelvin(IlluminantPreset preset);

// Get the display name for a named illuminant
const char* illuminant_name(IlluminantPreset preset);

} // namespace color_temp

#endif // COLOR_TEMP_ALGORITHMS_HPP
