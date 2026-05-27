#include "color_temp/algorithms.hpp"
#include <cmath>
#include <algorithm>

namespace color_temp {

// --- Kelvin to RGB multipliers (blackbody radiator approximation) ---
//
// Uses Tanner Helland's approximation to compute the RGB color cast for
// a blackbody radiator at the given Kelvin temperature.
//
// The returned multipliers ARE the color cast: multiply an image by these
// to make it look as if lit by a source at that temperature.
//   Warm (low K): R > 1, B < 1  → warm/orange cast
//   Cool (high K): R < 1, B > 1  → cool/blue cast
//
// Normalized so G = 1.0 for brightness preservation.
void kelvin_to_rgb_multipliers(int kelvin, float& r_mult, float& b_mult) {
    int t = std::max(1000, std::min(40000, kelvin));
    float tk = static_cast<float>(t) / 100.0f;

    float r_raw, g_raw, b_raw;

    // Red
    if (t <= 6600) {
        r_raw = 1.0f;
    } else {
        r_raw = 1.29293618606f * std::pow(tk - 60.0f, -0.1332047592f);
    }

    // Green
    if (t <= 6600) {
        g_raw = 0.390081578769f * std::log(tk) - 0.631841443788f;
    } else {
        g_raw = 1.1298908609f * std::pow(tk - 60.0f, -0.0755148492f);
    }

    // Blue
    if (t >= 6600) {
        b_raw = 1.0f;
    } else if (t <= 1900) {
        b_raw = 0.0f;
    } else {
        b_raw = 0.54320678911f * std::log(tk - 10.0f) - 1.19625408914f;
    }

    // Clamp to valid range
    r_raw = std::max(0.0f, r_raw);
    g_raw = std::max(0.0f, g_raw);
    b_raw = std::max(0.0f, b_raw);

    // Normalize so G = 1.0
    if (g_raw > 0.0f) {
        r_mult = r_raw / g_raw;
        b_mult = b_raw / g_raw;
    } else {
        r_mult = 1.0f;
        b_mult = 1.0f;
    }
}

// --- Illuminant preset to RGB multipliers ---

void illuminant_to_rgb_multipliers(IlluminantPreset preset, float& r_mult, float& b_mult) {
    // Illuminant presets are WHITE BALANCE CORRECTION factors.
    // They neutralize the color cast of a scene shot under that illuminant.
    // For a scene shot under warm light (low K), the correction boosts R and suppresses B.
    // For a scene shot under cool light (high K), the correction suppresses R and boosts B.
    switch (preset) {
        case IlluminantPreset::CANDLE:
            // ~1850K scene: very orange. Correct with strong R-boost / B-cut
            r_mult = 0.55f; b_mult = 3.50f;
            break;
        case IlluminantPreset::TUNGSTEN_40W:
            r_mult = 0.65f; b_mult = 2.60f;
            break;
        case IlluminantPreset::TUNGSTEN_100W:
            r_mult = 0.72f; b_mult = 2.00f;
            break;
        case IlluminantPreset::HALOGEN:
            r_mult = 0.80f; b_mult = 1.65f;
            break;
        case IlluminantPreset::WARM_FLUORESCENT:
            r_mult = 0.85f; b_mult = 1.40f;
            break;
        case IlluminantPreset::COOL_WHITE_FLUO:
            r_mult = 0.92f; b_mult = 1.15f;
            break;
        case IlluminantPreset::MIDDAY_SUN:
            r_mult = 1.0f; b_mult = 1.0f;
            break;
        case IlluminantPreset::CLOUDY:
            r_mult = 1.0f; b_mult = 1.0f;
            break;
        case IlluminantPreset::SHADE:
            // ~7500K scene: blue cast. Correct with R-boost / B-cut
            r_mult = 1.10f; b_mult = 0.85f;
            break;
        case IlluminantPreset::OVERCAST:
            r_mult = 1.18f; b_mult = 0.78f;
            break;
        case IlluminantPreset::BLUE_SKY:
            r_mult = 1.40f; b_mult = 0.60f;
            break;
        default:
            r_mult = 1.0f; b_mult = 1.0f;
            break;
    }
}

int illuminant_kelvin(IlluminantPreset preset) {
    switch (preset) {
        case IlluminantPreset::CANDLE:            return 1850;
        case IlluminantPreset::TUNGSTEN_40W:      return 2600;
        case IlluminantPreset::TUNGSTEN_100W:     return 2850;
        case IlluminantPreset::HALOGEN:           return 3200;
        case IlluminantPreset::WARM_FLUORESCENT:  return 3500;
        case IlluminantPreset::COOL_WHITE_FLUO:   return 4200;
        case IlluminantPreset::MIDDAY_SUN:        return 5500;
        case IlluminantPreset::CLOUDY:            return 6500;
        case IlluminantPreset::SHADE:             return 7500;
        case IlluminantPreset::OVERCAST:          return 8000;
        case IlluminantPreset::BLUE_SKY:          return 10000;
        default:                                   return 6500;
    }
}

const char* illuminant_name(IlluminantPreset preset) {
    switch (preset) {
        case IlluminantPreset::CANDLE:            return "Candle (~1850K)";
        case IlluminantPreset::TUNGSTEN_40W:      return "Tungsten 40W (~2600K)";
        case IlluminantPreset::TUNGSTEN_100W:     return "Tungsten 100W (~2850K)";
        case IlluminantPreset::HALOGEN:           return "Halogen (~3200K)";
        case IlluminantPreset::WARM_FLUORESCENT:  return "Warm Fluorescent (~3500K)";
        case IlluminantPreset::COOL_WHITE_FLUO:   return "Cool White Fluor. (~4200K)";
        case IlluminantPreset::MIDDAY_SUN:        return "Midday Sun (~5500K)";
        case IlluminantPreset::CLOUDY:            return "Cloudy / D65 (~6500K)";
        case IlluminantPreset::SHADE:             return "Shade (~7500K)";
        case IlluminantPreset::OVERCAST:          return "Overcast (~8000K)";
        case IlluminantPreset::BLUE_SKY:          return "Blue Sky (~10000K)";
        default:                                   return "Unknown";
    }
}

} // namespace color_temp
