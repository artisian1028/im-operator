#ifndef COLOR_TEMP_TYPES_HPP
#define COLOR_TEMP_TYPES_HPP

#include <cstdint>

namespace color_temp {

enum class ColorTempAlgorithm {
    KELVIN,         // Kelvin-based RGB adjustment (blackbody radiator model)
    PRESET,         // Named standard illuminant presets
    MANUAL,         // User-supplied RGB multipliers
    WHITE_BALANCE   // Auto white-balance via gray-world on the input
};

enum class ColorTempError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ImageTooSmall,
    InternalError
};

inline const char* color_temp_error_message(ColorTempError err) {
    switch (err) {
        case ColorTempError::Ok:               return "Success";
        case ColorTempError::NullInput:        return "Null input/output pointer";
        case ColorTempError::InvalidDimensions: return "Invalid image dimensions";
        case ColorTempError::InvalidBitDepth:   return "Invalid bit depth (must be 1-16)";
        case ColorTempError::InvalidChannels:   return "Invalid channel count (must be 3)";
        case ColorTempError::ImageTooSmall:     return "Image too small for algorithm";
        case ColorTempError::InternalError:    return "Internal processing error";
        default:                                return "Unknown error";
    }
}

inline bool operator!(ColorTempError err) { return err != ColorTempError::Ok; }
inline bool ok(ColorTempError err) { return err == ColorTempError::Ok; }

inline bool is_valid_bit_depth(int bit_depth) {
    return bit_depth >= 1 && bit_depth <= 16;
}
inline bool is_valid_dimensions(int width, int height) {
    return width > 0 && height > 0;
}

// Named standard illuminant presets
enum class IlluminantPreset {
    CANDLE,           // ~1850K
    TUNGSTEN_40W,     // ~2600K
    TUNGSTEN_100W,    // ~2850K
    HALOGEN,          // ~3200K
    WARM_FLUORESCENT, // ~3500K
    COOL_WHITE_FLUO,  // ~4200K
    MIDDAY_SUN,       // ~5500K
    CLOUDY,           // ~6500K (D65)
    SHADE,            // ~7500K
    OVERCAST,         // ~8000K
    BLUE_SKY          // ~10000K
};

} // namespace color_temp

#endif // COLOR_TEMP_TYPES_HPP
