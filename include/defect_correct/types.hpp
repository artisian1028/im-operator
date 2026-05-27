#ifndef DEFECT_CORRECT_TYPES_HPP
#define DEFECT_CORRECT_TYPES_HPP

#include <cstdint>

namespace defect_correct {

enum class DefectCorrectAlgorithm {
    ADAPTIVE,      // Auto-detect and fix by comparing to same-color neighbors
    MAP_BASED      // Fix known defect positions from a defect map
};

enum class DefectCorrectError {
    Ok = 0,
    NullInput,
    InvalidDimensions,
    InvalidBitDepth,
    InvalidChannels,
    ImageTooSmall,
    InternalError
};

inline const char* defect_correct_error_message(DefectCorrectError err) {
    switch (err) {
        case DefectCorrectError::Ok:               return "Success";
        case DefectCorrectError::NullInput:        return "Null input/output pointer";
        case DefectCorrectError::InvalidDimensions: return "Invalid image dimensions";
        case DefectCorrectError::InvalidBitDepth:   return "Invalid bit depth (must be 1-16)";
        case DefectCorrectError::InvalidChannels:   return "Invalid channel count (must be 1 for Bayer)";
        case DefectCorrectError::ImageTooSmall:     return "Image too small for algorithm";
        case DefectCorrectError::InternalError:    return "Internal processing error";
        default:                                    return "Unknown error";
    }
}

inline bool operator!(DefectCorrectError err) { return err != DefectCorrectError::Ok; }
inline bool ok(DefectCorrectError err) { return err == DefectCorrectError::Ok; }
inline bool is_valid_bit_depth(int bd) { return bd >= 1 && bd <= 16; }
inline bool is_valid_dimensions(int w, int h) { return w > 0 && h > 0; }

struct DefectPoint {
    int x, y;
};

struct DefectCorrectParams {
    float threshold = 0.3f;        // adaptive: relative deviation threshold [0.05, 1.0]
    const DefectPoint* map = nullptr; // map_based: defect coordinate array
    int map_count = 0;             // number of defect points
};

} // namespace defect_correct

#endif
