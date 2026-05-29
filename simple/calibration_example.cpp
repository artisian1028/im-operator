// calibration_example.cpp
// Demonstrates camera calibration using checkerboard corner detection
// Compile: linked with im_operator library
// Requires: test_data/checkerboard_640x480_8bit.raw (921600 bytes)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include "calibration.h"

#define IMG_W 640
#define IMG_H 480
#define CH    3
#define PIXELS (IMG_W * IMG_H * CH)

static bool load_raw(const char* filename, std::vector<uint8_t>& buf, size_t expected) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", filename);
        fprintf(stderr, "  Make sure to run 'generate_test_data' first.\n");
        return false;
    }
    buf.resize(expected);
    size_t rd = fread(buf.data(), 1, expected, f);
    fclose(f);
    if (rd != expected) { fprintf(stderr, "ERROR: short read from %s\n", filename); return false; }
    return true;
}

int main() {
    printf("=== Camera Calibration Example (Checkerboard Detection) ===\n\n");

    // Load checkerboard test image (640x480 RGB, 8-bit)
    std::vector<uint8_t> image;
    if (!load_raw("test_data/checkerboard_640x480_8bit.raw", image, PIXELS)) return 1;
    printf("Loaded checkerboard_640x480_8bit.raw (%zu bytes)\n", image.size());
    printf("  Dimensions: %dx%d, %d channels, 8-bit\n\n", IMG_W, IMG_H, CH);

    using namespace calibration;

    // ---------------------------------------------------------------
    // 1. Checkerboard corner detection
    // ---------------------------------------------------------------
    printf("--- Checkerboard Corner Detection ---\n");

    CheckerboardConfig config;
    config.cols = 8;              // 8 inner corners horizontally
    config.rows = 6;              // 6 inner corners vertically
    config.square_size = 30.0;    // 30mm per square
    config.sub_pixel = true;      // enable sub-pixel refinement

    printf("Input config:\n");
    printf("  cols=%d rows=%d square_size=%.1fmm sub_pixel=%s\n",
           config.cols, config.rows, config.square_size,
           config.sub_pixel ? "true" : "false");

    // Allocate corner storage (rows * cols points)
    CheckerboardCorners corners;
    std::vector<Point2D> corner_points(config.rows * config.cols);
    corners.points = corner_points.data();
    corners.rows = config.rows;
    corners.cols = config.cols;
    corners.valid = false;

    CalibrationError err = process_checkerboard_detect(
        image.data(),
        IMG_W, IMG_H,   // width, height
        CH,             // channels (3 for RGB)
        8,              // bit depth
        &config,
        &corners
    );

    if (err == CalibrationError::Ok) {
        printf("Detection result: SUCCESS\n");
        printf("  Valid: %s\n", corners.valid ? "yes" : "no");
        printf("  Corner count: %d x %d = %d corners\n",
               corners.rows, corners.cols, corners.rows * corners.cols);

        // Print first and last corner positions
        if (corners.valid && corners.points) {
            int total = corners.rows * corners.cols;
            printf("  First corner (0,0): (%.3f, %.3f)\n",
                   corners.points[0].x, corners.points[0].y);
            if (total > 1) {
                printf("  Last corner (%d,%d): (%.3f, %.3f)\n",
                       corners.rows - 1, corners.cols - 1,
                       corners.points[total - 1].x, corners.points[total - 1].y);
            }

            // Print a few sample corners
            printf("  Sample corners:\n");
            for (int r = 0; r < corners.rows && r < 3; ++r) {
                for (int c = 0; c < corners.cols && c < 3; ++c) {
                    int idx = r * corners.cols + c;
                    printf("    (%d,%d) -> (%.3f, %.3f)\n",
                           r, c,
                           corners.points[idx].x, corners.points[idx].y);
                }
            }
            if (corners.rows > 3) printf("    ...\n");
        }
    } else {
        printf("Detection result: FAIL\n");
        printf("  Error: %s\n", calibration_error_message(err));
        printf("  Note: The generated checkerboard may not have enough corners.\n");
    }

    // ---------------------------------------------------------------
    // 2. Demo validation: test with invalid inputs
    // ---------------------------------------------------------------
    printf("\n--- Validation with Invalid Inputs ---\n");

    // Test null image pointer
    {
        CheckerboardConfig cfg2;
        cfg2.cols = 8;
        cfg2.rows = 6;

        CalibrationError err2 = process_checkerboard_detect(
            nullptr, IMG_W, IMG_H, CH, 8, &cfg2, &corners
        );
        printf("  Null image pointer: ");
        if (err2 == CalibrationError::NullInput)
            printf("correctly returns NullInput\n");
        else
            printf("returned %d (%s)\n", (int)err2, calibration_error_message(err2));
    }

    // Test null config pointer
    {
        CalibrationError err3 = process_checkerboard_detect(
            image.data(), IMG_W, IMG_H, CH, 8, nullptr, &corners
        );
        printf("  Null config pointer: ");
        if (err3 != CalibrationError::Ok)
            printf("returns error (%s)\n", calibration_error_message(err3));
        else
            printf("unexpectedly returned Ok\n");
    }

    // Test invalid dimensions (zero width)
    {
        CheckerboardConfig cfg4;
        cfg4.cols = 8;
        cfg4.rows = 6;
        CalibrationError err4 = process_checkerboard_detect(
            image.data(), 0, IMG_H, CH, 8, &cfg4, &corners
        );
        printf("  Zero width (0x480): ");
        if (err4 != CalibrationError::Ok)
            printf("returns error (%s)\n", calibration_error_message(err4));
        else
            printf("unexpectedly returned Ok\n");
    }

    // ---------------------------------------------------------------
    // 3. Checkerboard calibration (Zhang's method)
    // ---------------------------------------------------------------
    printf("\n--- Checkerboard Calibration (Zhang's Method) ---\n");

    // Re-run detection to get fresh corners for calibration
    CheckerboardCorners cal_corners;
    std::vector<Point2D> cal_corner_points(config.rows * config.cols);
    cal_corners.points = cal_corner_points.data();
    cal_corners.rows = config.rows;
    cal_corners.cols = config.cols;
    cal_corners.valid = false;

    CalibrationError detect_err = process_checkerboard_detect(
        image.data(), IMG_W, IMG_H, CH, 8, &config, &cal_corners
    );

    if (ok(detect_err) && cal_corners.valid) {
        // Run Zhang calibration with a single view (at least 1 view needed for intrinsics)
        CameraIntrinsics intrinsics;
        CameraExtrinsics extrinsics;

        CalibrationError cal_err = process_checkerboard_calibrate(
            &cal_corners,           // corners for this view
            1,                      // view_count
            IMG_W, IMG_H,           // image dimensions
            &config,                // checkerboard config (square_size etc.)
            &intrinsics,
            &extrinsics,
            true                    // estimate distortion
        );

        if (ok(cal_err)) {
            printf("Calibration result: SUCCESS\n");
            printf("  Intrinsics:\n");
            printf("    fx = %.3f\n", intrinsics.fx);
            printf("    fy = %.3f\n", intrinsics.fy);
            printf("    cx = %.3f\n", intrinsics.cx);
            printf("    cy = %.3f\n", intrinsics.cy);
            printf("    k1 = %.6f\n", intrinsics.k1);
            printf("    k2 = %.6f\n", intrinsics.k2);
            printf("    p1 = %.6f\n", intrinsics.p1);
            printf("    p2 = %.6f\n", intrinsics.p2);
            printf("  Rotation (row-major):\n");
            printf("    [%.6f, %.6f, %.6f]\n",
                   extrinsics.rotation[0], extrinsics.rotation[1], extrinsics.rotation[2]);
            printf("    [%.6f, %.6f, %.6f]\n",
                   extrinsics.rotation[3], extrinsics.rotation[4], extrinsics.rotation[5]);
            printf("    [%.6f, %.6f, %.6f]\n",
                   extrinsics.rotation[6], extrinsics.rotation[7], extrinsics.rotation[8]);
            printf("  Translation: [%.3f, %.3f, %.3f]\n",
                   extrinsics.translation[0], extrinsics.translation[1], extrinsics.translation[2]);
        } else {
            printf("Calibration result: FAIL (%s)\n", calibration_error_message(cal_err));
            printf("  Note: A single view may not be sufficient for full calibration.\n");
        }
    } else {
        printf("Skipping checkerboard calibration: detection failed.\n");
    }

    // ---------------------------------------------------------------
    // 4. DLT (Direct Linear Transform): 3D-2D point correspondences
    // ---------------------------------------------------------------
    printf("\n--- DLT (Direct Linear Transform) ---\n");

    // Create simple manual 3D-2D correspondences (cube corners)
    // We project 8 corners of a 1x1x1 cube to image coordinates using a simple pinhole model
    // fx=fy=500, cx=320, cy=240 for a 640x480 image
    const double fx = 500.0, fy = 500.0, cx = 320.0, cy = 240.0;

    // 3D points (cube corners in world coordinates)
    // 2D points (simple simulated projection with some perspective)
    std::vector<DltCorrespondence> dlt_pts;
    dlt_pts.reserve(8);

    // Generate cube corners at various (X, Y, Z) and project them
    // For each 3D point, project using: u = fx * X/Z + cx, v = fy * Y/Z + cy
    struct { double x, y, z; } world_pts[8] = {
        {0.0, 0.0, 5.0}, {1.0, 0.0, 5.0}, {1.0, 1.0, 5.0}, {0.0, 1.0, 5.0},
        {0.0, 0.0, 6.0}, {1.0, 0.0, 6.0}, {1.0, 1.0, 6.0}, {0.0, 1.0, 6.0}
    };

    for (int i = 0; i < 8; ++i) {
        double X = world_pts[i].x;
        double Y = world_pts[i].y;
        double Z = world_pts[i].z;
        double u = fx * X / Z + cx;
        double v = fy * Y / Z + cy;
        dlt_pts.push_back({X, Y, Z, u, v});
    }

    if ((int)dlt_pts.size() >= 6) {
        DltParams dlt_params;
        dlt_params.correspondences = dlt_pts.data();
        dlt_params.count = (int)dlt_pts.size();

        DltResult dlt_result;

        CalibrationError dlt_err = process_dlt(&dlt_params, &dlt_result);

        if (ok(dlt_err)) {
            printf("DLT result: SUCCESS\n");
            printf("  Camera matrix P (3x4, row-major):\n");
            printf("    [%.4f, %.4f, %.4f, %.4f]\n",
                   dlt_result.P[0], dlt_result.P[1], dlt_result.P[2], dlt_result.P[3]);
            printf("    [%.4f, %.4f, %.4f, %.4f]\n",
                   dlt_result.P[4], dlt_result.P[5], dlt_result.P[6], dlt_result.P[7]);
            printf("    [%.4f, %.4f, %.4f, %.4f]\n",
                   dlt_result.P[8], dlt_result.P[9], dlt_result.P[10], dlt_result.P[11]);
            printf("  Recovered intrinsics:\n");
            printf("    fx = %.3f (expected ~%.1f)\n", dlt_result.K.fx, fx);
            printf("    fy = %.3f (expected ~%.1f)\n", dlt_result.K.fy, fy);
            printf("    cx = %.3f (expected ~%.1f)\n", dlt_result.K.cx, cx);
            printf("    cy = %.3f (expected ~%.1f)\n", dlt_result.K.cy, cy);
            printf("  Residual: %.6f\n", dlt_result.residual);
        } else {
            printf("DLT result: FAIL (%s)\n", calibration_error_message(dlt_err));
        }
    } else {
        printf("Skipping DLT: insufficient correspondences.\n");
    }

    printf("\nDone.\n");
    return 0;
}
