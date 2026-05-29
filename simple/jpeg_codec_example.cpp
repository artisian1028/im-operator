// jpeg_codec_example.cpp
// Demonstrates JPEG encoding (RGB -> JPEG) and decoding (JPEG -> RGB)
// Compile: linked with im_operator library
// Requires: test_data/rgb_640x480_8bit.raw (921600 bytes)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include "jpeg_codec.h"

#define WIDTH  640
#define HEIGHT 480
#define CH     3
#define PIXELS (WIDTH * HEIGHT * CH)

static bool load_raw(const char* filename, std::vector<uint8_t>& buf, size_t expected) {
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", filename); return false; }
    buf.resize(expected);
    size_t rd = fread(buf.data(), 1, expected, f);
    fclose(f);
    if (rd != expected) { fprintf(stderr, "ERROR: short read from %s\n", filename); return false; }
    return true;
}

// Compute PSNR between two 8-bit RGB images
static double compute_psnr(const uint8_t* a, const uint8_t* b, size_t n) {
    double mse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
    }
    mse /= (double)n;
    if (mse < 1e-12) return 999.0;
    return 10.0 * log10(255.0 * 255.0 / mse);
}

static size_t read_file(const char* filename, std::vector<uint8_t>& buf) {
    FILE* f = fopen(filename, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf.resize(sz);
    fread(buf.data(), 1, sz, f);
    fclose(f);
    return (size_t)sz;
}

static void test_quality(const std::vector<uint8_t>& src, int quality, const char* jpeg_name, const char* decoded_name) {
    using namespace jpeg_codec;

    size_t max_jpeg = get_max_jpeg_size(WIDTH, HEIGHT, CH);
    std::vector<uint8_t> jpeg_buf(max_jpeg);
    size_t jpeg_size = max_jpeg;

    JpegParams params;
    params.quality = quality;

    printf("\n--- Quality = %d ---\n", quality);

    // Encode
    JpegError err = process_jpeg_encode(src.data(), jpeg_buf.data(), &jpeg_size,
                                         WIDTH, HEIGHT, CH,
                                         JpegAlgorithm::ENCODE_BASELINE, 8, params);
    if (err != JpegError::Ok) {
        printf("  Encode FAIL: %s\n", jpeg_error_message(err));
        return;
    }
    printf("  Encoded size: %zu bytes (%.1fx compression ratio)\n",
           jpeg_size, (double)PIXELS / (double)jpeg_size);

    // Save JPEG to file
    FILE* fj = fopen(jpeg_name, "wb");
    if (fj) { fwrite(jpeg_buf.data(), 1, jpeg_size, fj); fclose(fj); }

    // Decode
    int dw = 0, dh = 0, dch = 0;
    std::vector<uint8_t> decoded(PIXELS);

    err = process_jpeg_decode(jpeg_buf.data(), jpeg_size, decoded.data(),
                               &dw, &dh, &dch, JpegAlgorithm::DECODE_BASELINE);
    if (err != JpegError::Ok) {
        printf("  Decode FAIL: %s\n", jpeg_error_message(err));
        return;
    }
    printf("  Decoded: %dx%d, %d channels\n", dw, dh, dch);

    // Save decoded
    FILE* fd = fopen(decoded_name, "wb");
    if (fd) { fwrite(decoded.data(), 1, PIXELS, fd); fclose(fd); }

    // Compare
    if (dw == WIDTH && dh == HEIGHT && dch == CH) {
        double psnr = compute_psnr(src.data(), decoded.data(), PIXELS);
        printf("  PSNR: %.2f dB\n", psnr);
    } else {
        printf("  Dimension mismatch, cannot compute PSNR\n");
    }
}

int main() {
    printf("=== JPEG Codec Example ===\n\n");

    std::vector<uint8_t> src;
    if (!load_raw("test_data/rgb_640x480_8bit.raw", src, PIXELS)) return 1;
    printf("Loaded rgb_640x480_8bit.raw (%zu bytes)\n", src.size());
    printf("Image: %dx%d, %d channels\n", WIDTH, HEIGHT, CH);

    // Test different quality levels
    test_quality(src, 90,  "output_q90.jpg", "decoded_q90.raw");
    test_quality(src, 50,  "output_q50.jpg", "decoded_q50.raw");
    test_quality(src, 10,  "output_q10.jpg", "decoded_q10.raw");

    // Summary of file sizes
    printf("\n=== File Size Summary ===\n");
    printf("  Original (raw):    %zu bytes\n", src.size());

    for (int q : {10, 50, 90}) {
        char name[32];
        snprintf(name, sizeof(name), "output_q%d.jpg", q);
        std::vector<uint8_t> buf;
        size_t sz = read_file(name, buf);
        if (sz > 0)
            printf("  Quality %3d JPEG:    %zu bytes\n", q, sz);
    }

    printf("\nDone.\n");
    return 0;
}
