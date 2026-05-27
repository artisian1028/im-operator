// Byte-level comparison of CPU vs GPU JPEG output
#include "jpeg_codec/algorithms.hpp"
#include <cstdio>
#include <vector>
#include <cstring>

using namespace jpeg_codec;

static void fill_gradient(uint8_t* src, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = ((size_t)y * w + x) * 3;
            src[i]   = (uint8_t)((x * 255) / (w > 1 ? w - 1 : 1));
            src[i+1] = (uint8_t)((y * 255) / (h > 1 ? h - 1 : 1));
            src[i+2] = (uint8_t)(((x + y) * 128) / (w + h > 2 ? w + h - 2 : 1));
        }
    }
}

void test_flat(int w, int h) {
    size_t px = (size_t)w * h * 3;
    std::vector<uint8_t> src(px);
    for (size_t i = 0; i < px; i += 3) { src[i] = 80; src[i+1] = 120; src[i+2] = 200; }
    size_t max_jpg = px * 3 + 65536;
    std::vector<uint8_t> jpg(max_jpg), dec(px);
    size_t js = max_jpg;
    JpegParams p; p.quality = 90; p.chroma_subsample = 0;
    JpegError e = process_jpeg_encode(src.data(), jpg.data(), &js, w, h, 3, JpegAlgorithm::ENCODE_BASELINE, 8, p);
    int ow, oh, oc;
    JpegError de = process_jpeg_decode(jpg.data(), js, dec.data(), &ow, &oh, &oc, JpegAlgorithm::DECODE_BASELINE);
    printf("FLAT %dx%d enc=%s(%zuB) dec=%s\n", w, h, jpeg_error_message(e), js, jpeg_error_message(de));
}

int main() {
    // Test flat image at various sizes
    test_flat(64, 64);
    test_flat(128, 128);
    test_flat(256, 256);

    // Test gradient at multiple sizes
    for (int sz : {64, 128, 256}) {
        int w = sz, h = sz;
        size_t px = (size_t)w * h * 3;
        std::vector<uint8_t> src(px), dec(px);
        fill_gradient(src.data(), w, h);
        size_t max_jpg = px * 3 + 65536;
        std::vector<uint8_t> jpg(max_jpg);
        size_t js = max_jpg;
        JpegParams p; p.quality = 90; p.chroma_subsample = 0;
        JpegError e = process_jpeg_encode(src.data(), jpg.data(), &js, w, h, 3, JpegAlgorithm::ENCODE_BASELINE, 8, p);
        int ow, oh, oc;
        JpegError de = process_jpeg_decode(jpg.data(), js, dec.data(), &ow, &oh, &oc, JpegAlgorithm::DECODE_BASELINE);
        printf("GRAD %dx%d enc=%s(%zuB) dec=%s\n", w, h, jpeg_error_message(e), js, jpeg_error_message(de));
    }

    // Compare CPU vs GPU for 64x64
    int w = 64, h = 64;
    size_t px = (size_t)w * h * 3;
    std::vector<uint8_t> src(px);
    fill_gradient(src.data(), w, h);

    size_t max_jpg = px * 3 + 65536;
    std::vector<uint8_t> cpu_jpg(max_jpg), gpu_jpg(max_jpg);
    size_t cpu_sz = max_jpg, gpu_sz = max_jpg;

    JpegParams p; p.quality = 90; p.chroma_subsample = 0;

    JpegError e;
    e = process_jpeg_encode(src.data(), cpu_jpg.data(), &cpu_sz, w, h, 3,
                             JpegAlgorithm::ENCODE_BASELINE, 8, p);
    printf("CPU: %s (%zu bytes)\n", jpeg_error_message(e), cpu_sz);

    e = process_jpeg_encode(src.data(), gpu_jpg.data(), &gpu_sz, w, h, 3,
                             JpegAlgorithm::ENCODE_CUDA, 8, p);
    printf("GPU: %s (%zu bytes)\n", jpeg_error_message(e), gpu_sz);

    // Compare byte by byte
    size_t min_sz = cpu_sz < gpu_sz ? cpu_sz : gpu_sz;
    int first_diff = -1;
    for (size_t i = 0; i < min_sz; i++) {
        if (cpu_jpg[i] != gpu_jpg[i]) {
            first_diff = (int)i;
            break;
        }
    }

    if (first_diff >= 0) {
        printf("\nFirst difference at byte %d (0x%X):\n", first_diff, first_diff);
        printf("  CPU: ");
        for (int i = first_diff - 4; i <= first_diff + 4; i++)
            printf("%02X ", i >= 0 && i < (int)cpu_sz ? cpu_jpg[i] : 0);
        printf("\n  GPU: ");
        for (int i = first_diff - 4; i <= first_diff + 4; i++)
            printf("%02X ", i >= 0 && i < (int)gpu_sz ? gpu_jpg[i] : 0);
        printf("\n");

        // Show more context around the diff
        printf("\nCPU bytes %d-%d: ", first_diff, first_diff + 31);
        for (int i = first_diff; i < first_diff + 32 && i < (int)cpu_sz; i++)
            printf("%02X ", cpu_jpg[i]);
        printf("\nGPU bytes %d-%d: ", first_diff, first_diff + 31);
        for (int i = first_diff; i < first_diff + 32 && i < (int)gpu_sz; i++)
            printf("%02X ", gpu_jpg[i]);
        printf("\n");
    } else if (cpu_sz != gpu_sz) {
        printf("\nCPU and GPU match for first %zu bytes, but sizes differ (CPU=%zu, GPU=%zu)\n",
               min_sz, cpu_sz, gpu_sz);
        printf("CPU last 8 bytes: ");
        for (size_t i = cpu_sz > 8 ? cpu_sz - 8 : 0; i < cpu_sz; i++)
            printf("%02X ", cpu_jpg[i]);
        printf("\nGPU last 8 bytes: ");
        for (size_t i = gpu_sz > 8 ? gpu_sz - 8 : 0; i < gpu_sz; i++)
            printf("%02X ", gpu_jpg[i]);
        printf("\n");
    } else {
        printf("\nCPU and GPU JPEGs are IDENTICAL (%zu bytes)\n", cpu_sz);
    }

    // Try to decode both
    std::vector<uint8_t> dec(px);
    int ow, oh, oc;
    e = process_jpeg_decode(cpu_jpg.data(), cpu_sz, dec.data(), &ow, &oh, &oc,
                             JpegAlgorithm::DECODE_BASELINE);
    printf("CPU decode: %s (%dx%d)\n", jpeg_error_message(e), ow, oh);

    e = process_jpeg_decode(gpu_jpg.data(), gpu_sz, dec.data(), &ow, &oh, &oc,
                             JpegAlgorithm::DECODE_BASELINE);
    printf("GPU decode: %s (%dx%d)\n", jpeg_error_message(e), ow, oh);

    return 0;
}
