// Real-world JPEG GPU benchmark — non-trivial test images
#include "jpeg_codec/algorithms.hpp"
#include <cstdio>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>

using namespace jpeg_codec;
using Clock = std::chrono::high_resolution_clock;

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

static void fill_checker(uint8_t* src, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            size_t i = ((size_t)y * w + x) * 3;
            uint8_t v = (uint8_t)(((x / 16 + y / 16) & 1) ? 200 : 50);
            src[i] = v; src[i+1] = v; src[i+2] = v;
        }
    }
}

int main() {
    printf("=== GPU JPEG Real-World Throughput Benchmark ===\n");
    printf("GPU: %s\n\n", has_cuda() ? cuda_device_name() : "N/A");
    if (!has_cuda()) { printf("CUDA not available\n"); return 1; }

    struct TestCase { int w, h; const char* name; const char* desc; void (*fill)(uint8_t*,int,int); };
    TestCase tests[] = {
        {1920, 1080, "FHD-grad",  "FHD gradient", fill_gradient},
        {2048, 2048, "2K-grad",   "2K gradient",  fill_gradient},
        {3840, 2160, "4K-grad",   "4K gradient",  fill_gradient},
        {1920, 1080, "FHD-check", "FHD checker",  fill_checker},
        {2048, 2048, "2K-check",  "2K checker",   fill_checker},
        {3840, 2160, "4K-check",  "4K checker",   fill_checker},
    };

    JpegParams p;
    p.quality = 90;
    p.chroma_subsample = 0; // 4:4:4 for maximum stress

    printf("%-12s %8s | %10s %10s %10s | %10s %10s\n",
           "Test", "Raw MB", "Time(ms)", "JPEG KB", "Ratio", "Raw GB/s", "JPEG MB/s");
    printf("%-12s %8s | %10s %10s %10s | %10s %10s\n",
           "----", "------", "--------", "-------", "-----", "--------", "---------");

    for (auto& t : tests) {
        size_t px = (size_t)t.w * t.h * 3;
        std::vector<uint8_t> src(px);
        t.fill(src.data(), t.w, t.h);

        // Allocate large output buffer (3x raw for safety)
        size_t max_jpg = px * 3 + 65536;
        std::vector<uint8_t> jpg(max_jpg);
        std::vector<uint8_t> dec(px);

        // CPU encode for reference
        size_t cpu_js = max_jpg;
        std::vector<uint8_t> cpu_jpg(max_jpg);
        JpegError cpu_e = process_jpeg_encode(src.data(), cpu_jpg.data(), &cpu_js,
                                               t.w, t.h, 3,
                                               JpegAlgorithm::ENCODE_BASELINE, 8, p);

        // Warmup GPU
        size_t js = max_jpg;
        JpegError e = process_jpeg_encode(src.data(), jpg.data(), &js,
                                           t.w, t.h, 3,
                                           JpegAlgorithm::ENCODE_CUDA, 8, p);
        if (e != JpegError::Ok) {
            printf("%-12s WARMUP FAILED: %s\n", t.name, jpeg_error_message(e));
            continue;
        }

        // Timed runs
        const int ITERS = 20;
        double best_ms = 1e9;
        size_t best_js = 0;
        for (int i = 0; i < ITERS; i++) {
            js = max_jpg;
            auto t0 = Clock::now();
            e = process_jpeg_encode(src.data(), jpg.data(), &js,
                                     t.w, t.h, 3,
                                     JpegAlgorithm::ENCODE_CUDA, 8, p);
            auto t1 = Clock::now();
            if (e != JpegError::Ok) {
                printf("%-12s ENCODE ERROR: %s\n", t.name, jpeg_error_message(e));
                best_ms = -1;
                break;
            }
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best_ms) { best_ms = ms; best_js = js; }
        }
        if (best_ms < 0) continue;

        double raw_mb = (double)px / (1024.0 * 1024.0);
        double jpg_kb = (double)best_js / 1024.0;
        double ratio = (double)px / (double)best_js;
        double raw_gbps = raw_mb / best_ms * 1000.0 / 1024.0;
        double jpg_mbps = jpg_kb / best_ms * 1000.0 / 1024.0;

        printf("%-12s %7.1fMB | %9.2f %9.1f %9.1f:1 | %9.2f %9.1f\n",
               t.name, raw_mb, best_ms, jpg_kb, ratio, raw_gbps, jpg_mbps);

        // Verify decode
        int ow, oh, oc;
        e = process_jpeg_decode(jpg.data(), best_js, dec.data(), &ow, &oh, &oc,
                                 JpegAlgorithm::DECODE_BASELINE);
        if (e != JpegError::Ok) {
            printf("  -> DECODE FAILED: %s  (CPU JPEG=%zu bytes, GPU JPEG=%zu bytes)\n",
                   jpeg_error_message(e), cpu_js, best_js);
            printf("     GPU JPEG[0..3]=%02X %02X %02X %02X  end=%02X %02X\n",
                   jpg[0], jpg[1], jpg[2], jpg[3],
                   best_js>=2?jpg[best_js-2]:0, best_js>=1?jpg[best_js-1]:0);
        }
    }

    printf("\nDone.\n");
    return 0;
}
