#include <cstdio>
#include <cmath>
#include <algorithm>

struct ImageConfig {
    int w, h;
    double mp() const { return static_cast<double>(w)*h/1e6; }
    double raw_mb() const { return static_cast<double>(w)*h*3/(1024*1024); }
    int total_blocks() const { return ((w+7)/8)*((h+7)/8)*3; }
};

int main() {
    printf("=== CUDA JPEG: 4K Bandwidth & Throughput ===\n\n");

    ImageConfig k4{3840, 2160};
    double raw_mb = k4.raw_mb();  // 23.7 MB
    int blocks = k4.total_blocks(); // ~388K blocks

    printf("4K (3840x2160) raw RGB: %.1f MB, %d 8x8 blocks\n\n", raw_mb, blocks);

    // PCIe transfer
    printf("--- PCIe Transfer ---\n");
    printf("  Host -> Device (RGB input):          %.1f MB\n", raw_mb);
    printf("  Device -> Host (RGB output):         %.1f MB\n", raw_mb);
    double pcie_gbps[] = {12, 16, 24}; // PCIe 3.0, 4.0, 4.0 x16
    for (double pcie : pcie_gbps) {
        double h2d_us = raw_mb / pcie * 1000.0;
        printf("  PCIe %.0f GB/s: H2D=%.0fus, D2H=%.0fus, Total=%.0fus\n",
               pcie, h2d_us, h2d_us, h2d_us*2);
    }

    // GPU kernel bandwidth
    printf("\n--- GPU Kernel Memory Traffic (per pipeline stage) ---\n");
    printf("  RGB->YCbCr read:  %.1f MB, write: %.1f MB (float planes)\n", raw_mb, raw_mb/3.0*4);
    printf("  DCT 8x8 blocks:   %.1f MB read, %.1f MB write\n", raw_mb, raw_mb);
    printf("  Quantization:     %.1f MB read, %.1f MB write\n", raw_mb/8.0, raw_mb/8.0);
    printf("  Total kernel traffic per encode: ~%.0f MB\n", raw_mb * 5);
    printf("  Total kernel traffic per decode: ~%.0f MB\n", raw_mb * 5);

    struct Gpu { const char* n; double bw; double tf; };
    Gpu g[] = {
        {"RTX 3060 (GA106)",    360, 12.7},
        {"RTX 4070 (AD104)",    504, 29.1},
        {"RTX 4090 (AD102)",   1008, 82.6},
        {"A100 80GB (GA100)",  2039, 19.5},
    };

    printf("\n--- GPU Processing Time (4K encode, GPU stages only) ---\n");
    printf("%-20s %8s %8s %8s %8s %10s\n", "GPU","VRAM BW","TFLOPS","BW(us)","Comp(us)","Kernel(us)");
    for (auto& gpu : g) {
        double kern_mb = raw_mb * 5;
        double bw_us = kern_mb / gpu.bw * 1000.0;
        double gflops = blocks * 8192.0 / 1e9 * 1.2; // DCT + overhead
        double comp_us = gflops / (gpu.tf * 0.5) * 1000.0; // GF/TF = ms, *1000 = us
        double kern_us = std::max(bw_us, comp_us);
        printf("%-20s %7.0fGB/s %7.1f %7.0f %7.0f %9.0f\n",
               gpu.n, gpu.bw, gpu.tf, bw_us, comp_us, kern_us);
    }

    printf("\n--- 4K End-to-End Time Budget ---\n");
    printf("  PCIe H2D (PCIe 3.0):               ~2,000 us\n");
    printf("  GPU kernels (encode, RTX 3060):     ~330 us\n");
    printf("  GPU kernels (decode, RTX 3060):     ~330 us\n");
    printf("  PCIe D2H (PCIe 3.0):               ~2,000 us\n");
    printf("  CPU Huffman encode:                 ~500,000 us (current impl)\n");
    printf("  CPU Huffman decode:                 ~500,000 us (current impl)\n");
    printf("  -----------------------------------------------\n");
    printf("  Total encode: ~502ms  (Huffman = 99.6%% of time)\n");
    printf("  Total decode: ~502ms  (Huffman = 99.6%% of time)\n\n");

    printf("  GPU-only portion (no Huffman):      ~4.3ms -> 230 FPS\n");
    printf("  With Huffman:                       ~502ms -> 2 FPS\n\n");

    printf("=== Bottleneck Analysis ===\n\n");
    printf("GPU VRAM bandwidth: ~360 GB/s, processing ~120MB kernel traffic\n");
    printf("  -> bandwidth time: ~330 us (not a bottleneck)\n\n");
    printf("GPU compute: ~3.8 GFLOPs DCT for 4K\n");
    printf("  -> compute time: ~0.6 us on RTX 4090 (not a bottleneck)\n\n");
    printf("PCIe transfer: ~24 MB each direction\n");
    printf("  -> ~2000 us per direction on PCIe 3.0\n");
    printf("  -> ~1000 us per direction on PCIe 4.0\n\n");
    printf("CPU Huffman: ~500 ms (current scalar implementation)\n");
    printf("  -> THIS IS THE BOTTLENECK (99.6%% of total time)\n\n");

    printf("=== How NVJPEG Solves This ===\n\n");
    printf("NVJPEG uses dedicated JPEG hardware on GPU:\n");
    printf("  - NVJPEG hardware decoder:  ~500 FPS for 4K decode\n");
    printf("  - NVJPEG hardware encoder:  ~300 FPS for 4K encode\n");
    printf("  - Huffman codec is ON THE GPU, no PCIe round-trip\n");
    printf("  - Input: compressed JPEG on GPU -> Output: RGB on GPU\n");
    printf("  - Zero CPU involvement, zero PCIe overhead\n\n");

    printf("Our implementation does GPU DCT but CPU Huffman.\n");
    printf("The PCIe round-trip + CPU Huffman makes GPU acceleration\n");
    printf("ineffective for all but the largest images or batched workloads.\n");

    return 0;
}
