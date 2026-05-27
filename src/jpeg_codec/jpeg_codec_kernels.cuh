#ifndef JPEG_CODEC_CUDA_KERNELS_CUH
#define JPEG_CODEC_CUDA_KERNELS_CUH

#include <cstdint>
#include <cstddef>

// ============================================================
//  GPU Huffman lookup table (16-bit index for fast decode)
// ============================================================

struct GpuHuffEntry {
    uint8_t symbol;   // decoded symbol value
    uint8_t code_len; // Huffman code length in bits (0 = invalid)
};

struct GpuHuffTable {
    GpuHuffEntry* d_entries;  // 65536 entries on device
};

// ============================================================
//  JPEG CUDA workspace
// ============================================================

struct JpegCudaWorkspace {
    // Image I/O
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;

    // YCbCr float planes (padded to 8x8 boundaries)
    float* d_y_plane = nullptr;
    float* d_cb_plane = nullptr;
    float* d_cr_plane = nullptr;

    // Quantized DCT coefficients (int16_t, zigzag order, per block)
    int16_t* d_quant_coeffs = nullptr;

    // Pre-computed DC differences (per block)
    int16_t* d_dc_diffs = nullptr;

    // Quantization tables (2 tables × 64 ints = 512 bytes, raster order)
    int* d_quant_tables = nullptr;

    // Huffman bitstream (output from pass2 encoder)
    uint32_t* d_bitstream = nullptr;
    size_t d_bitstream_capacity = 0;

    // Per-block metadata
    int* d_block_lengths = nullptr;   // bit length of each block (from pass1)
    int* d_block_offsets = nullptr;   // bit offset of each block (from prefix sum)
    int* d_block_sums = nullptr;      // scratch for GPU parallel prefix sum
    int total_blocks = 0;

    // GPU-side byte-stuffed output
    uint8_t* d_stuffed = nullptr;     // byte-stuffed scan data
    int* d_stuffed_size = nullptr;    // output size in bytes
    int* d_total_bits = nullptr;      // total bitstream bits (from prefix sum)

    // Huffman lookup tables (for GPU decode)
    GpuHuffTable dc_luma_table;
    GpuHuffTable ac_luma_table;
    GpuHuffTable dc_chroma_table;
    GpuHuffTable ac_chroma_table;

    // Pinned host memory
    uint8_t* h_input_pinned = nullptr;
    uint8_t* h_output_pinned = nullptr;
    size_t h_input_bytes = 0;
    size_t h_output_bytes = 0;

    // Image dimensions
    int width = 0;
    int height = 0;
    int bit_depth = 0;
    int blocks_x = 0;
    int blocks_y_dim = 0;
    int blocks_per_channel = 0;

    JpegCudaWorkspace() = default;
    ~JpegCudaWorkspace() { release(); }
    JpegCudaWorkspace(const JpegCudaWorkspace&) = delete;
    JpegCudaWorkspace& operator=(const JpegCudaWorkspace&) = delete;
    JpegCudaWorkspace(JpegCudaWorkspace&& o) noexcept;
    JpegCudaWorkspace& operator=(JpegCudaWorkspace&&) = delete;

    bool ensure(int w, int h, int bit_depth);
    void release();
};

// ============================================================
//  Kernel declarations
// ============================================================

#ifdef __cplusplus
extern "C" {
#endif

// --- Color conversion kernels ---
void cuda_jpeg_rgb_to_ycbcr(const uint8_t* d_input, float* d_y, float* d_cb, float* d_cr,
                             int width, int height, int bit_depth);
void cuda_jpeg_ycbcr_to_rgb(const float* d_y, const float* d_cb, const float* d_cr,
                             uint8_t* d_output, int width, int height, int bit_depth);

// --- Fused DCT + Quantize (single plane, planar order) ---
void cuda_jpeg_fdct_quantize(float* d_plane, int16_t* d_quant_coeffs,
                              const int* d_qtable,
                              int width, int height,
                              int num_blocks_x, int num_blocks_y,
                              int block_offset);

// --- Fused DCT + Quantize (MCU order, stride=3 for interleaved Y,Cb,Cr) ---
void cuda_jpeg_fdct_quantize_mcu(float* d_plane, int16_t* d_quant_coeffs,
                                  const int* d_qtable,
                                  int width, int height,
                                  int num_blocks_x, int num_blocks_y,
                                  int component);

// --- Fused 3-plane DCT + Quantize (Y, Cb, Cr in one kernel launch) ---
void cuda_jpeg_fdct_quantize_3plane(
    float* d_y, float* d_cb, float* d_cr,
    int16_t* d_quant_coeffs,
    const int* d_luma_qt, const int* d_chroma_qt,
    int width, int height,
    int num_blocks_x, int num_blocks_y,
    int blocks_per_channel);

// --- IDCT + Dequantize (int16_t zigzag coefficients → spatial float plane) ---
void cuda_jpeg_idct_reconstruct(const int16_t* d_quant_coeffs, const int* d_qtable,
                                 float* d_spatial,
                                 int width, int height,
                                 int num_blocks_x, int num_blocks_y,
                                 int block_offset);

// --- Dequantize only (int16_t zigzag → float raster) ---
void cuda_jpeg_dequantize(const int16_t* d_quant_in, float* d_dct_out,
                           const int* d_qtable, int total_blocks, int block_offset);

// --- DC difference computation ---
void cuda_jpeg_compute_dc_diffs(const int16_t* d_quant_coeffs,
                                 int16_t* d_dc_diffs,
                                 int blocks_per_comp, int blocks_x);

// --- Two-pass Huffman encoder (DC diffs computed inline) ---
void cuda_huffman_pass1_lengths(const int16_t* d_quant_coeffs,
                                 int* d_block_lengths,
                                 int total_blocks,
                                 int blocks_y, int blocks_cb, int blocks_cr);

void cuda_huffman_pass2_write(const int16_t* d_quant_coeffs,
                               const int* d_block_offsets,
                               uint32_t* d_bitstream,
                               int total_blocks,
                               int blocks_y, int blocks_cb, int blocks_cr);

// --- GPU parallel prefix sum ---
void cuda_gpu_prefix_sum(int* d_lengths, int* d_offsets,
                          int* d_temp, int* d_total, int n);

// --- GPU-side byte stuffing ---
void cuda_byte_stuff_bitstream(const uint32_t* d_bitstream, int total_bits,
                                uint8_t* d_stuffed, int* d_stuffed_size);

// --- GPU parallel byte stuffing (two-pass, multi-thread) ---
void cuda_byte_stuff_parallel(const uint32_t* d_bitstream,
                               int* d_total_bits,
                               uint8_t* d_byte_buffers, int* d_byte_counts,
                               int* d_byte_offsets, int* d_scratch,
                               int* d_total_bytes, uint8_t* d_stuffed_out,
                               int words_per_thread, int num_threads);

// --- Huffman decode ---
void cuda_huffman_decode_blocks(
    JpegCudaWorkspace* ws,
    float* d_blocks,
    int num_blocks_y, int num_blocks_cb, int num_blocks_cr,
    int num_blocks_x, int num_blocks_y_dim,
    const int* d_luma_qt, const int* d_chroma_qt);

// --- Huffman table management ---
bool cuda_huffman_build_tables(JpegCudaWorkspace* ws);

#ifdef __cplusplus
}
#endif

#endif // JPEG_CODEC_CUDA_KERNELS_CUH
