// Quick DCT correctness test — standalone CUDA
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>

#define CUDA_CHK(c) do { cudaError_t e=c; if(e!=cudaSuccess){printf("CUDA err %d at %d: %s\n",e,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

// Precomputed cosine tables (same as optimized kernel)
__constant__ float d_dct_cos[8][8] = {
    { 1.00000000f,  1.00000000f,  1.00000000f,  1.00000000f,  1.00000000f,  1.00000000f,  1.00000000f,  1.00000000f },
    { 0.98078528f,  0.83146961f,  0.55557023f,  0.19509032f, -0.19509032f, -0.55557023f, -0.83146961f, -0.98078528f },
    { 0.92387953f,  0.38268343f, -0.38268343f, -0.92387953f, -0.92387953f, -0.38268343f,  0.38268343f,  0.92387953f },
    { 0.83146961f, -0.19509032f, -0.98078528f, -0.55557023f,  0.55557023f,  0.98078528f,  0.19509032f, -0.83146961f },
    { 0.70710678f, -0.70710678f, -0.70710678f,  0.70710678f,  0.70710678f, -0.70710678f, -0.70710678f,  0.70710678f },
    { 0.55557023f, -0.98078528f,  0.19509032f,  0.83146961f, -0.83146961f, -0.19509032f,  0.98078528f, -0.55557023f },
    { 0.38268343f, -0.92387953f,  0.92387953f, -0.38268343f, -0.38268343f,  0.92387953f, -0.92387953f,  0.38268343f },
    { 0.19509032f, -0.55557023f,  0.83146961f, -0.98078528f,  0.98078528f, -0.83146961f,  0.55557023f, -0.19509032f }
};

__constant__ float d_dct_scale_c[8] = {
    0.70710678f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};

__constant__ int d_kZigzag[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

// GPU separable DCT (same as optimized kernel)
__global__ void kernel_fdct(const float* __restrict__ plane,
                             float* __restrict__ dct_out,
                             int width, int height, int num_blocks_x) {
    int tid = threadIdx.x;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    __shared__ float s_data[64];
    __shared__ float s_tmp[64];

    int px_base = block_x * 8;
    int py_base = block_y * 8;

    for (int row = 0; row < 8; row++) {
        int py = min(py_base + row, height - 1);
        int px = min(px_base + tid, width - 1);
        s_data[row * 8 + tid] = plane[py * width + px];
    }
    __syncthreads();

    for (int y = 0; y < 8; y++) {
        float sum = 0.0f;
        for (int x = 0; x < 8; x++)
            sum += s_data[y * 8 + x] * d_dct_cos[tid][x];
        s_tmp[y * 8 + tid] = sum;
    }
    __syncthreads();

    float scale_v = 0.25f * d_dct_scale_c[tid];
    for (int u = 0; u < 8; u++) {
        float sum = 0.0f;
        for (int y = 0; y < 8; y++)
            sum += s_tmp[y * 8 + u] * d_dct_cos[tid][y];
        float coeff = sum * scale_v * d_dct_scale_c[u];
        int block_idx = block_y * num_blocks_x + block_x;
        dct_out[block_idx * 64 + tid * 8 + u] = coeff;
    }
}

// CPU reference DCT (common.hpp style)
void cpu_fdct_8x8(const float in[64], float out[64]) {
    float tmp[64];
    for (int y = 0; y < 8; y++) {
        for (int u = 0; u < 8; u++) {
            float sum = 0;
            for (int x = 0; x < 8; x++)
                sum += in[y*8+x] * cosf((2*x+1)*u*3.14159265f/16);
            float scale = 0.5f * (u==0 ? 0.70710678f : 1.0f);
            tmp[y*8+u] = sum * scale;
        }
    }
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            float sum = 0;
            for (int y = 0; y < 8; y++)
                sum += tmp[y*8+u] * cosf((2*y+1)*v*3.14159265f/16);
            float scale = 0.5f * (v==0 ? 0.70710678f : 1.0f);
            out[v*8+u] = sum * scale;
        }
    }
}

int main() {
    printf("=== GPU DCT Correctness Test ===\n\n");

    // Test 1: Flat image (all pixels = 100.0f)
    {
        printf("Test 1: Flat image (all = 100.0f)\n");
        int w = 16, h = 16;
        int padded_w = 16, padded_h = 16;
        int bx = 2, by = 2;
        float* h_plane = new float[w*h];
        for (int i = 0; i < w*h; i++) h_plane[i] = 100.0f;

        float *d_plane, *d_dct;
        CUDA_CHK(cudaMalloc(&d_plane, padded_w * padded_h * sizeof(float)));
        CUDA_CHK(cudaMalloc(&d_dct, bx * by * 64 * sizeof(float)));
        CUDA_CHK(cudaMemcpy(d_plane, h_plane, padded_w*padded_h*sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(8, 1, 1);
        dim3 grid(bx, by);
        kernel_fdct<<<grid, block>>>(d_plane, d_dct, padded_w, padded_h, bx);
        CUDA_CHK(cudaDeviceSynchronize());

        float h_dct[4*64];
        CUDA_CHK(cudaMemcpy(h_dct, d_dct, bx*by*64*sizeof(float), cudaMemcpyDeviceToHost));

        // CPU reference
        float cpu_dct[64];
        float block_in[64];
        for (int i = 0; i < 64; i++) block_in[i] = 100.0f;
        cpu_fdct_8x8(block_in, cpu_dct);

        printf("  CPU DC  = %.6f\n", cpu_dct[0]);
        printf("  GPU DC  = %.6f\n", h_dct[0]);
        printf("  CPU AC max = %.6f\n", [&]{float m=0;for(int i=1;i<64;i++)m=fmaxf(m,fabsf(cpu_dct[i]));return m;}());
        printf("  GPU AC max = %.6f\n", [&]{float m=0;for(int i=1;i<64;i++)m=fmaxf(m,fabsf(h_dct[i]));return m;}());

        bool match = fabsf(h_dct[0] - cpu_dct[0]) < 0.01f;
        printf("  DC match: %s\n", match ? "PASS" : "FAIL");

        CUDA_CHK(cudaFree(d_plane));
        CUDA_CHK(cudaFree(d_dct));
        delete[] h_plane;
    }

    // Test 2: Level-shifted flat image (Y-128)
    {
        printf("\nTest 2: Level-shifted flat image (all = -10.84f, simulating Y-128)\n");
        int w = 16, h = 16;
        int bx = w/8, by = h/8;
        float* h_plane = new float[w*h];
        for (int i = 0; i < w*h; i++) h_plane[i] = -10.84f;  // Y=117.16 - 128

        float *d_plane, *d_dct;
        CUDA_CHK(cudaMalloc(&d_plane, w * h * sizeof(float)));
        CUDA_CHK(cudaMalloc(&d_dct, bx * by * 64 * sizeof(float)));
        CUDA_CHK(cudaMemcpy(d_plane, h_plane, w*h*sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(8, 1, 1);
        dim3 grid(bx, by);
        kernel_fdct<<<grid, block>>>(d_plane, d_dct, w, h, bx);
        CUDA_CHK(cudaDeviceSynchronize());

        float h_dct[4*64];
        CUDA_CHK(cudaMemcpy(h_dct, d_dct, bx*by*64*sizeof(float), cudaMemcpyDeviceToHost));

        float cpu_dct[64];
        float block_in[64];
        for (int i = 0; i < 64; i++) block_in[i] = -10.84f;
        cpu_fdct_8x8(block_in, cpu_dct);

        printf("  CPU DC  = %.6f (expect 8 * -10.84 = %.2f)\n", cpu_dct[0], 8.0f * -10.84f);
        printf("  GPU DC  = %.6f\n", h_dct[0]);
        printf("  CPU AC max = %.6f\n", [&]{float m=0;for(int i=1;i<64;i++)m=fmaxf(m,fabsf(cpu_dct[i]));return m;}());
        printf("  GPU AC max = %.6f\n", [&]{float m=0;for(int i=1;i<64;i++)m=fmaxf(m,fabsf(h_dct[i]));return m;}());

        bool match = fabsf(h_dct[0] - cpu_dct[0]) < 0.01f;
        printf("  DC match: %s\n", match ? "PASS" : "FAIL");

        CUDA_CHK(cudaFree(d_plane));
        CUDA_CHK(cudaFree(d_dct));
        delete[] h_plane;
    }

    // Test 3: Gradient
    {
        printf("\nTest 3: Gradient image\n");
        int w = 16, h = 16;
        int bx = w/8, by = h/8;
        float* h_plane = new float[w*h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                h_plane[y*w+x] = (float)(x * 255) / (w - 1) - 128.0f;

        float *d_plane, *d_dct;
        CUDA_CHK(cudaMalloc(&d_plane, w * h * sizeof(float)));
        CUDA_CHK(cudaMalloc(&d_dct, bx * by * 64 * sizeof(float)));
        CUDA_CHK(cudaMemcpy(d_plane, h_plane, w*h*sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(8, 1, 1);
        dim3 grid(bx, by);
        kernel_fdct<<<grid, block>>>(d_plane, d_dct, w, h, bx);
        CUDA_CHK(cudaDeviceSynchronize());

        float h_dct[4*64];
        CUDA_CHK(cudaMemcpy(h_dct, d_dct, bx*by*64*sizeof(float), cudaMemcpyDeviceToHost));

        // CPU reference for first 8x8 block
        float block_in[64];
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
                block_in[y*8+x] = h_plane[y*w+x];
        float cpu_dct[64];
        cpu_fdct_8x8(block_in, cpu_dct);

        printf("  CPU DC  = %.6f\n", cpu_dct[0]);
        printf("  GPU DC  = %.6f\n", h_dct[0]);
        float max_diff = 0;
        for (int i = 0; i < 64; i++) max_diff = fmaxf(max_diff, fabsf(h_dct[i] - cpu_dct[i]));
        printf("  Max diff across all 64 coeffs = %.6f\n", max_diff);
        printf("  Result: %s\n", max_diff < 0.5f ? "PASS" : "FAIL");

        CUDA_CHK(cudaFree(d_plane));
        CUDA_CHK(cudaFree(d_dct));
        delete[] h_plane;
    }

    printf("\n=== Done ===\n");
    return 0;
}
