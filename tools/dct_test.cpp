#include "../src/jpeg_codec/common.hpp"
#include <cstdio>
#include <cmath>

int main() {
    float block[64];

    // Test pure DCT/IDCT roundtrip (no quant)
    printf("=== Pure DCT/IDCT roundtrip: constant = 50.0 ===\n");
    for (int i = 0; i < 64; i++) block[i] = 50.0f;

    jpeg_codec::detail::fdct_8x8(block);
    printf("DCT DC = %f\n", block[0]);

    jpeg_codec::detail::idct_8x8(block);
    printf("IDCT output:\n");
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            printf("%8.2f ", block[y * 8 + x]);
        }
        printf("\n");
    }

    // Also check: is the DCT output in the right order?
    printf("\n=== DCT of constant 1.0 ===\n");
    for (int i = 0; i < 64; i++) block[i] = 1.0f;
    jpeg_codec::detail::fdct_8x8(block);
    printf("DCT output:\n");
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            printf("%8.2f ", block[y * 8 + x]);
        }
        printf("\n");
    }

    // DCT of impulse
    printf("\n=== DCT of impulse at (0,0) = 100 ===\n");
    for (int i = 0; i < 64; i++) block[i] = 0.0f;
    block[0] = 100.0f;
    jpeg_codec::detail::fdct_8x8(block);
    printf("DCT output (should be ~12.5 for all coeffs):\n");
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            printf("%8.2f ", block[y * 8 + x]);
        }
        printf("\n");
    }

    return 0;
}
