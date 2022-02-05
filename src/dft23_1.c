#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>

#include "dft23.h"

void gen_dft23_coeffs_1(complex float *coeffs) {
    for (int k = 0; k < DFT_SIZE; k++) {
        for (int n = 0; n < DFT_SIZE; n++)
            coeffs[24*k+n] = cos(2 * M_PI / DFT_SIZE * k * n) - I * sin(2 * M_PI / DFT_SIZE * k * n);
        coeffs[24*k+DFT_SIZE] = 0;
    }
}

inline static void dft23_chadd4(__m256 a, float *out) {
        __m256 sum = _mm256_add_ps(a, _mm256_permute2f128_ps(a, a, 1));
        sum = _mm256_add_ps(sum, _mm256_permute_ps(sum, 0b00001110));
        _mm256_maskstore_ps(out, _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1), sum);
}

void dft23_1(complex float *buf, complex float *coeffs, complex float *out) {
    for (int k = 0; k < DFT_SIZE; k++) {
        __m256 sum = _mm256_set1_ps(0.0f);
        for (int i = 0; i < 6; i++) {
            __m256 a = _mm256_loadu_ps((float*) &buf[4*i]);
            __m256 b = _mm256_load_ps((float*) &coeffs[24*k + 4*i]);
            __m256 c = dft23_cmult4(a, b);
            sum = _mm256_add_ps(sum, c);
        }

        dft23_chadd4(sum, (float *) &out[k]);
    }
}
