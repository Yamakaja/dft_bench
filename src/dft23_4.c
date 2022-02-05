#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>

#include "dft23.h"

void dft23_4(complex float *buf, float *coeffs, complex float *out) {
    __m256 k0_r = _mm256_setzero_ps(), k0_i = _mm256_setzero_ps();
    __m256 k1_r = _mm256_setzero_ps(), k1_i = _mm256_setzero_ps();
    __m256 k2_r = _mm256_setzero_ps(), k2_i = _mm256_setzero_ps();

    size_t coeff_offset = 0;

    for (int n = 0; n < DFT_SIZE; n++) {
        __m256 x_r = _mm256_set1_ps(crealf(buf[n]));
        __m256 x_i = _mm256_set1_ps(cimagf(buf[n]));

        // 0-7
        __m256 c0_r = _mm256_loadu_ps(&coeffs[coeff_offset]);
        __m256 c0_i = _mm256_loadu_ps(&coeffs[coeff_offset+8]);
        coeff_offset += 16; 

        k0_r = _mm256_add_ps(k0_r, _mm256_fmsub_ps(x_r, c0_r, _mm256_mul_ps(x_i, c0_i)));
        k0_i = _mm256_add_ps(k0_i, _mm256_fmadd_ps(x_r, c0_i, _mm256_mul_ps(x_i, c0_r)));

        // 8-15
        __m256 c1_r = _mm256_loadu_ps(&coeffs[coeff_offset]);
        __m256 c1_i = _mm256_loadu_ps(&coeffs[coeff_offset+8]);
        coeff_offset += 16; 

        k1_r = _mm256_add_ps(k1_r, _mm256_fmsub_ps(x_r, c1_r, _mm256_mul_ps(x_i, c1_i)));
        k1_i = _mm256_add_ps(k1_i, _mm256_fmadd_ps(x_r, c1_i, _mm256_mul_ps(x_i, c1_r)));

        // 16-23
        __m256 c2_r = _mm256_loadu_ps(&coeffs[coeff_offset]);
        __m256 c2_i = _mm256_loadu_ps(&coeffs[coeff_offset+8]);
        coeff_offset += 16; 

        k2_r = _mm256_add_ps(k2_r, _mm256_fmsub_ps(x_r, c2_r, _mm256_mul_ps(x_i, c2_i)));
        k2_i = _mm256_add_ps(k2_i, _mm256_fmadd_ps(x_r, c2_i, _mm256_mul_ps(x_i, c2_r)));
    }

    _mm256_storeu_ps((float *) &out[0], _mm256_unpacklo_ps(k0_r, k0_i));
    _mm256_storeu_ps((float *) &out[4], _mm256_unpackhi_ps(k0_r, k0_i));

    _mm256_storeu_ps((float *) &out[8], _mm256_unpacklo_ps(k1_r, k1_i));
    _mm256_storeu_ps((float *) &out[12], _mm256_unpackhi_ps(k1_r, k1_i));

    _mm256_storeu_ps((float *) &out[16], _mm256_unpacklo_ps(k2_r, k2_i));
    _mm256_storeu_ps((float *) &out[20], _mm256_unpackhi_ps(k2_r, k2_i));
}


inline static size_t coeff_index_gen_4(size_t k, size_t n, bool imag) {
    size_t slot = k & ~0x7UL;
    size_t i = k & 0x7UL;
    const size_t IDX_TR[8] = {0, 1, 4, 5, 2, 3, 6, 7};
    return 2*(24*n + slot) + IDX_TR[i] + imag*8;
}

void gen_dft23_coeffs_4(float *coeffs) {
    for (int n = 0; n < DFT_SIZE; n++) {
        for (int k = 0; k < DFT_SIZE; k++) {
            coeffs[coeff_index_gen_4(k, n, false)] = cos(2 * M_PI / DFT_SIZE * k * n);
            coeffs[coeff_index_gen_4(k, n, true)] = -sin(2 * M_PI / DFT_SIZE * k * n);
        }
        coeffs[coeff_index_gen_4(DFT_SIZE, n, false)] = 0;
        coeffs[coeff_index_gen_4(DFT_SIZE, n, true)] = 0;
    }
}
