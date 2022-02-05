#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>

#include "dft23.h"

void dft23_3(complex float *buf, float *coeffs, complex float *out) {
    __m256 b_0, b_1, b_r, b_i, c_r, c_i, s_r, s_i, o_r, o_i, o;
    // b_0, b_1: Input samples, interleaved
    // b_r, b_i: Input samples, split into real and imaginary vector
    // c_r, c_i: DFT coefficient, split into complex and imaginary
    // s_r, s_i: Result of complex multiplication
    // o_r, o_i: DFT result accumulators, used for deferred processing of result

    s_r = _mm256_setzero_ps();
    s_i = _mm256_setzero_ps();

#define LOOP_BODY(j)                                                                    \
    {                                                                                   \
        b_0 = _mm256_loadu_ps((float *) &buf[8*j]);                                     \
        b_1 = _mm256_loadu_ps((float *) &buf[8*j+4]);                                   \
                                                                                        \
        b_r = _mm256_shuffle_ps(b_0, b_1, _PERM_MASK(0, 2, 0, 2));                      \
        b_i = _mm256_shuffle_ps(b_0, b_1, _PERM_MASK(1, 3, 1, 3));                      \
                                                                                        \
        c_r = _mm256_loadu_ps(&coeffs[48*k+j*16]);                                      \
        c_i = _mm256_loadu_ps(&coeffs[48*k+j*16+8]);                                    \
                                                                                        \
        s_r = _mm256_add_ps(s_r, _mm256_fmsub_ps(b_r, c_r, _mm256_mul_ps(b_i, c_i)));   \
        s_i = _mm256_add_ps(s_i, _mm256_fmadd_ps(b_r, c_i, _mm256_mul_ps(b_i, c_r)));   \
    }

    int k = 0;
    for (int j = 0; j < 3; j++)
        LOOP_BODY(j);

#define _mm256_flip(x) _mm256_permute2f128_ps(x, x, 1)

    o_r = _mm256_add_ps(s_r, _mm256_flip(s_r));
    o_i = _mm256_add_ps(s_i, _mm256_flip(s_i));
    o = _mm256_permute2f128_ps(o_r, o_i, 0x20);

    for (k = 1; k < DFT_SIZE; k++) {
        s_r = _mm256_setzero_ps();
        s_i = _mm256_setzero_ps();

        LOOP_BODY(0);

        o = _mm256_hadd_ps(o, o);

        LOOP_BODY(1);

        o = _mm256_hadd_ps(o, o);

        LOOP_BODY(2);

        out[k-1] = _mm256_cvtss_f32(o) + I*_mm256_cvtss_f32(_mm256_flip(o));

        o_r = _mm256_add_ps(s_r, _mm256_flip(s_r));
        o_i = _mm256_add_ps(s_i, _mm256_flip(s_i));
        o = _mm256_permute2f128_ps(o_r, o_i, 0x20);
    }

    o = _mm256_hadd_ps(o, o);
    o = _mm256_hadd_ps(o, o);

    out[k-1] = _mm256_cvtss_f32(o) + I * _mm256_cvtss_f32(_mm256_flip(o));
}


inline static size_t coeff_index_gen_3(size_t k, size_t n, bool imag) {
    size_t slot = n & ~0x7UL;
    size_t i = n & 0x7UL;
    const size_t IDX_TR[8] = {0, 1, 4, 5, 2, 3, 6, 7};
    return 2*(24*k + slot) + IDX_TR[i] + imag*8;
}

void gen_dft23_coeffs_3(float *coeffs) {
    for (int k = 0; k < DFT_SIZE; k++) {
        for (int n = 0; n < DFT_SIZE; n++) {
            coeffs[coeff_index_gen_3(k, n, false)] = cos(2 * M_PI / DFT_SIZE * k * n);
            coeffs[coeff_index_gen_3(k, n, true)] = -sin(2 * M_PI / DFT_SIZE * k * n);
        }
        coeffs[coeff_index_gen_3(k, DFT_SIZE, false)] = 0;
        coeffs[coeff_index_gen_3(k, DFT_SIZE, true)] = 0;
    }
}
