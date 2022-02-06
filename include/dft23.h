#pragma once

#include <complex.h>
#include <immintrin.h>

#define DFT_SIZE 23

#define _PERM_MASK(a, b, c, d) ((d << 6) + (c << 4) + (b << 2) + a)

inline static __m256 dft23_cmult4(__m256 a, __m256 b) {
    __m256 neg = _mm256_set_ps(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    __m256 c = _mm256_mul_ps(a, b); // a*c|b*d
    b = _mm256_permute_ps(b, 0b10110001); // Swapping real/imag; d|c
    __m256 d = _mm256_mul_ps(a, b); // a*d|b*c
    c = _mm256_mul_ps(c, neg); // a*c|-b*d
    return _mm256_permute_ps(_mm256_hadd_ps(c, d), 0b11011000);
}

void gen_dft23_coeffs_1(complex float *coeffs);
void gen_dft23_coeffs_3(float *coeffs);
void gen_dft23_coeffs_4(float *coeffs);
void dft23_1(complex float *buf, complex float *coeffs, complex float *out);
void dft23_2(complex float *buf, complex float *out);
void dft23_3(complex float *buf, float *coeffs, complex float *out);
void dft23_4(complex float *buf, float *coeffs, complex float *out);
