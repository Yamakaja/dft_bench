#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <stdbool.h>

#include <immintrin.h>

#include <math.h>
#include <complex.h>

#include <fftw3.h>

#include "xoroshiro128plus.h"

#define SAMPLE_RATE 1000
#define SINE_FREQ   240
#define NOISE_AMPLITUDE 0.05

#define DFT_SIZE 23

xoroshiro128plus_t xoro_state;

complex double gaussian(uint64_t u) {
    uint64_t u_0 = (u >> 16) & 0xFFFFFFFFFFFFUL;
    uint64_t u_1 = u & 0xFFFF;

    double f = sqrt(-2 * log(u_0 * 3.55271367880050e-15));

    double out[2];
    out[0] = f * cos(2 * M_PI * 1.52587890625e-05 * u_1);
    out[1] = f * sin(2 * M_PI * 1.52587890625e-05 * u_1);
    return out[0] + I*out[1];
}

void prepare_sample_sequence(complex float *buf, size_t len) {
    float t = 0;

    for (size_t i = 0; i < len; i++) {
        buf[i] = sinf(2 * M_PI * SINE_FREQ * t) + NOISE_AMPLITUDE * gaussian(xoroshiro128plus_next(&xoro_state));
        t += 1.0f/SAMPLE_RATE;
    }
}


void dump_to_file(complex float *buf, size_t len) {
    FILE *dout = fopen("/tmp/samples.dat", "w");

    if (dout == NULL) {
        perror("fopen");
        exit(EXIT_FAILURE);
    }
    
    for (size_t i = 0; i < len;)
        i += fwrite(buf, sizeof(*buf), len-i, dout);

    fclose(dout);
}

int64_t bench_fftw(complex float *buf, size_t len) {
    struct timespec tp_start; // tv_sec, tv_nsec
    struct timespec tp_end; // tv_sec, tv_nsec

    // Setup fftw
    fftwf_complex *in, *out;
    fftwf_plan p;

    in = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * DFT_SIZE);
    out = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * DFT_SIZE);

    p = fftwf_plan_dft_1d(DFT_SIZE, in, out, FFTW_FORWARD, FFTW_MEASURE);

    complex float sum = 0;

    clock_gettime(CLOCK_MONOTONIC, &tp_start);

    for (ssize_t i = 0; i < ((ssize_t) (len / DFT_SIZE)); i++) {
        for (size_t j = 0; j < DFT_SIZE; j++)
            in[j] = buf[DFT_SIZE*i + j];

        fftwf_execute(p);

        for (int j = 0; j < DFT_SIZE; j++)
            sum += out[j];
    }
    
    clock_gettime(CLOCK_MONOTONIC, &tp_end);

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);

    printf("%f + %fj\n", creal(sum), cimag(sum));

    int64_t s = tp_end.tv_sec - tp_start.tv_sec;
    int64_t ns = (int64_t) tp_end.tv_nsec - (int64_t) tp_start.tv_nsec;

    return s * 1000000000L + ns;
}

inline __m256 dft23_cmult4(__m256 a, __m256 b) {
    __m256 neg = _mm256_set_ps(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    __m256 c = _mm256_mul_ps(a, b); // a*c|b*d
    b = _mm256_permute_ps(b, 0b10110001); // Swapping real/imag; d|c
    __m256 d = _mm256_mul_ps(a, b); // a*d|b*c
    c = _mm256_mul_ps(c, neg); // a*c|-b*d
    return _mm256_permute_ps(_mm256_hadd_ps(c, d), 0b11011000);
}

inline void dft23_chadd4(__m256 a, float *out) {
        __m256 sum = _mm256_add_ps(a, _mm256_permute2f128_ps(a, a, 1));
        sum = _mm256_add_ps(sum, _mm256_permute_ps(sum, 0b00001110));
        _mm256_maskstore_ps(out, _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1), sum);
}

void dft23(complex float *buf, complex float *coeffs, complex float *out) {
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

void dft23_2(complex float *buf, complex float *out) {
    // np.exp(-1j * np.arange(23)/23 * 2*np.pi)
    const complex float k_rotators[DFT_SIZE+1] =
      { 1.f       +0.f          * I,  0.96291729f-0.26979677f * I,
        0.85441940f-0.51958395f * I,  0.68255314f-0.73083596f * I,
        0.46006504f-0.88788522f * I,  0.20345601f-0.97908409f * I,
       -0.06824241f-0.99766877f * I, -0.33487961f-0.94226092f * I,
       -0.57668032f-0.81696989f * I, -0.77571129f-0.63108794f * I,
       -0.91721130f-0.39840109f * I, -0.99068595f-0.13616665f * I,
       -0.99068595f+0.13616665f * I, -0.91721130f+0.39840109f * I,
       -0.77571129f+0.63108794f * I, -0.57668032f+0.81696989f * I,
       -0.33487961f+0.94226092f * I, -0.06824241f+0.99766877f * I,
        0.20345601f+0.97908409f * I,  0.46006504f+0.88788522f * I,
        0.68255314f+0.73083596f * I,  0.85441940f+0.51958395f * I,
        0.96291729f+0.26979677f * I,  0};

    for (int k = 0; k < DFT_SIZE; k += 4) {
        __m256 coeff = _mm256_set_ps(0, 1, 0, 1, 0, 1, 0, 1);

        __m256 rotators = _mm256_loadu_ps((float *) &k_rotators[k]);

        __m256 sum = _mm256_set1_ps(0);
        for (int n = 0; n < DFT_SIZE; n++) {
            __m256 sample = _mm256_maskload_ps((float *) &buf[n], _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1));
            sample = _mm256_permutevar8x32_ps(sample, _mm256_set_epi32(1, 0, 1, 0, 1, 0, 1, 0));

            sum = _mm256_add_ps(sum, dft23_cmult4(coeff, sample));
            coeff = dft23_cmult4(coeff, rotators);
        }

        _mm256_store_ps((float *) &out[k], sum);
    }
}

#define _PERM_MASK(a, b, c, d) ((d << 6) + (c << 4) + (b << 2) + a)

inline __m256 _mm256_neg_ps(__m256 x) {
    return _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(x),
                _mm256_set_epi32(1 << 31, 1 << 31, 1 << 31, 1 << 31, 1 << 31, 1 << 31, 1 << 31, 1 << 31)));
}

void dft23_3(complex float *buf, float *coeffs, complex float *out) {
    __m256 b_0, b_1, b_r, b_i, c_r, c_i, s_r, s_i;

    for (int k = 0; k < DFT_SIZE; k++) {
        s_r = _mm256_set1_ps(0);
        s_i = _mm256_set1_ps(0);

        for (int j = 0; j < 3; j++) {
            b_0 = _mm256_loadu_ps((float *) &buf[8*j]);
            b_1 = _mm256_loadu_ps((float *) &buf[8*j+4]);

            b_r = _mm256_shuffle_ps(b_0, b_1, _PERM_MASK(0, 2, 0, 2));
            b_i = _mm256_shuffle_ps(b_0, b_1, _PERM_MASK(1, 3, 1, 3));
            c_r = _mm256_loadu_ps(&coeffs[48*k+j*16]);
            c_i = _mm256_loadu_ps(&coeffs[48*k+j*16+8]);

            s_r = _mm256_add_ps(s_r, _mm256_fmsub_ps(b_r, c_r, _mm256_mul_ps(b_i, c_i)));
            s_i = _mm256_add_ps(s_i, _mm256_fmadd_ps(b_r, c_i, _mm256_mul_ps(b_i, c_r)));
        }

        s_r = _mm256_hadd_ps(s_r, _mm256_permute2f128_ps(s_r, s_r, 1));
        s_i = _mm256_hadd_ps(s_i, _mm256_permute2f128_ps(s_i, s_i, 1));
        s_r = _mm256_hadd_ps(s_r, s_r);
        s_i = _mm256_hadd_ps(s_i, s_i);
        out[k] = _mm256_cvtss_f32(_mm256_hadd_ps(s_r, s_r)) + I * _mm256_cvtss_f32(_mm256_hadd_ps(s_i, s_i));
    }
}

void gen_dft23_coeffs(complex float *coeffs) {
    for (int k = 0; k < DFT_SIZE; k++) {
        for (int n = 0; n < DFT_SIZE; n++)
            coeffs[24*k+n] = cos(2 * M_PI / DFT_SIZE * k * n) - I * sin(2 * M_PI / DFT_SIZE * k * n);
        coeffs[24*k+DFT_SIZE] = 0;
    }
}

inline size_t coeff_index_gen(size_t k, size_t n, bool imag) {
    size_t slot = n & ~0x7UL;
    size_t i = n & 0x7UL;
    const size_t IDX_TR[8] = {0, 1, 4, 5, 2, 3, 6, 7};
    return 2*(24*k + slot) + IDX_TR[i] + imag*8;
}

void gen_dft23_coeffs_separated(float *coeffs) {
    for (int k = 0; k < DFT_SIZE; k++) {
        for (int n = 0; n < DFT_SIZE; n++) {
            coeffs[coeff_index_gen(k, n, false)] = cos(2 * M_PI / DFT_SIZE * k * n);
            coeffs[coeff_index_gen(k, n, true)] = -sin(2 * M_PI / DFT_SIZE * k * n);
        }
        coeffs[coeff_index_gen(k, DFT_SIZE, false)] = 0;
        coeffs[coeff_index_gen(k, DFT_SIZE, true)] = 0;
    }
}

int64_t bench_dft23(complex float *buf, size_t len, int algorithm) {
    struct timespec tp_start; // tv_sec, tv_nsec
    struct timespec tp_end; // tv_sec, tv_nsec

    complex float *coeffs = fftwf_malloc(sizeof(complex float) * 24 * DFT_SIZE);
    float *coeffs_separated = fftwf_malloc(sizeof(complex float) * 24 * DFT_SIZE);
    complex float sum = 0;

    complex float *out = fftwf_malloc(sizeof(complex float) * 24);

    gen_dft23_coeffs(coeffs);
    gen_dft23_coeffs_separated(coeffs_separated);

    clock_gettime(CLOCK_MONOTONIC, &tp_start);

    for (ssize_t i = 0; i < ((ssize_t) (len / DFT_SIZE)); i++) {
        switch (algorithm) {
            case 0:
                dft23(&buf[DFT_SIZE*i], coeffs, out);
                break;
            case 1:
                dft23_2(&buf[DFT_SIZE*i], out);
                break;
            case 2:
                dft23_3(&buf[DFT_SIZE*i], coeffs_separated, out);
                break;
        }

        for (int j = 0; j < DFT_SIZE; j++)
            sum += out[j];
    }

    clock_gettime(CLOCK_MONOTONIC, &tp_end);

    fftwf_free(out);
    fftwf_free(coeffs);
    fftwf_free(coeffs_separated);

    printf("%f + %fj\n", creal(sum), cimag(sum));

    int64_t s = tp_end.tv_sec - tp_start.tv_sec;
    int64_t ns = (int64_t) tp_end.tv_nsec - (int64_t) tp_start.tv_nsec;

    return s * 1000000000L + ns;
}

int main(int argc, char *argv[]) {
    xoroshiro128plus_init(&xoro_state, 0xCAFEBABE8BADBEEFUL);

    xoroshiro128plus_next(&xoro_state);

    size_t len = 1024 * 1024 * 32;
    if (argc > 1 && sscanf(argv[1], "%lu", &len) != 1) {
        fprintf(stderr, "%s: Invalid bufer size: %s\n", argv[0], argv[1]);
        return EXIT_FAILURE;
    }

    int which = 0xFF;
    if (argc > 2 && sscanf(argv[2], "%d", &which) != 1) {
        fprintf(stderr, "%s: Invalid selector id: %s\n", argv[0], argv[2]);
        return EXIT_FAILURE;
    }

    complex float *samples = malloc(sizeof(complex float) * len);
    prepare_sample_sequence(samples, len);

    // dump_to_file(samples, len);

    int64_t duration;
    if (which & 0x1) {
        duration = bench_fftw(samples, len);
        printf("FFTW:   Duration: %ld ns, Throughput: %f MS/s\n", duration, len / ((double)duration) * 1e3);
    }

    if (which & 0x2) {
        duration = bench_dft23(samples, len, 0);
        printf("DFT:    Duration: %ld ns, Throughput: %f MS/s\n", duration, len / ((double)duration) * 1e3);
    }

    if (which & 0x4) {
        duration = bench_dft23(samples, len, 1);
        printf("DFT:    Duration: %ld ns, Throughput: %f MS/s\n", duration, len / ((double)duration) * 1e3);
    }

    if (which & 0x8) {
        duration = bench_dft23(samples, len, 2);
        printf("DFT:    Duration: %ld ns, Throughput: %f MS/s\n", duration, len / ((double)duration) * 1e3);
    }

    // Teardown
    free(samples);

    return EXIT_SUCCESS;
}
