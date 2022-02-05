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

#include "dft23.h"
#include "xoroshiro128plus.h"

#define SAMPLE_RATE 1000
#define SINE_FREQ   240
#define NOISE_AMPLITUDE 0.05

#define DEBUG_VALS false

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

        for (int j = 0; j < DFT_SIZE; j++) {
            if (DEBUG_VALS)
                printf("%d: %.3f + j%.4f\n", j, crealf(out[j]), cimagf(out[j]));

            sum += out[j];
        }

        if (DEBUG_VALS)
            break;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &tp_end);

    fftwf_destroy_plan(p);
    fftwf_free(in);
    fftwf_free(out);

    printf("FFTW:   SUM(X_k) = %f + %fj\n", creal(sum), cimag(sum));

    int64_t s = tp_end.tv_sec - tp_start.tv_sec;
    int64_t ns = (int64_t) tp_end.tv_nsec - (int64_t) tp_start.tv_nsec;

    return s * 1000000000L + ns;
}

inline __m256 _mm256_neg_ps(__m256 x) {
    return _mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(x),
                _mm256_set_epi32(1 << 31, 1 << 31, 1 << 31, 1 << 31, 1 << 31, 1 << 31, 1 << 31, 1 << 31)));
}

int64_t bench_dft23(complex float *buf, size_t len, int algorithm) {
    struct timespec tp_start; // tv_sec, tv_nsec
    struct timespec tp_end; // tv_sec, tv_nsec

    complex float *coeffs_1 = fftwf_malloc(sizeof(complex float) * 24 * DFT_SIZE);
    float *coeffs_3 = fftwf_malloc(sizeof(complex float) * 24 * DFT_SIZE);
    float *coeffs_4 = fftwf_malloc(sizeof(complex float) * 24 * DFT_SIZE);
    complex float sum = 0;

    complex float *out = fftwf_malloc(sizeof(complex float) * 24);

    gen_dft23_coeffs_1(coeffs_1);
    gen_dft23_coeffs_3(coeffs_3);
    gen_dft23_coeffs_4(coeffs_4);

    clock_gettime(CLOCK_MONOTONIC, &tp_start);

    for (ssize_t i = 0; i < ((ssize_t) (len / DFT_SIZE)); i++) {
        switch (algorithm) {
            case 0:
                dft23_1(&buf[DFT_SIZE*i], coeffs_1, out);
                break;
            case 1:
                dft23_2(&buf[DFT_SIZE*i], out);
                break;
            case 2:
                dft23_3(&buf[DFT_SIZE*i], coeffs_3, out);
                break;
            case 3:
                dft23_4(&buf[DFT_SIZE*i], coeffs_4, out);
                break;
        }

        for (int j = 0; j < DFT_SIZE; j++) {
            if (DEBUG_VALS)
                printf("%d: %.3f + j%.4f\n", j, crealf(out[j]), cimagf(out[j]));

            sum += out[j];
        }

        if (DEBUG_VALS)
            break;
    }

    clock_gettime(CLOCK_MONOTONIC, &tp_end);

    fftwf_free(out);
    fftwf_free(coeffs_1);
    fftwf_free(coeffs_3);
    fftwf_free(coeffs_4);

    printf("DFTv%d:  SUM(X_k) = %f + %fj\n", algorithm+1, creal(sum), cimag(sum));

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
        puts("");
    }

    for (int i = 1; i < 5; i++) {
        if (!((1 << i) & which))
            continue;
        duration = bench_dft23(samples, len, i-1);
        printf("DFTv%d:  Duration: %ld ns, Throughput: %f MS/s\n", i, duration, len / ((double)duration) * 1e3);
        puts("");
    }

    // Teardown
    free(samples);

    return EXIT_SUCCESS;
}
