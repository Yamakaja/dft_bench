#include <math.h>
#include <stdint.h>
#include <immintrin.h>

#include "dft23.h"

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
