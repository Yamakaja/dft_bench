# Benchmarking a 23-DFT

This code was created because i wanted to benchmark fftw against my own (naive) AVX-2 implementation of a 23-point DFT. Though i haven't been able to match fftw just yet:

```
$ build/dft_bench $((32 * 1024 * 1024))
1049859.625000 + 0.000120j
FFTW:   Duration: 208327497 ns, Throughput: 161.065786 MS/s
1049859.625000 + -0.000202j
DFT:    Duration: 306784595 ns, Throughput: 109.374566 MS/s
```

