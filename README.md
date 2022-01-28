# Benchmarking a 23-DFT

This code was created because i wanted to benchmark FFTW against my own (naive) AVX-2 implementation of a 23-point DFT. After iterating through a couple different approaches, i've managed to beat FFTW!:

```
$ ./dft_bench $((1024 * 1024 * 32))
5234866.500000 + 2819.280029j
FFTW:   Duration: 399798919 ns, Throughput: 167.856542 MS/s
5234867.500000 + 2819.287842j
DFT:    Duration: 559902033 ns, Throughput: 119.858225 MS/s
5234867.000000 + 2819.321777j
DFT:    Duration: 1509244166 ns, Throughput: 44.465213 MS/s
5234867.500000 + 2819.275146j
DFT:    Duration: 279784284 ns, Throughput: 239.859305 MS/s
```

