# Benchmarking a 23-DFT

This code was created because i wanted to benchmark FFTW against my own (naive) AVX-2 implementation of a 23-point DFT. After iterating through a couple different approaches, i've managed to beat FFTW!:

```
$ ./dft_bench $((1024 * 1024 * 32))
1049933.000000 + 1069.911133j
FFTW:   Duration: 198184545 ns, Throughput: 169.309025 MS/s
1049933.000000 + 1069.924072j
DFT:    Duration: 282934895 ns, Throughput: 118.594180 MS/s
1049933.000000 + 1069.945679j
DFT:    Duration: 742166532 ns, Throughput: 45.211459 MS/s
1049932.875000 + 1069.913696j
DFT:    Duration: 187973120 ns, Throughput: 178.506544 MS/s
```

