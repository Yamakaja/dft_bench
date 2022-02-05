# Benchmarking a 23-DFT

This code was created because i wanted to benchmark FFTW against my own (naive) AVX-2 implementation of a 23-point DFT. After iterating through a couple different approaches, i've managed to beat FFTW!:

```
$ ./dft_bench $((1024 * 1024 * 32))
FFTW:   SUM(X_k) = 1049933.000000 + 1069.911133j
FFTW:   Duration: 194307987 ns, Throughput: 172.686839 MS/s

DFTv1:  SUM(X_k) = 1049933.000000 + 1069.924072j
DFTv1:  Duration: 273867358 ns, Throughput: 122.520742 MS/s

DFTv2:  SUM(X_k) = 1049933.000000 + 1069.945679j
DFTv2:  Duration: 728436320 ns, Throughput: 46.063645 MS/s

DFTv3:  SUM(X_k) = 1049932.875000 + 1069.911133j
DFTv3:  Duration: 137370308 ns, Throughput: 244.262625 MS/s

DFTv4:  SUM(X_k) = 1049932.750000 + 1069.919922j
DFTv4:  Duration: 78593301 ns, Throughput: 426.937558 MS/s
```

