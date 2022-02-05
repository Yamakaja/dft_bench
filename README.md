# Benchmarking a 23-DFT

This code was created because i wanted to benchmark FFTW against my own (naive) AVX-2 implementation of a 23-point DFT. After iterating through a couple different approaches, i've managed to beat FFTW!:

Performance on `AMD Ryzen 9 3900X`:
```
$ ./dft_bench $((1024 * 1024 * 32))
FFTW:   Duration: 185233232 ns, Throughput: 181.146934 MS/s
DFTv1:  Duration: 274761021 ns, Throughput: 122.122242 MS/s
DFTv2:  Duration: 748271623 ns, Throughput: 44.842583 MS/s
DFTv3:  Duration: 129925410 ns, Throughput: 258.259197 MS/s
DFTv4:  Duration: 61371889 ns, Throughput: 546.739436 MS/s
```

