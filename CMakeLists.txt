cmake_minimum_required(VERSION 3.10)

# set the project name
project(dft_bench)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -march=native -mtune=native -O3 -mavx2 -funroll-loops")

add_executable(dft_bench src/dft_bench.c src/xoroshiro128plus.c)
target_include_directories(dft_bench PUBLIC include)
target_link_libraries(dft_bench m fftw3f)

