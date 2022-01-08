#ifndef H_XOROSHIRO128PLUS
#define H_XOROSHIRO128PLUS

typedef struct xoroshiro128plus_t {
    uint64_t s[2]; // xoroshiro state
    uint64_t x; // Splitmix state
} xoroshiro128plus_t;

uint64_t xoroshiro128plus_next(xoroshiro128plus_t *);

void xoroshiro128plus_jump(xoroshiro128plus_t *);

uint64_t xoroshiro128plus_next(xoroshiro128plus_t *);

uint64_t splitmix64_next(xoroshiro128plus_t *);

void xoroshiro128plus_init(xoroshiro128plus_t *, uint64_t seed);

#endif
