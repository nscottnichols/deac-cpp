#ifndef COMMON_H 
#define COMMON_H
#include <stdint.h>

// See https://prng.di.unimi.it/ and https://en.wikipedia.org/wiki/Xorshift
struct splitmix64_state {
    uint64_t s;
};

uint64_t splitmix64(struct splitmix64_state *state) {
    uint64_t result = (state->s += 0x9E3779B97f4A7C15);
    result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9;
    result = (result ^ (result >> 27)) * 0x94D049BB133111EB;
    return result ^ (result >> 31);
}

uint64_t rol64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

struct xoshiro256p_state {
    uint64_t s[4];
};

uint64_t xoshiro256p(struct xoshiro256p_state *state) {
    uint64_t *s = state->s;
    uint64_t const result = s[0] + s[3];
    uint64_t const t = s[1] << 17;
    
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    
    s[2] ^= t;
    s[3] = rol64(s[3], 45);
    
    return result;
}

struct xoshiro256p_state xoshiro256p_init(uint64_t seed) {
    struct splitmix64_state smstate = {seed};
    struct xoshiro256p_state result = {0};
    
    uint64_t tmp = splitmix64(&smstate);
    result.s[0] = (uint32_t)tmp;
    result.s[1] = (uint32_t)(tmp >> 32);
    
    tmp = splitmix64(&smstate);
    result.s[2] = (uint32_t)tmp;
    result.s[3] = (uint32_t)(tmp >> 32);
    
    return result;
}

static inline double to_double(uint64_t x) {
    const union { uint64_t i; double d; } u = { .i = UINT64_C(0x3FF) << 52 | x >> 12 };
    return u.d - 1.0;
}

static inline double to_double2(uint64_t x) {
    return (x >> 11) * 0x1.0p-53;
}
#endif
