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

uint64_t xoshiro256p_next(uint64_t * s) {
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


uint64_t xoshiro256p(struct xoshiro256p_state *state) {
    uint64_t *s = state->s;
    return xoshiro256p_next(s);
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

/* This is the jump function for the generator. It is equivalent
 *    to 2^128 calls to next(); it can be used to generate 2^128
 *       non-overlapping subsequences for parallel computations. */
void xoshiro256p_jump(uint64_t * s) {
	static const uint64_t JUMP[] = { 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c };

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	uint64_t s2 = 0;
	uint64_t s3 = 0;
	for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++) {
		for(int b = 0; b < 64; b++) {
			if (JUMP[i] & UINT64_C(1) << b) {
				s0 ^= s[0];
				s1 ^= s[1];
				s2 ^= s[2];
				s3 ^= s[3];
			}
			xoshiro256p_next(s);	
		}
    }
    
	s[0] = s0;
	s[1] = s1;
	s[2] = s2;
	s[3] = s3;
}

/* This is the long-jump function for the generator. It is equivalent to
 *    2^192 calls to next(); it can be used to generate 2^64 starting points,
 *       from each of which jump() will generate 2^64 non-overlapping
 *          subsequences for parallel distributed computations. */

void xoshiro256p_long_jump(uint64_t * s) {
	static const uint64_t LONG_JUMP[] = { 0x76e15d3efefdcbbf, 0xc5004e441c522fb3, 0x77710069854ee241, 0x39109bb02acbe635 };

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	uint64_t s2 = 0;
	uint64_t s3 = 0;
	for(int i = 0; i < sizeof LONG_JUMP / sizeof *LONG_JUMP; i++) {
		for(int b = 0; b < 64; b++) {
			if (LONG_JUMP[i] & UINT64_C(1) << b) {
				s0 ^= s[0];
				s1 ^= s[1];
				s2 ^= s[2];
				s3 ^= s[3];
			}
			xoshiro256p_next(s);	
		}
	}
	s[0] = s0;
	s[1] = s1;
	s[2] = s2;
	s[3] = s3;
}

void xoshiro256p_copy_state(uint64_t * s_new, uint64_t * s_old) {
    for (int i=0; i<4; i++) {
        s_new[i] = s_old[i];
    }
}

static inline double to_double(uint64_t x) {
    const union { uint64_t i; double d; } u = { .i = UINT64_C(0x3FF) << 52 | x >> 12 };
    return u.d - 1.0;
}

static inline double to_double2(uint64_t x) {
    return (x >> 11) * 0x1.0p-53;
}
#endif
