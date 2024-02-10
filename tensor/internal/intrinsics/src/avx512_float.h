#ifndef AVX512_H
#define AVX512_H
#include <stdint.h>

void _mm512_mul_to(float *a, float *b, float *c, int64_t n);
void _mm512_mul_to_const(float *a, float b, float *c, int64_t n);

#endif
