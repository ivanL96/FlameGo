#ifndef AVXi_H
#define AVXi_H
#include <stdint.h>

void _mm256i_add_to(int32_t *a, int32_t *b, int32_t *c, int64_t n);
void _mm256i_add_to_const(int32_t *a, int32_t b, int32_t *c, int64_t n);

void _mm256i_mul_to(int32_t *a, int32_t *b, int32_t *c, int64_t n);
void _mm256i_mul_to_const(int32_t *a, int32_t b, int32_t *c, int64_t n);

#endif
