#ifndef AVX_H
#define AVX_H
#include <stdint.h>

void _mm256_add_to(float *a, float *b, float *c, int64_t n);
void _mm256_add_to_const(float *a, float b, float *c, int64_t n);

void _mm256_mul_to(float *a, float *b, float *c, int64_t n);
void _mm256_mul_to_const(float *a, float b, float *c, int64_t n);

void _mm256_sub_to(float *a, float *b, float *c, int64_t n);
void _mm256_sub_to_const_a(float a, float *b, float *c, int64_t n);
void _mm256_sub_to_const_b(float *a, float b, float *c, int64_t n);

void _mm256_div_to(float *a, float *b, float *c, int64_t n);
void _mm256_div_to_const_b(float *a, float b, float *c, int64_t n);
void _mm256_div_to_const_a(float a, float *b, float *c, int64_t n);

// void _mm256_pow_to(float *a, float *b, float *c, int64_t n);
// void _mm256_pow_to_const_b(float *a, float b, float *c, int64_t n);
// void _mm256_pow_to_const_a(float a, float *b, float *c, int64_t n);

void _mm256_relu(float* a, float* c, int64_t n);

#endif