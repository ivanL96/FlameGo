#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

void _mm256i_add_to(int32_t *a, int32_t *b, int32_t *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++)
    {
        __m256i v1 = _mm256_loadu_si256((__m256i *)&a[i * 8]);
        __m256i v2 = _mm256_loadu_si256((__m256i *)&b[i * 8]);
        __m256i v = _mm256_add_epi32(v1, v2);
        _mm256_storeu_si256((__m256i *)&c[i * 8], v);
    }

    for (int i = 0; i < remain; i++)
    {
        c[epoch * 8 + i] = a[epoch * 8 + i] + b[epoch * 8 + i];
    }
}

void _mm256i_add_to_const(int32_t *a, int32_t b, int32_t *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;
    __m256i v2 = _mm256_set1_epi32(b);

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++)
    {
        __m256i v1 = _mm256_loadu_si256((__m256i *)&a[i * 8]);
        __m256i v = _mm256_add_epi32(v1, v2);
        _mm256_storeu_si256((__m256i *)&c[i * 8], v);
    }

    for (int i = 0; i < remain; i++)
    {
        c[epoch * 8 + i] = a[epoch * 8 + i] + b;
    }
}


void _mm256i_mul_to(int32_t *a, int32_t *b, int32_t *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++)
    {
        __m256i v1 = _mm256_loadu_si256((__m256i *)&a[i * 8]);
        __m256i v2 = _mm256_loadu_si256((__m256i *)&b[i * 8]);
        __m256i v = _mm256_mul_epi32(v1, v2);
        _mm256_storeu_si256((__m256i *)&c[i * 8], v);
    }

    for (int i = epoch * 8; i < n; i++)
    {
        c[epoch * 8 + i] = a[epoch * 8 + i] * b[epoch * 8 + i];
    }
}

void _mm256i_mul_to_const(int32_t *a, int32_t b, int32_t *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;
    __m256i v2 = _mm256_set1_epi32(b);

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++)
    {
        __m256i v1 = _mm256_loadu_si256((__m256i*)a);
        __m256i v = _mm256_mul_epi32(v1, v2);
        _mm256_storeu_si256((__m256i*)c, v);
        a += 8;
        c += 8;
    }

    for (int i = 0; i < remain; i++)
    {
        c[epoch * 8 + i] = a[epoch * 8 + i] * b;
    }
}