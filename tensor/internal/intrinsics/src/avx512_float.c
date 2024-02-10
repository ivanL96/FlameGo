#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

void _mm512_mul_to(float *a, float *b, float *c, int64_t n)
{
    int epoch = n / 16;
    int remain = n % 16;

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++)
    {
        __m512 v1 = _mm512_loadu_ps(a + i * 16);
        __m512 v2 = _mm512_loadu_ps(b + i * 16);
        __m512 v = _mm512_mul_ps(v1, v2);
        _mm512_storeu_ps(c + i * 16, v);
    }
    int offset = epoch * 16;
    for (int i = 0; i < remain; i++)
    {
        c[offset + i] = a[offset + i] * b[offset + i];
    }
}

void _mm512_mul_to_const(float *a, float b, float *c, int64_t n)
{
    int epoch = n / 16;
    int remain = n % 16;
    __m512 v2 = _mm512_set1_ps(b);

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++)
    {
        __m512 v1 = _mm512_loadu_ps(a + i * 16);
        __m512 v = _mm512_mul_ps(v1, v2);
        _mm512_storeu_ps(c + i * 16, v);
    }
    int offset = epoch * 16;
    for (int i = 0; i < remain; i++)
    {
        c[offset + i] = a[offset + i] * b;
    }
}
