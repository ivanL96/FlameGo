#include <immintrin.h>
#include <stdint.h>

void _mm512_mul_to(float *a, float *b, float *c, int64_t n)
{
    int epoch = n / 16;
    int remain = n % 16;
    for (int i = 0; i < epoch; i++)
    {
        __m512 v1 = _mm512_loadu_ps(a);
        __m512 v2 = _mm512_loadu_ps(b);
        __m512 v = _mm512_mul_ps(v1, v2);
        _mm512_storeu_ps(c, v);
        a += 16;
        b += 16;
        c += 16;
    }
    for (int i = 0; i < remain; i++)
    {
        c[i] = a[i] * b[i];
    }
}
