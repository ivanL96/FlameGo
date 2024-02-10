#include <stdint.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

void _mm256_add_to(float *a, float *b, float *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++)
    {
        __m256 v1 = _mm256_loadu_ps(a + i * 8);
        __m256 v2 = _mm256_loadu_ps(b + i * 8);
        __m256 v = _mm256_add_ps(v1, v2);
        _mm256_storeu_ps(c + i * 8, v);
    }
    for (int i = 0; i < remain; i++)
    {
        c[epoch * 8 + i] = a[epoch * 8 + i] + b[epoch * 8 + i];
    }
}

void _mm256_add_to_const(float *a, float b, float *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;
    __m256 v2 = _mm256_set1_ps(b);

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++)
    {
        __m256 v1 = _mm256_loadu_ps(a + i * 8);
        __m256 v = _mm256_add_ps(v1, v2);
        _mm256_storeu_ps(c + i * 8, v);
    }
    for (int i = 0; i < remain; i++)
    {
        c[epoch * 8 + i] = a[epoch * 8 + i] + b;
    }
}

void _mm256_mul_to(float *a, float *b, float *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++)
    {
        __m256 v1 = _mm256_loadu_ps(a + i * 8);
        __m256 v2 = _mm256_loadu_ps(b + i * 8);
        __m256 v = _mm256_mul_ps(v1, v2);
        _mm256_storeu_ps(c + i * 8, v);
    }
    for (int i = 0; i < remain; i++)
    {
        c[epoch * 8 + i] = a[epoch * 8 + i] * b[epoch * 8 + i];
    }
}

void _mm256_mul_to_const(float *a, float b, float *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;
    __m256 v2 = _mm256_set1_ps(b);

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++)
    {
        __m256 v1 = _mm256_loadu_ps(a + i * 8);
        __m256 v = _mm256_mul_ps(v1, v2);
        _mm256_storeu_ps(c + i * 8, v);
    }
    for (int i = 0; i < remain; i++)
    {
        c[epoch * 8 + i] = a[epoch * 8 + i] * b;
    }
}

void _mm256_sub_to(float *a, float *b, float *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;
    for (int i = 0; i < epoch; i++)
    {
        __m256 v1 = _mm256_loadu_ps(a);
        __m256 v2 = _mm256_loadu_ps(b);
        __m256 v = _mm256_sub_ps(v1, v2);
        _mm256_storeu_ps(c, v);
        a += 8;
        b += 8;
        c += 8;
    }
    for (int i = 0; i < remain; i++)
    {
        c[i] = a[i] - b[i];
    }
}

void _mm256_sub_to_const(float *a, float b, float *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;
    for (int i = 0; i < epoch; i++)
    {
        __m256 v1 = _mm256_loadu_ps(a);
        __m256 v2 = _mm256_set1_ps(b);
        __m256 v = _mm256_sub_ps(v1, v2);
        _mm256_storeu_ps(c, v);
        a += 8;
        c += 8;
    }
    for (int i = 0; i < remain; i++)
    {
        c[i] = a[i] - b;
    }
}

void _mm256_div_to(float *a, float *b, float *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;
    for (int i = 0; i < epoch; i++)
    {
        __m256 v1 = _mm256_loadu_ps(a);
        __m256 v2 = _mm256_loadu_ps(b);
        __m256 v = _mm256_div_ps(v1, v2);
        _mm256_storeu_ps(c, v);
        a += 8;
        b += 8;
        c += 8;
    }
    for (int i = 0; i < remain; i++)
    {
        c[i] = a[i] / b[i];
    }
}

// C._mm256_div_to_const_b((*C.float)(&af[0]), C.float(bf[0]), (*C.float)(&cf[0]), C.longlong(len(a)))
void _mm256_div_to_const_b(float *a, float b, float *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;
    for (int i = 0; i < epoch; i++)
    {
        __m256 v1 = _mm256_loadu_ps(a);
        __m256 v2 = _mm256_set1_ps(b);
        __m256 v = _mm256_div_ps(v1, v2);
        _mm256_storeu_ps(c, v);
        a += 8;
        c += 8;
    }
    for (int i = 0; i < remain; i++)
    {
        c[i] = a[i] / b;
    }
}

void _mm256_div_to_const_a(float a, float *b, float *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;
    for (int i = 0; i < epoch; i++)
    {
        __m256 v1 = _mm256_set1_ps(a);
        __m256 v2 = _mm256_loadu_ps(b);
        __m256 v = _mm256_div_ps(v1, v2);
        _mm256_storeu_ps(c, v);
        b += 8;
        c += 8;
    }
    for (int i = 0; i < remain; i++)
    {
        c[i] = a / b[i];
    }
}


// void _mm256_pow_to(float *a, float *b, float *c, int64_t n)
// {
//     int epoch = n / 8;
//     int remain = n % 8;
//     for (int i = 0; i < epoch; i++)
//     {
//         __m256 v1 = _mm256_loadu_ps(a);
//         __m256 v2 = _mm256_loadu_ps(b);
//         __m256 v = _mm256_pow_ps(v1, v2);
//         _mm256_storeu_ps(c, v);
//         a += 8;
//         b += 8;
//         c += 8;
//     }
//     for (int i = 0; i < remain; i++)
//     {
//         c[i] = powf(a[i], b[i]);
//     }
// }

// void _mm256_pow_to_const_b(float *a, float b, float *c, int64_t n)
// {
//     int epoch = n / 8;
//     int remain = n % 8;
//     for (int i = 0; i < epoch; i++)
//     {
//         __m256 v1 = _mm256_loadu_ps(a);
//         __m256 v2 = _mm256_set1_ps(b);
//         __m256 v = _mm256_pow_ps(v1, v2);
//         _mm256_storeu_ps(c, v);
//         a += 8;
//         c += 8;
//     }
//     for (int i = 0; i < remain; i++)
//     {
//         c[i] = powf(a[i], b);
//     }
// }

// void _mm256_pow_to_const_a(float a, float *b, float *c, int64_t n)
// {
//     int epoch = n / 8;
//     int remain = n % 8;
//     for (int i = 0; i < epoch; i++)
//     {
//         __m256 v1 = _mm256_set1_ps(a);
//         __m256 v2 = _mm256_loadu_ps(b);
//         __m256 v = _mm256_pow_ps(v1, v2);
//         _mm256_storeu_ps(c, v);
//         b += 8;
//         c += 8;
//     }
//     for (int i = 0; i < remain; i++)
//     {
//         c[i] = powf(a, b[i]);
//     }
// }

void _mm256_neg_to(float *a, float *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;
    for (int i = 0; i < epoch; i++)
    {
        __m256 v1 = _mm256_loadu_ps(a);
        __m256 v2 = _mm256_set1_ps(-1);
        __m256 v = _mm256_mul_ps(v1, v2);
        _mm256_storeu_ps(c, v);
        a += 8;
        c += 8;
    }
    for (int i = 0; i < remain; i++)
    {
        c[i] = -a[i];
    }
}
