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
    int offset = epoch * 8;
    for (int i = 0; i < remain; i++)
    {
        c[offset + i] = a[offset + i] + b[offset + i];
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
    int offset = epoch * 8;
    for (int i = 0; i < remain; i++)
    {
        c[offset + i] = a[offset + i] + b;
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
    int offset = epoch * 8;
    for (int i = 0; i < remain; i++)
    {
        c[offset + i] = a[offset + i] * b[offset + i];
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
    int offset = epoch * 8;
    for (int i = 0; i < remain; i++)
    {
        c[offset + i] = a[offset + i] * b;
    }
}

void _mm256_sub_to(float *a, float *b, float *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++)
    {
        __m256 v1 = _mm256_loadu_ps(a + i * 8);
        __m256 v2 = _mm256_loadu_ps(b + i * 8);
        __m256 v = _mm256_sub_ps(v1, v2);
        _mm256_storeu_ps(c + i * 8, v);
    }
    int offset = epoch * 8;
    for (int i = 0; i < remain; i++)
    {
        c[offset + i] = a[offset + i] - b[offset + i];
    }
}

void _mm256_sub_to_const_a(float a, float *b, float *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;
    __m256 v1 = _mm256_set1_ps(a);

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++)
    {
        __m256 v2 = _mm256_loadu_ps(b + i * 8);
        __m256 v = _mm256_sub_ps(v1, v2);
        _mm256_storeu_ps(c + i * 8, v);
    }
    int offset = epoch * 8;
    for (int i = 0; i < remain; i++)
    {
        c[offset + i] = a - b[offset + i];
    }
}

void _mm256_sub_to_const_b(float *a, float b, float *c, int64_t n)
{
    int epoch = n / 8;
    int remain = n % 8;
    __m256 v2 = _mm256_set1_ps(b);

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++)
    {
        __m256 v1 = _mm256_loadu_ps(a + i * 8);
        __m256 v = _mm256_sub_ps(v1, v2);
        _mm256_storeu_ps(c + i * 8, v);
    }
    int offset = epoch * 8;
    for (int i = 0; i < remain; i++)
    {
        c[offset + i] = a[offset + i] - b;
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
//     for (int i = 0; i < remain; i++){
//         c[i] = powf(a, b[i]);
//     }
// }

void _mm256_relu(float* a, float* c, int64_t n) {
    int epoch = n / 8;
    int remain = n % 8;

    __m256 zero = _mm256_set1_ps(0.0f);

    #pragma omp parallel for
    for (int i = 0; i < epoch; i++){
        __m256 v1 = _mm256_loadu_ps(a + i * 8);
        __m256 v = _mm256_max_ps(v1, zero);
        _mm256_storeu_ps(c + i * 8, v);
    }

    int offset = epoch * 8;
    for (int i = 0; i < remain; ++i) {
        c[offset + i] = (a[offset + i] > 0.0f) ? a[offset + i] : 0.0f;
    }
}

void _mm256_gradientstep(float* val, float* grad, float lr, int64_t n) {

    int epoch = n / 8;
    int remain = n % 8;

    __m256 LR = _mm256_set1_ps(lr);

    for (int i = 0; i < epoch; i++){
        __m256 VAL = _mm256_loadu_ps(val + i * 8);
        __m256 GRAD = _mm256_loadu_ps(grad + i * 8);
        __m256 step = _mm256_mul_ps(GRAD, LR);
        _mm256_storeu_ps(val + i * 8, _mm256_sub_ps(VAL, step));
    }

    int offset = epoch * 8;
    for (int i = 0; i < remain; i++){
        val[offset + i] -= grad[offset + i] * lr;
    }
}

