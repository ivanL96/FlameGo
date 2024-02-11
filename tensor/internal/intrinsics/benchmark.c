#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>
#include <avx_float.h>

uint64_t current_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000 + (uint64_t)ts.tv_nsec;
}

int main() {
    const int64_t num_elements = 1000000;
    const float lr = 0.1f;

    float* val = (float*)malloc(num_elements * sizeof(float));
    float* grad = (float*)malloc(num_elements * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < num_elements; ++i) {
        val[i] = (float)rand() / RAND_MAX;
        grad[i] = (float)rand() / RAND_MAX;
    }

    clock_t start_time = current_time_ns();
    int repeat = 1000;
    for (int i = 0; i < repeat; ++i) {
        _mm256_gradientstep(val, grad, lr, num_elements);
    }
    clock_t end_time = current_time_ns();

    double elapsed_time = (double)(end_time - start_time);
    printf("Elapsed time: %.3f ns\n", elapsed_time / (double)repeat);

    free(val);
    free(grad);
    return 0;
}

// gcc -o bench benchmark.c src/avx_float.c -Isrc -mavx2 -O3 -fopenmp
// _mm256_gradientstep (1000000 elements) Elapsed time: 98399.300 ns