#define _POSIX_C_SOURCE 200112L

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 100000000LL
#define CACHE_LINE_SIZE 64

double run_test(int stride)
{
    long long *counters = NULL;
    size_t count = (size_t)stride * 2;
    size_t size = count * sizeof(long long);

    if (posix_memalign((void **)&counters, CACHE_LINE_SIZE, size) != 0) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < 2; i++)
        counters[i * stride] = 0;

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();

        for (long long i = 0; i < N / 2; i++)
            counters[tid * stride]++;
    }

    double end = omp_get_wtime();

    printf("Total count: %lld\n", counters[0] + counters[stride]);

    free(counters);
    return end - start;
}

int main(void)
{
    printf("FALSE SHARING EXPERIMENT\n");
    printf("------------------------\n");

    printf("\nAdjacent counters (false sharing):\n");
    double adjacent = run_test(1);
    printf("Time: %.6f seconds\n", adjacent);

    printf("\nPadded counters (reduced false sharing):\n");
    double padded = run_test(8);
    printf("Time: %.6f seconds\n", padded);

    printf("\nComparison:\n");
    printf("Adjacent: %.6f seconds\n", adjacent);
    printf("Padded:   %.6f seconds\n", padded);

    return 0;
}