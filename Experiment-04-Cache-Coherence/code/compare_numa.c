#define _POSIX_C_SOURCE 200112L

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 100000000LL
#define CACHE_LINE_SIZE 64

double run_test(int threads, int stride)
{
    long long *counters = NULL;
    size_t count = (size_t)threads * stride;
    size_t size = count * sizeof(long long);

    if (posix_memalign((void **)&counters, CACHE_LINE_SIZE, size) != 0) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < threads; i++)
        counters[i * stride] = 0;

    double start = omp_get_wtime();

    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();

        for (long long i = 0; i < N / threads; i++)
            counters[tid * stride]++;
    }

    double end = omp_get_wtime();

    free(counters);
    return end - start;
}

int main(void)
{
    int thread_counts[] = {1, 2, 4};

    printf("NUMA AND FALSE SHARING COMPARISON\n");
    printf("---------------------------------\n");

    for (int i = 0; i < 3; i++) {
        int threads = thread_counts[i];

        double adjacent = run_test(threads, 1);
        double padded = run_test(threads, 8);

        printf("\nThreads: %d\n", threads);
        printf("Adjacent counters: %.6f seconds\n", adjacent);
        printf("Padded counters:   %.6f seconds\n", padded);
    }

    printf("\nNUMA comparison commands:\n");
    printf("numactl --interleave=all ./numa_sweep\n");
    printf("numactl --cpunodebind=0 --membind=0 ./numa_sweep\n");

    return 0;
}