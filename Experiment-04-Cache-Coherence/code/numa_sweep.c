
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define DEFAULT_SIZE_MB 512
#define CACHE_LINE_SIZE 64
#define PASSES 4

int main(int argc, char *argv[]) {
    size_t size_mb = DEFAULT_SIZE_MB;

    if (argc > 1) {
        size_mb = (size_t)atol(argv[1]);
    }

    if (size_mb == 0) {
        printf("Invalid memory size.\n");
        return 1;
    }

    size_t bytes = size_mb * 1024ULL * 1024ULL;

    unsigned char *buffer = malloc(bytes);

    if (buffer == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    // Initialize memory using first-touch placement.
    memset(buffer, 1, bytes);

    unsigned long long total_sum = 0;

    double start = omp_get_wtime();

    for (int pass = 0; pass < PASSES; pass++) {
        unsigned long long sum = 0;

        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (size_t i = 0; i < bytes; i += CACHE_LINE_SIZE) {
            sum += buffer[i];
        }

        total_sum += sum;
    }

    double end = omp_get_wtime();

    printf("\nNUMA MEMORY ACCESS EXPERIMENT\n");
    printf("-----------------------------\n");
    printf("Memory size       : %zu MB\n", size_mb);
    printf("Cache line size   : %d bytes\n", CACHE_LINE_SIZE);
    printf("Number of passes  : %d\n", PASSES);
    printf("Total sum         : %llu\n", total_sum);
    printf("Execution time    : %.6f seconds\n", end - start);

    printf("Experiment completed successfully.\n");

    free(buffer);

    return 0;
}
