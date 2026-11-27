#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("Usage: %s <input_file> <num_threads>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    int num_threads = atoi(argv[2]);

    if (num_threads <= 0) {
        printf("Number of threads must be positive.\n");
        return 1;
    }

    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        printf("Error opening dataset file.\n");
        return 1;
    }

    long long N = 0;
    double value;

    /* Count number of elements */
    while (fscanf(file, "%lf", &value) == 1) {
        N++;
    }

    rewind(file);

    double *data = malloc(N * sizeof(double));

    if (data == NULL) {
        printf("Memory allocation failed.\n");
        fclose(file);
        return 1;
    }

    /* Read dataset */
    for (long long i = 0; i < N; i++) {
        fscanf(file, "%lf", &data[i]);
    }

    fclose(file);

    double sum = 0.0;

    omp_set_num_threads(num_threads);

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < N; i++) {
        sum += data[i];
    }

    double end = omp_get_wtime();

    printf("OpenMP Dataset Sum\n");
    printf("------------------\n");
    printf("Dataset            : %s\n", filename);
    printf("Number of elements : %lld\n", N);
    printf("Threads            : %d\n", num_threads);
    printf("Sum                : %.2f\n", sum);
    printf("Execution time     : %.6f seconds\n",
           end - start);

    free(data);

    return 0;
}
