#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];

    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        printf("Error opening dataset file.\n");
        return 1;
    }

    long long N = 0;
    double value;

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

    for (long long i = 0; i < N; i++) {
        fscanf(file, "%lf", &data[i]);
    }

    fclose(file);

    double sum = 0.0;

    clock_t start = clock();

    for (long long i = 0; i < N; i++) {
        sum += data[i];
    }

    clock_t end = clock();

    double execution_time =
        (double)(end - start) / CLOCKS_PER_SEC;

    printf("Sequential Dataset Sum\n");
    printf("----------------------\n");
    printf("Dataset            : %s\n", filename);
    printf("Number of elements : %lld\n", N);
    printf("Sum                : %.2f\n", sum);
    printf("Execution time     : %.6f seconds\n",
           execution_time);

    free(data);

    return 0;
}
