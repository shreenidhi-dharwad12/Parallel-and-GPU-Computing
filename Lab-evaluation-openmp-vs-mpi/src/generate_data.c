#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {

    if (argc != 3) {
        printf("Usage: %s <number_of_elements> <output_file>\n", argv[0]);
        return 1;
    }

    long long N = atoll(argv[1]);
    const char *filename = argv[2];

    if (N <= 0) {
        printf("Number of elements must be positive.\n");
        return 1;
    }

    FILE *file = fopen(filename, "w");

    if (file == NULL) {
        printf("Error opening output file.\n");
        return 1;
    }

    for (long long i = 0; i < N; i++) {
        fprintf(file, "1.0\n");
    }

    fclose(file);

    printf("Dataset generated successfully.\n");
    printf("Number of elements : %lld\n", N);
    printf("Output file        : %s\n", filename);

    return 0;
}
