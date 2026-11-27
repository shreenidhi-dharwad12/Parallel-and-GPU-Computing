#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <input_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char *filename = argv[1];

    double *data = NULL;
    long long N = 0;

    /* Root process reads the complete dataset */
    if (rank == 0) {

        FILE *file = fopen(filename, "r");

        if (file == NULL) {
            printf("Error opening dataset file.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double value;

        while (fscanf(file, "%lf", &value) == 1) {
            N++;
        }

        rewind(file);

        data = malloc(N * sizeof(double));

        if (data == NULL) {
            printf("Memory allocation failed.\n");
            fclose(file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (long long i = 0; i < N; i++) {
            fscanf(file, "%lf", &data[i]);
        }

        fclose(file);
    }

    /* Send dataset size to all processes */
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    /* Calculate number of elements for each process */
    long long base = N / size;
    long long remainder = N % size;

    long long local_N = base + (rank < remainder ? 1 : 0);

    double *local_data = malloc(local_N * sizeof(double));

    if (local_data == NULL) {
        printf("Process %d: Memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *counts = NULL;
    int *displacements = NULL;

    if (rank == 0) {

        counts = malloc(size * sizeof(int));
        displacements = malloc(size * sizeof(int));

        long long displacement = 0;

        for (int i = 0; i < size; i++) {
            long long count = base + (i < remainder ? 1 : 0);

            counts[i] = (int)count;
            displacements[i] = (int)displacement;

            displacement += count;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    /* Distribute dataset among processes */
    MPI_Scatterv(
        data,
        counts,
        displacements,
        MPI_DOUBLE,
        local_data,
        (int)local_N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    /* Calculate local sum */
    double local_sum = 0.0;

    for (long long i = 0; i < local_N; i++) {
        local_sum += local_data[i];
    }

    /* Combine all local sums */
    double global_sum = 0.0;

    MPI_Reduce(
        &local_sum,
        &global_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("MPI Dataset Sum\n");
        printf("----------------\n");
        printf("Dataset            : %s\n", filename);
        printf("Number of elements : %lld\n", N);
        printf("Processes          : %d\n", size);
        printf("Sum                : %.2f\n", global_sum);
        printf("Execution time     : %.6f seconds\n",
               end - start);
    }

    free(local_data);

    if (rank == 0) {
        free(data);
        free(counts);
        free(displacements);
    }

    MPI_Finalize();

    return 0;
}
