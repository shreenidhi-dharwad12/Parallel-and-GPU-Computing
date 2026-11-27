# OpenMP vs MPI for Data Processing

---

## Aim

To perform the same dataset operation using OpenMP and MPI and compare their execution time, speedup, and efficiency.

## Problem Definition

Large datasets may require significant processing time when processed sequentially. Parallel computing can divide the workload among multiple workers and improve processing performance.

In this experiment, the operation selected is **finding the sum of all numerical values in a dataset**.

The same operation is implemented using:

* Sequential execution
* OpenMP
* MPI

The implementations are tested with different dataset sizes and different numbers of threads/processes.

## Objectives

1. Implement sequential dataset summation.
2. Implement dataset summation using OpenMP.
3. Implement dataset summation using MPI.
4. Test different dataset sizes.
5. Test different numbers of threads and processes.
6. Measure execution time.
7. Calculate speedup and efficiency.
8. Generate performance graphs.
9. Compare OpenMP and MPI.

## Experimental Environment

| Component            | Details            |
| -------------------- | ------------------ |
| Operating System     | Ubuntu 24.04.1 LTS |
| Environment          | WSL2               |
| Architecture         | x86_64             |
| Compiler             | GCC 13.3.0         |
| MPI                  | Open MPI 4.1.6     |
| Programming Language | C                  |
| Graph Tool           | Python Matplotlib  |

## Dataset

Three datasets were used:

| Dataset          | Number of Elements |
| ---------------- | -----------------: |
| dataset_100k.txt |            100,000 |
| dataset_1m.txt   |          1,000,000 |
| dataset_5m.txt   |          5,000,000 |

The same datasets are used for Sequential, OpenMP, and MPI to provide a fair comparison.

## Sequential Approach

The sequential program reads the dataset and calculates the sum using a single execution flow.

### Algorithm

1. Read the input dataset.
2. Store the numerical values in memory.
3. Initialize the sum to zero.
4. Traverse all elements sequentially.
5. Add each element to the sum.
6. Display the final sum and execution time.

### Source File

```text
src/sequential_sum.c
```

### Compilation

```bash
gcc -O2 src/sequential_sum.c -o src/sequential_sum
```

### Execution

```bash
./src/sequential_sum data/dataset_5m.txt
```

## OpenMP Approach

OpenMP is used to parallelize the summation loop.

Multiple threads process different portions of the dataset. The `reduction` clause is used to safely combine the partial sums.

```c
#pragma omp parallel for reduction(+:sum)
```

### Source File

```text
src/openmp_sum.c
```

### Compilation

```bash
gcc -O2 -fopenmp src/openmp_sum.c -o src/openmp_sum
```

### Execution

```bash
./src/openmp_sum data/dataset_5m.txt 1
./src/openmp_sum data/dataset_5m.txt 2
./src/openmp_sum data/dataset_5m.txt 4
./src/openmp_sum data/dataset_5m.txt 8
```

## MPI Approach

MPI divides the dataset among multiple processes.

The root process reads the dataset and distributes the data using `MPI_Scatterv`. Each process calculates its local sum, and `MPI_Reduce` combines the partial sums into the final result.

### Main MPI Functions

* `MPI_Init()`
* `MPI_Comm_rank()`
* `MPI_Comm_size()`
* `MPI_Bcast()`
* `MPI_Scatterv()`
* `MPI_Reduce()`
* `MPI_Finalize()`

### Source File

```text
src/mpi_sum.c
```

### Compilation

```bash
mpicc -O2 src/mpi_sum.c -o src/mpi_sum
```

### Execution

```bash
mpirun -np 1 ./src/mpi_sum data/dataset_5m.txt
mpirun -np 2 ./src/mpi_sum data/dataset_5m.txt
mpirun -np 4 ./src/mpi_sum data/dataset_5m.txt
mpirun -np 8 ./src/mpi_sum data/dataset_5m.txt
```

## Experimental Configurations

### Dataset Sizes

* 100K
* 1M
* 5M

### OpenMP Threads

* 1
* 2
* 4
* 8

### MPI Processes

* 1
* 2
* 4
* 8

## Results

The complete experimental results are available in:

```text
results/results.csv
```

Performance calculations are available in:

```text
results/performance.csv
```

## Performance Metrics

### Execution Time

Execution time is the time required to perform the dataset summation operation.

### Speedup

```text
Speedup = Sequential Execution Time / Parallel Execution Time
```

### Efficiency

```text
Efficiency = Speedup / Number of Workers
```

## 5M Dataset Results

| Method     | Workers | Execution Time (s) |    Speedup | Efficiency |
| ---------- | ------: | -----------------: | ---------: | ---------: |
| Sequential |       1 |           0.002510 |     1.0000 |     1.0000 |
| OpenMP     |       1 |           0.005395 |     0.4652 |     0.4652 |
| OpenMP     |       2 |       **0.002895** | **0.8670** | **0.4335** |
| OpenMP     |       4 |           0.023842 |     0.1053 |     0.0263 |
| OpenMP     |       8 |           0.054147 |     0.0464 |     0.0058 |
| MPI        |       1 |           0.027350 |     0.0918 |     0.0918 |
| MPI        |       2 |           0.026673 |     0.0941 |     0.0471 |
| MPI        |       4 |       **0.026183** | **0.0959** | **0.0240** |
| MPI        |       8 |           0.026661 |     0.0941 |     0.0118 |

## Observations

### OpenMP

OpenMP achieved its best observed performance with **2 threads** for the 5M dataset.

Increasing the number of threads beyond 2 increased the execution time because the overhead of parallel execution became significant for this simple operation.

### MPI

MPI achieved its lowest measured execution time with **4 processes** for the 5M dataset.

However, the improvement was small because MPI introduces communication overhead while distributing data and combining partial results.

### Overall Comparison

For this experiment, **OpenMP performed better than MPI**.

OpenMP uses shared memory and has lower communication overhead, whereas MPI requires communication between processes.

The experiment also shows that increasing the number of workers does not always improve performance.

## Graphs

All generated graphs are stored in the `graphs/` directory.

### Execution Time

```text
graphs/execution_time_100K.png
graphs/execution_time_1M.png
graphs/execution_time_5M.png
```

### Speedup

```text
graphs/speedup_100K.png
graphs/speedup_1M.png
graphs/speedup_5M.png
```

### Efficiency

```text
graphs/efficiency_100K.png
graphs/efficiency_1M.png
graphs/efficiency_5M.png
```

Graphs can be regenerated using:

```bash
python3 src/generate_graphs.py
```

## Project Structure

```text
Lab-evaluation-openmp-vs-mpi/
│
├── README.md
├── data/
│   ├── dataset_100k.txt
│   ├── dataset_1m.txt
│   └── dataset_5m.txt
│
├── graphs/
│   ├── execution_time_100K.png
│   ├── execution_time_1M.png
│   ├── execution_time_5M.png
│   ├── speedup_100K.png
│   ├── speedup_1M.png
│   ├── speedup_5M.png
│   ├── efficiency_100K.png
│   ├── efficiency_1M.png
│   └── efficiency_5M.png
│
├── report/
├── results/
│   ├── results.csv
│   ├── performance.csv
│   └── screenshots/
│
└── src/
    ├── generate_data.c
    ├── sequential_sum.c
    ├── openmp_sum.c
    ├── mpi_sum.c
    ├── analyze_results.py
    └── generate_graphs.py
```

## Conclusion

The same dataset summation operation was successfully implemented using Sequential execution, OpenMP, and MPI.

The programs were tested with different dataset sizes and different numbers of threads/processes.

The results show that OpenMP performed better than MPI for this particular experiment. OpenMP achieved its best observed result with 2 threads for the 5M dataset, while MPI achieved its best observed result with 4 processes.

The experiment demonstrates that parallel computing performance depends on workload size and parallelization overhead. Simply increasing the number of threads or processes does not always result in better performance.


**Author:** Shreenidhi Dharwad
**Year:** 3rd Year B.E.
**College:** KLE Technological University, Hubballi