# Sparse Matrix-Vector Multiplication using CSR and CUDA

## 1. Objective
To implement Sparse Matrix-Vector Multiplication (SpMV) using Compressed Sparse Row (CSR) format and compare CPU, Scalar CSR and Vector CSR implementations.

## 2. CSR Representation
CSR stores a sparse matrix using three arrays:

- **Values:** Stores non-zero elements.
- **Column Index:** Stores the column positions.
- **Row Pointer:** Stores the starting position of each row.

Example matrix:

    10  0  0  20
     0 30  0   0
    40  0 50   0
     0  0  0  60

CSR arrays:

- Row Pointer: {0, 2, 3, 5, 6}
- Column Index: {0, 3, 1, 0, 2, 3}
- Values: {10, 20, 30, 40, 50, 60}

Input vector: {1, 2, 3, 4}

Expected output: {90, 60, 190, 240}

## 3. Implementations

### CPU SpMV
Performs matrix-vector multiplication sequentially on the CPU.

### Scalar CSR
One CUDA thread processes one matrix row.

### Vector CSR
One CUDA warp (32 threads) processes one matrix row using parallel reduction.

## 4. Performance Results

| Implementation | Average Time |
|---|---:|
| CPU | 0.00001796 ms |
| Scalar CSR | 0.00820792 ms |
| Vector CSR | 0.00677404 ms |

Note: Results are for a small demonstration matrix. GPU kernel launch overhead affects the measured times.

## 5. Load Imbalance Analysis

An unequal number of non-zero elements in different rows causes load imbalance.

Test row lengths: 1, 5, 1, 3.

Scalar and Vector CSR produced the same output:

{10, 55, 60, 44}

## 6. Compilation and Execution

Compile the main program:

    nvcc -arch=sm_120 src\sparse_matvec.cu -o spmv.exe

Run:

    spmv.exe

Compile the load imbalance program:

    nvcc -arch=sm_120 src\load_imbalance.cu -o imbalance.exe

Run:

    imbalance.exe

## 7. Conclusion

Sparse Matrix-Vector Multiplication was implemented successfully using CSR format. CPU, Scalar CSR and Vector CSR produced matching results. The experiment demonstrated GPU parallel processing, performance measurement and load imbalance.
