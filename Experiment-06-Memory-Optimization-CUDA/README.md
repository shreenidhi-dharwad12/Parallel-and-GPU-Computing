# Experiment 06 - Memory Optimization in CUDA



## 1. Aim



To study how global memory access patterns affect CUDA GPU performance by comparing uncoalesced and coalesced matrix transpose operations and implementing naive matrix multiplication.



## 2. Learning Outcomes



- Understand memory coalescing and its importance in GPU performance.

- Implement 2D CUDA kernels using grids and blocks.

- Use shared memory to improve memory access patterns.

- Measure kernel execution time using CUDA events.

- Calculate memory bandwidth and floating-point performance.

- Verify GPU results using CPU reference calculations.



## 3. Theory



### 3.1 Memory Coalescing



A GPU executes threads in groups of 32 called warps. When threads in a warp access neighbouring global memory addresses, the GPU combines these accesses into fewer memory transactions. This is called memory coalescing.



When threads access scattered addresses, more memory transactions are required, reducing effective memory bandwidth.



| Feature | Coalesced Access | Uncoalesced Access |
|---|---|---|
| Memory access | Neighbouring addresses | Scattered addresses |
| Transactions | Fewer | More |
| Bandwidth usage | More efficient | Less efficient |



### 3.2 Matrix Transpose



Matrix transpose changes the rows of a matrix into columns.



For an input matrix, the transpose operation is:



`out[j][i] = in[i][j]`



In row-major storage, an element at row `r` and column `c` is stored at:



`index = r * N + c`



**Uncoalesced Transpose**



- Each CUDA thread copies one matrix element.

- Reads are performed in neighbouring memory locations.

- Writes are separated by a stride of N.

- This causes inefficient global memory access.



**Coalesced Transpose**



- Uses shared memory to store a 32 x 32 tile.

- Threads load data using coalesced reads.

- The tile is synchronized using `__syncthreads()`.

- Data is written in transposed order using coalesced writes.

- An extra column in shared memory helps avoid bank conflicts.



### 3.3 Naive Matrix Multiplication



Matrix multiplication calculates:



`C = A x B`



For two N x N matrices:



`C[i][j] = sum(A[i][k] * B[k][j])`



In the naive CUDA kernel:



- One thread calculates one output element.

- Each thread performs N iterations.

- Each iteration reads one element from A and one from B.

- The result is stored in matrix C.

- Data is repeatedly fetched from global memory because shared-memory reuse is not implemented.



### 3.4 Performance Formulas



| Measurement | Formula |
|---|---|
| Transpose bandwidth | `(2 x N x N x 4) / time` |
| Matrix multiplication GFLOPS | `(2 x N x N x N) / time` |
| Transpose speedup | `Uncoalesced time / Coalesced time` |



Time is measured in seconds for these formulas. Bandwidth is reported in GB/s and floating-point performance in GFLOPS.



## 4. Software and Hardware Requirements



- NVIDIA GPU.

- NVIDIA GPU driver.

- CUDA Toolkit and NVCC compiler.

- C/C++ compiler.

- Visual Studio C++ Build Tools on Windows, or Google Colab with GPU runtime.

- CUDA C/C++ programming language.



## 5. Program Details



**Source file:** `code/exp6_cuda_memory.cu`



The program contains the following kernels and functions:



| Component | Description |
|---|---|
| `transposeNaive()` | Performs uncoalesced matrix transpose |
| `transposeCoalesced()` | Performs coalesced transpose using shared memory |
| `matMulNaive()` | Performs naive matrix multiplication |
| `timeTranspose()` | Measures transpose execution time |
| `checkTranspose()` | Verifies transpose output |
| `checkMatMul()` | Checks selected matrix multiplication results |
| `main()` | Allocates memory, launches kernels and displays results |



### Matrix Sizes



| Parameter | Default Value |
|---|---:|
| Transpose matrix size | 4096 x 4096 |
| Matrix multiplication size | 1024 x 1024 |
| Transpose tile size | 32 x 32 |
| Matrix multiplication block size | 16 x 16 |
| Transpose repetitions | 20 |
| Matrix multiplication repetitions | 5 |



### Folder Structure

```text
Experiment-06-Memory-Optimization-CUDA/
|-- README.md
|-- code/
|   |-- exp6_cuda_memory.cu
|-- screenshots/
```

## 6. Procedure



1. Open a CUDA-supported computer or Google Colab with GPU runtime enabled.

2. Verify NVIDIA GPU availability.

3. Create the CUDA source file `exp6_cuda_memory.cu`.

4. Implement the uncoalesced transpose kernel.

5. Implement the coalesced transpose kernel using shared memory.

6. Implement the naive matrix multiplication kernel.

7. Add CUDA event timing and result verification functions.

8. Compile the CUDA program.

9. Execute the program using the default matrix sizes.

10. Record the execution time, bandwidth, GFLOPS and validation results.

11. Repeat the experiment and compare the results.



### Environment Verification



```bash

nvidia-smi

nvcc --version

```



### Compilation



**Linux / Google Colab**



```bash

nvcc -O2 -o exp6 code/exp6_cuda_memory.cu

```



**Windows Developer Command Prompt**



```cmd

nvcc -O2 -o exp6.exe code\exp6_cuda_memory.cu

```



### Execution



**Linux / Google Colab**



```bash

./exp6

```



**Windows**



```cmd

exp6.exe

```



Optional smaller matrix sizes:



```bash

./exp6 2048 512

```



The first argument specifies the transpose matrix size, and the second specifies the matrix multiplication size.



## 7. Sample Output



The following output is illustrative, taken from the lab manual. Actual results depend on the GPU model, driver, clock speed and system configuration.



```text

GPU: Tesla T4 (compute capability 7.5)



=== PART A: Matrix transpose (N = 4096) ===

Uncoalesced :    1.850 ms |   145.1 GB/s | PASS

Coalesced   :    0.720 ms |   372.8 GB/s | PASS

Speedup (uncoalesced / coalesced) = 2.57x



=== PART B: Naive matrix multiplication (N = 1024) ===

Naive matmul:    3.900 ms |   550.6 GFLOPS | PASS



=== SUMMARY ===

Transpose uncoalesced : 1.850 ms

Transpose coalesced   : 0.720 ms

Naive matmul          : 3.900 ms

```



## 8. Observation Table



Fill in this table using the actual results from the GPU execution.



| Kernel | Matrix Size | Time (ms) | Bandwidth / GFLOPS | Result |
|---|---:|---:|---:|---|
| Uncoalesced transpose | 4096 | - | - | - |
| Coalesced transpose | 4096 | - | - | - |
| Naive matrix multiplication | 1024 | - | - | - |



**Coalesced transpose speedup:** __________ x



**GPU model:** __________________________



**Compute capability:** __________________



## 9. Analysis and Discussion



### Why is coalesced transpose faster?



The uncoalesced kernel performs writes to separated memory locations. This increases the number of memory transactions. The coalesced kernel uses shared memory to arrange the data so that both global memory reads and writes are more efficient.



### Why is shared memory padding used?



The shared-memory tile is declared as `[32][33]` instead of `[32][32]`. The extra column helps avoid shared-memory bank conflicts during transposed access.



### Why is naive matrix multiplication slower?



The naive kernel repeatedly fetches matrix values from global memory. It does not reuse data through shared memory, resulting in more memory traffic.



### Why are CUDA events used?



CUDA kernel launches are asynchronous. CUDA events measure execution time on the GPU and provide more accurate kernel timing than measuring only the CPU launch time.



## 10. Result



The experiment demonstrates the effect of global memory access patterns on GPU performance. Both transpose implementations are verified against the original matrix, and naive matrix multiplication is checked using CPU reference calculations.



Actual timing and performance results are recorded after execution on an NVIDIA GPU.



## 11. Conclusion



This experiment demonstrates that memory access patterns have a major impact on CUDA performance. Coalesced memory access improves global memory efficiency compared with uncoalesced access. Shared memory helps rearrange matrix data, while naive matrix multiplication demonstrates the limitations of repeatedly accessing global memory without data reuse.



## 12. Troubleshooting



| Problem | Possible Solution |
|---|---|
| `nvcc` not found | Install CUDA Toolkit or configure the compiler path |
| No CUDA device detected | Check the NVIDIA driver and GPU availability |
| Compilation error | Verify CUDA syntax and compiler setup |
| Result shows FAIL | Check kernel indexing and boundary conditions |
| Very small speedup | Use larger matrix sizes |
| Out of memory | Reduce the matrix sizes |


