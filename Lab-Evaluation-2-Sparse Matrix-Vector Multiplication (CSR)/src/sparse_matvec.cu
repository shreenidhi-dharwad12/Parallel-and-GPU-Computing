
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;
using namespace chrono;

// CPU implementation
void cpuSpMV(int *rowPtr, int *colIdx, float *values,
             float *x, float *y, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        y[i] = 0;

        for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++)
        {
            y[i] += values[j] * x[colIdx[j]];
        }
    }
}

// Scalar CSR: One thread per row
__global__ void scalarCSR(int *rowPtr, int *colIdx,
                          float *values, float *x,
                          float *y, int rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        float sum = 0;

        for (int j = rowPtr[row]; j < rowPtr[row + 1]; j++)
        {
            sum += values[j] * x[colIdx[j]];
        }

        y[row] = sum;
    }
}

// Vector CSR: One warp per row
__global__ void vectorCSR(int *rowPtr, int *colIdx,
                          float *values, float *x,
                          float *y, int rows)
{
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    int row = blockIdx.x * (blockDim.x / 32) + warp;

    if (row < rows)
    {
        float sum = 0;

        for (int j = rowPtr[row] + lane;
             j < rowPtr[row + 1]; j += 32)
        {
            sum += values[j] * x[colIdx[j]];
        }

        for (int offset = 16; offset > 0; offset /= 2)
        {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0)
            y[row] = sum;
    }
}

int main()
{
    const int rows = 4;
    const int nnz = 6;
    const int iterations = 10000;

    // CSR representation
    int h_rowPtr[] = {0, 2, 3, 5, 6};
    int h_colIdx[] = {0, 3, 1, 0, 2, 3};

    float h_values[] = {10, 20, 30, 40, 50, 60};
    float h_x[] = {1, 2, 3, 4};

    float h_cpu[rows], h_scalar[rows], h_vector[rows];

    // CPU timing
    auto cpuStart = high_resolution_clock::now();

    for (int i = 0; i < iterations; i++)
    {
        cpuSpMV(h_rowPtr, h_colIdx, h_values,
                h_x, h_cpu, rows);
    }

    auto cpuEnd = high_resolution_clock::now();

    double cpuTime =
        duration<double, milli>(cpuEnd - cpuStart).count()
        / iterations;

    // GPU memory allocation
    int *d_rowPtr, *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_rowPtr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_x, rows * sizeof(float));
    cudaMalloc(&d_y, rows * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_rowPtr, h_rowPtr,
               (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_colIdx, h_colIdx,
               nnz * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_values, h_values,
               nnz * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x, h_x,
               rows * sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up
    scalarCSR<<<1, 32>>>(d_rowPtr, d_colIdx, d_values, d_x, d_y, rows);
    vectorCSR<<<1, 128>>>(d_rowPtr, d_colIdx, d_values, d_x, d_y, rows);
    cudaDeviceSynchronize();

    // Scalar CSR timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++)
    {
        scalarCSR<<<1, 32>>>(d_rowPtr, d_colIdx,
                             d_values, d_x, d_y, rows);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float scalarTime;
    cudaEventElapsedTime(&scalarTime, start, stop);
    scalarTime /= iterations;

    // Vector CSR timing
    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++)
    {
        vectorCSR<<<1, 128>>>(d_rowPtr, d_colIdx,
                              d_values, d_x, d_y, rows);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float vectorTime;
    cudaEventElapsedTime(&vectorTime, start, stop);
    vectorTime /= iterations;

    // Copy final results from GPU
    cudaMemcpy(h_scalar, d_y, rows * sizeof(float),
               cudaMemcpyDeviceToHost);

    scalarCSR<<<1, 32>>>(d_rowPtr, d_colIdx,
                         d_values, d_x, d_y, rows);
    cudaDeviceSynchronize();

    cudaMemcpy(h_scalar, d_y, rows * sizeof(float),
               cudaMemcpyDeviceToHost);

    vectorCSR<<<1, 128>>>(d_rowPtr, d_colIdx,
                          d_values, d_x, d_y, rows);
    cudaDeviceSynchronize();

    cudaMemcpy(h_vector, d_y, rows * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Display results
    cout << "\n========== SpMV RESULTS ==========\n";

    cout << "CPU Output:    ";
    for (int i = 0; i < rows; i++)
        cout << h_cpu[i] << " ";

    cout << "\nScalar Output: ";
    for (int i = 0; i < rows; i++)
        cout << h_scalar[i] << " ";

    cout << "\nVector Output: ";
    for (int i = 0; i < rows; i++)
        cout << h_vector[i] << " ";

    cout << "\n\n========== PERFORMANCE ==========\n";

    cout << "CPU Average Time:    " << cpuTime << " ms\n";
    cout << "Scalar Average Time: " << scalarTime << " ms\n";
    cout << "Vector Average Time: " << vectorTime << " ms\n";

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
