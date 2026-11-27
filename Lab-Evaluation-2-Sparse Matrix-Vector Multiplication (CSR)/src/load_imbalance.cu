
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

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
            sum += values[j] * x[colIdx[j]];

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
            sum += values[j] * x[colIdx[j]];

        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        if (lane == 0)
            y[row] = sum;
    }
}

int main()
{
    const int rows = 4;
    const int nnz = 10;

    // Unequal number of non-zero elements per row
    int h_rowPtr[] = {0, 1, 6, 7, 10};
    int h_colIdx[] = {0, 0, 1, 2, 3, 4, 2, 0, 1, 2};

    float h_values[] = {10, 1, 2, 3, 4, 5, 20, 6, 7, 8};
    float h_x[] = {1, 2, 3, 4, 5};

    float h_scalar[rows], h_vector[rows];

    int *d_rowPtr, *d_colIdx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_rowPtr, (rows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_x, 5 * sizeof(float));
    cudaMalloc(&d_y, rows * sizeof(float));

    cudaMemcpy(d_rowPtr, h_rowPtr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, 5 * sizeof(float), cudaMemcpyHostToDevice);

    scalarCSR<<<1, 32>>>(d_rowPtr, d_colIdx, d_values, d_x, d_y, rows);
    cudaDeviceSynchronize();

    cudaMemcpy(h_scalar, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    vectorCSR<<<1, 128>>>(d_rowPtr, d_colIdx, d_values, d_x, d_y, rows);
    cudaDeviceSynchronize();

    cudaMemcpy(h_vector, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Scalar CSR Output: ";
    for (int i = 0; i < rows; i++)
        cout << h_scalar[i] << " ";

    cout << "\nVector CSR Output: ";
    for (int i = 0; i < rows; i++)
        cout << h_vector[i] << " ";

    cout << "\n\nNon-zero elements per row:\n";

    for (int i = 0; i < rows; i++)
        cout << "Row " << i + 1 << ": "
             << h_rowPtr[i + 1] - h_rowPtr[i] << "\n";

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
