// SPDX-FileCopyrightText: 2025 Adapted for OpenFOAM integration
// Original: 2021 CSC - IT Center for Science Ltd.
// SPDX-License-Identifier: MIT

#include "cudaCore.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Error checking macro
#define GPU_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// Update the temperature values in the interior
__global__ void evolve_interior_kernel(double *currdata, double *prevdata, double a, double dt,
                                       int nx, int ny, int nz,
                                       double inv_dx2, double inv_dy2, double inv_dz2)
{
    // CUDA threads are arranged in column major order
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i > 1 && j > 1 && k > 1 && i < nx && j < ny && k < nz) {
        int ind = i * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        int ip = (i + 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        int im = (i - 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        int jp = i * (ny + 2) * (nz + 2) + (j + 1) * (nz + 2) + k;
        int jm = i * (ny + 2) * (nz + 2) + (j - 1) * (nz + 2) + k;
        int kp = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k + 1);
        int km = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k - 1);

        currdata[ind] = prevdata[ind] + a * dt * (
            ( prevdata[ip] - 2.0 * prevdata[ind] + prevdata[im] ) * inv_dx2 +
            ( prevdata[jp] - 2.0 * prevdata[ind] + prevdata[jm] ) * inv_dy2 +
            ( prevdata[kp] - 2.0 * prevdata[ind] + prevdata[km] ) * inv_dz2
        );
    }
}

// Update x-direction edges
__global__ void evolve_x_edges_kernel(double *currdata, double *prevdata, double a, double dt,
                                      int nx, int ny, int nz,
                                      double inv_dx2, double inv_dy2, double inv_dz2)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i;

    if (j > 0 && k > 0 && j < ny+1 && k < nz+1) {
        // Process i=1 edge
        i = 1;
        int ind = i * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        int ip = (i + 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        int im = (i - 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        int jp = i * (ny + 2) * (nz + 2) + (j + 1) * (nz + 2) + k;
        int jm = i * (ny + 2) * (nz + 2) + (j - 1) * (nz + 2) + k;
        int kp = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k + 1);
        int km = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k - 1);

        currdata[ind] = prevdata[ind] + a * dt * (
            ( prevdata[ip] - 2.0 * prevdata[ind] + prevdata[im] ) * inv_dx2 +
            ( prevdata[jp] - 2.0 * prevdata[ind] + prevdata[jm] ) * inv_dy2 +
            ( prevdata[kp] - 2.0 * prevdata[ind] + prevdata[km] ) * inv_dz2
        );

        // Process i=nx edge
        i = nx;
        ind = i * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        ip = (i + 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        im = (i - 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        jp = i * (ny + 2) * (nz + 2) + (j + 1) * (nz + 2) + k;
        jm = i * (ny + 2) * (nz + 2) + (j - 1) * (nz + 2) + k;
        kp = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k + 1);
        km = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k - 1);

        currdata[ind] = prevdata[ind] + a * dt * (
            ( prevdata[ip] - 2.0 * prevdata[ind] + prevdata[im] ) * inv_dx2 +
            ( prevdata[jp] - 2.0 * prevdata[ind] + prevdata[jm] ) * inv_dy2 +
            ( prevdata[kp] - 2.0 * prevdata[ind] + prevdata[km] ) * inv_dz2
        );
    }
}

// Update y-direction edges
__global__ void evolve_y_edges_kernel(double *currdata, double *prevdata, double a, double dt,
                                      int nx, int ny, int nz,
                                      double inv_dx2, double inv_dy2, double inv_dz2)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j;

    if (i > 0 && k > 0 && i < nx+1 && k < nz+1) {
        // Process j=1 edge
        j = 1;
        int ind = i * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        int ip = (i + 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        int im = (i - 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        int jp = i * (ny + 2) * (nz + 2) + (j + 1) * (nz + 2) + k;
        int jm = i * (ny + 2) * (nz + 2) + (j - 1) * (nz + 2) + k;
        int kp = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k + 1);
        int km = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k - 1);

        currdata[ind] = prevdata[ind] + a * dt * (
            ( prevdata[ip] - 2.0 * prevdata[ind] + prevdata[im] ) * inv_dx2 +
            ( prevdata[jp] - 2.0 * prevdata[ind] + prevdata[jm] ) * inv_dy2 +
            ( prevdata[kp] - 2.0 * prevdata[ind] + prevdata[km] ) * inv_dz2
        );

        // Process j=ny edge
        j = ny;
        ind = i * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        ip = (i + 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        im = (i - 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        jp = i * (ny + 2) * (nz + 2) + (j + 1) * (nz + 2) + k;
        jm = i * (ny + 2) * (nz + 2) + (j - 1) * (nz + 2) + k;
        kp = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k + 1);
        km = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k - 1);

        currdata[ind] = prevdata[ind] + a * dt * (
            ( prevdata[ip] - 2.0 * prevdata[ind] + prevdata[im] ) * inv_dx2 +
            ( prevdata[jp] - 2.0 * prevdata[ind] + prevdata[jm] ) * inv_dy2 +
            ( prevdata[kp] - 2.0 * prevdata[ind] + prevdata[km] ) * inv_dz2
        );
    }
}

// Update z-direction edges
__global__ void evolve_z_edges_kernel(double *currdata, double *prevdata, double a, double dt,
                                      int nx, int ny, int nz,
                                      double inv_dx2, double inv_dy2, double inv_dz2)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k;

    if (i > 0 && j > 0 && i < nx+1 && j < ny+1) {
        // Process k=1 edge
        k = 1;
        int ind = i * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        int ip = (i + 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        int im = (i - 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        int jp = i * (ny + 2) * (nz + 2) + (j + 1) * (nz + 2) + k;
        int jm = i * (ny + 2) * (nz + 2) + (j - 1) * (nz + 2) + k;
        int kp = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k + 1);
        int km = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k - 1);

        currdata[ind] = prevdata[ind] + a * dt * (
            ( prevdata[ip] - 2.0 * prevdata[ind] + prevdata[im] ) * inv_dx2 +
            ( prevdata[jp] - 2.0 * prevdata[ind] + prevdata[jm] ) * inv_dy2 +
            ( prevdata[kp] - 2.0 * prevdata[ind] + prevdata[km] ) * inv_dz2
        );

        // Process k=nz edge
        k = nz;
        ind = i * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        ip = (i + 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        im = (i - 1) * (ny + 2) * (nz + 2) + j * (nz + 2) + k;
        jp = i * (ny + 2) * (nz + 2) + (j + 1) * (nz + 2) + k;
        jm = i * (ny + 2) * (nz + 2) + (j - 1) * (nz + 2) + k;
        kp = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k + 1);
        km = i * (ny + 2) * (nz + 2) + j * (nz + 2) + (k - 1);

        currdata[ind] = prevdata[ind] + a * dt * (
            ( prevdata[ip] - 2.0 * prevdata[ind] + prevdata[im] ) * inv_dx2 +
            ( prevdata[jp] - 2.0 * prevdata[ind] + prevdata[jm] ) * inv_dy2 +
            ( prevdata[kp] - 2.0 * prevdata[ind] + prevdata[km] ) * inv_dz2
        );
    }
}

// Main evolution function - performs one time step
extern "C" void cuda_evolve(CUDAField* curr, CUDAField* prev, double alpha, double dt)
{
    int nx = curr->nx;
    int ny = curr->ny;
    int nz = curr->nz;

    double inv_dx2 = 1.0 / (curr->dx * curr->dx);
    double inv_dy2 = 1.0 / (curr->dy * curr->dy);
    double inv_dz2 = 1.0 / (curr->dz * curr->dz);

    double *currdata = curr->d_temperature;
    double *prevdata = prev->d_temperature;

    // Launch interior kernel
    int blocksizes[3] = {16, 8, 8};
    dim3 dimBlock(blocksizes[0], blocksizes[1], blocksizes[2]);
    dim3 dimGrid((nz + 2 + blocksizes[0] - 1) / blocksizes[0],
                 (ny + 2 + blocksizes[1] - 1) / blocksizes[1],
                 (nx + 2 + blocksizes[2] - 1) / blocksizes[2]);

    evolve_interior_kernel<<<dimGrid, dimBlock>>>(currdata, prevdata, alpha, dt, nx, ny, nz,
                                                   inv_dx2, inv_dy2, inv_dz2);

    // Launch edge kernels with 2D blocks
    blocksizes[0] = 32;
    blocksizes[1] = 32;
    blocksizes[2] = 1;
    dimBlock.x = blocksizes[0];
    dimBlock.y = blocksizes[1];
    dimBlock.z = blocksizes[2];

    // X-edges
    dimGrid.x = (nz + 2 + blocksizes[0] - 1) / blocksizes[0];
    dimGrid.y = (ny + 2 + blocksizes[1] - 1) / blocksizes[1];
    dimGrid.z = 1;
    evolve_x_edges_kernel<<<dimGrid, dimBlock>>>(currdata, prevdata, alpha, dt, nx, ny, nz,
                                                 inv_dx2, inv_dy2, inv_dz2);

    // Y-edges
    dimGrid.x = (nz + 2 + blocksizes[0] - 1) / blocksizes[0];
    dimGrid.y = (nx + 2 + blocksizes[1] - 1) / blocksizes[1];
    evolve_y_edges_kernel<<<dimGrid, dimBlock>>>(currdata, prevdata, alpha, dt, nx, ny, nz,
                                                 inv_dx2, inv_dy2, inv_dz2);

    // Z-edges
    dimGrid.x = (ny + 2 + blocksizes[0] - 1) / blocksizes[0];
    dimGrid.y = (nx + 2 + blocksizes[1] - 1) / blocksizes[1];
    evolve_z_edges_kernel<<<dimGrid, dimBlock>>>(currdata, prevdata, alpha, dt, nx, ny, nz,
                                                 inv_dx2, inv_dy2, inv_dz2);

    cudaDeviceSynchronize();
    GPU_CHECK( cudaGetLastError() );
}

// Initialize CUDA
extern "C" int cuda_initialize(void)
{
    int dev_count;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA initialization failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (dev_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }

    printf("CUDA initialized: %d device(s) found\n", dev_count);
    return 0;
}

// Finalize CUDA
extern "C" void cuda_finalize(void)
{
    cudaDeviceReset();
}
