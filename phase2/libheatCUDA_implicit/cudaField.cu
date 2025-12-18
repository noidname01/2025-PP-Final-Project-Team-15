// SPDX-FileCopyrightText: 2025 Adapted for OpenFOAM integration
// SPDX-License-Identifier: MIT

#include "cudaField.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Error checking macro
#define GPU_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Allocate memory for CUDA field
void cuda_field_allocate(CUDAField* field, int nx, int ny, int nz)
{
    field->nx = nx;
    field->ny = ny;
    field->nz = nz;

    // Include ghost layers (+2 in each dimension)
    size_t field_size = (nx + 2) * (ny + 2) * (nz + 2) * sizeof(double);

    // Allocate pinned host memory for fast transfer
    GPU_CHECK( cudaMallocHost(&field->h_temperature, field_size) );

    // Allocate device memory
    GPU_CHECK( cudaMalloc(&field->d_temperature, field_size) );

    // Initialize to zero
    GPU_CHECK( cudaMemset(field->d_temperature, 0, field_size) );
    memset(field->h_temperature, 0, field_size);
}

// Set grid spacing
void cuda_field_set_spacing(CUDAField* field, double dx, double dy, double dz)
{
    field->dx = dx;
    field->dy = dy;
    field->dz = dz;
}

// Copy data from host to device
void cuda_field_copy_to_device(CUDAField* field)
{
    size_t field_size = (field->nx + 2) * (field->ny + 2) * (field->nz + 2) * sizeof(double);
    GPU_CHECK( cudaMemcpy(field->d_temperature, field->h_temperature,
                         field_size, cudaMemcpyHostToDevice) );
}

// Copy data from device to host
void cuda_field_copy_to_host(CUDAField* field)
{
    size_t field_size = (field->nx + 2) * (field->ny + 2) * (field->nz + 2) * sizeof(double);
    GPU_CHECK( cudaMemcpy(field->h_temperature, field->d_temperature,
                         field_size, cudaMemcpyDeviceToHost) );
}

// Free allocated memory
void cuda_field_free(CUDAField* field)
{
    if (field->d_temperature) {
        GPU_CHECK( cudaFree(field->d_temperature) );
        field->d_temperature = NULL;
    }

    if (field->h_temperature) {
        GPU_CHECK( cudaFreeHost(field->h_temperature) );
        field->h_temperature = NULL;
    }
}

// Helper: get field size in bytes
size_t cuda_field_size(const CUDAField* field)
{
    return (field->nx + 2) * (field->ny + 2) * (field->nz + 2) * sizeof(double);
}

// Helper: get host pointer
double* cuda_field_host_ptr(CUDAField* field)
{
    return field->h_temperature;
}

// Helper: get device pointer
double* cuda_field_device_ptr(CUDAField* field)
{
    return field->d_temperature;
}
