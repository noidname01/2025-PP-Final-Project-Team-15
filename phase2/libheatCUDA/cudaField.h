// SPDX-FileCopyrightText: 2025 Adapted for OpenFOAM integration
// SPDX-License-Identifier: MIT

#ifndef CUDA_FIELD_H
#define CUDA_FIELD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Simplified Field structure for CUDA library
// Plain C struct compatible with both CUDA and C++
typedef struct {
    int nx, ny, nz;           // Field dimensions (without ghost layers)
    double dx, dy, dz;        // Grid spacing

    // Host data (pinned memory for fast transfer)
    double* h_temperature;

    // Device data
    double* d_temperature;

} CUDAField;

// Memory management functions
void cuda_field_allocate(CUDAField* field, int nx, int ny, int nz);
void cuda_field_set_spacing(CUDAField* field, double dx, double dy, double dz);
void cuda_field_copy_to_device(CUDAField* field);
void cuda_field_copy_to_host(CUDAField* field);
void cuda_field_free(CUDAField* field);

// Helper functions
size_t cuda_field_size(const CUDAField* field);
double* cuda_field_host_ptr(CUDAField* field);
double* cuda_field_device_ptr(CUDAField* field);

#ifdef __cplusplus
}
#endif

#endif // CUDA_FIELD_H
