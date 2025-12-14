// SPDX-FileCopyrightText: 2025 Adapted for OpenFOAM integration
// SPDX-License-Identifier: MIT

#ifndef CUDA_CORE_H
#define CUDA_CORE_H

#include "cudaField.h"

#ifdef __cplusplus
extern "C" {
#endif

// Main evolution function
// Performs one time step of heat equation: ∂T/∂t = α∇²T
// curr: current temperature field (output)
// prev: previous temperature field (input)
// alpha: thermal diffusivity
// dt: time step
void cuda_evolve(CUDAField* curr, CUDAField* prev, double alpha, double dt);

// Initialize CUDA device and check for errors
int cuda_initialize(void);

// Finalize and cleanup CUDA
void cuda_finalize(void);

#ifdef __cplusplus
}
#endif

#endif // CUDA_CORE_H
