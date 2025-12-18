// SPDX-FileCopyrightText: 2025 Adapted for OpenFOAM integration (Implicit solver with cuSPARSE)
// SPDX-License-Identifier: MIT

#include "cudaCore.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <math.h>
#include <vector>

// Error checking macros
#define GPU_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

#define CUSPARSE_CHECK(call) \
do { \
    cusparseStatus_t err = call; \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error at %s:%d - %d\n", __FILE__, __LINE__, err); \
        return; \
    } \
} while(0)

#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, err); \
        return; \
    } \
} while(0)

// PCG solver parameters
#define MAX_ITER 1000
#define TOLERANCE 1e-6

// Global handles and sparse matrix data
static cusparseHandle_t cusparse_handle = NULL;
static cublasHandle_t cublas_handle = NULL;
static cusparseSpMatDescr_t matA = NULL;

// CSR matrix storage
static int *d_csrRowPtr = NULL;
static int *d_csrColInd = NULL;
static double *d_csrValues = NULL;
static int nnz = 0;
static int matrix_size = 0;

// PCG workspace vectors
static double *d_r = NULL;    // residual
static double *d_p = NULL;    // search direction
static double *d_Ap = NULL;   // A*p
static double *d_z = NULL;    // preconditioned residual
static double *d_diag = NULL; // diagonal for Jacobi preconditioner

// Buffer for SpMV operation
static void *d_spmv_buffer = NULL;
static size_t spmv_buffer_size = 0;

// ====================================================================================
// Sparse Matrix Construction
// ====================================================================================

// Build CSR matrix for implicit heat equation: A = (I - dt*α*∇²)
void build_sparse_matrix(int nx, int ny, int nz, double dx, double dy, double dz,
                         double alpha, double dt)
{
    // Matrix size (interior points only)
    int n = nx * ny * nz;

    if (matrix_size == n) {
        // Matrix already built with correct size, just update values
        // (We'll update in-place for efficiency)
        return;
    }

    // Free old matrix if exists
    if (matA) cusparseDestroySpMat(matA);
    if (d_csrRowPtr) cudaFree(d_csrRowPtr);
    if (d_csrColInd) cudaFree(d_csrColInd);
    if (d_csrValues) cudaFree(d_csrValues);
    if (d_diag) cudaFree(d_diag);

    matrix_size = n;

    // Compute stencil coefficients
    double inv_dx2 = 1.0 / (dx * dx);
    double inv_dy2 = 1.0 / (dy * dy);
    double inv_dz2 = 1.0 / (dz * dz);

    double coeff_diag = 1.0 + dt * alpha * (2.0*inv_dx2 + 2.0*inv_dy2 + 2.0*inv_dz2);
    double coeff_x = -dt * alpha * inv_dx2;
    double coeff_y = -dt * alpha * inv_dy2;
    double coeff_z = -dt * alpha * inv_dz2;

    // Build CSR matrix on host first
    std::vector<int> h_csrRowPtr(n + 1);
    std::vector<int> h_csrColInd;
    std::vector<double> h_csrValues;
    std::vector<double> h_diag(n);

    h_csrRowPtr[0] = 0;
    int current_nnz = 0;

    // For each interior point (i,j,k)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                int row = i * ny * nz + j * nz + k;

                // -z neighbor (k-1)
                if (k > 0) {
                    int col = i * ny * nz + j * nz + (k-1);
                    h_csrColInd.push_back(col);
                    h_csrValues.push_back(coeff_z);
                    current_nnz++;
                }

                // -y neighbor (j-1)
                if (j > 0) {
                    int col = i * ny * nz + (j-1) * nz + k;
                    h_csrColInd.push_back(col);
                    h_csrValues.push_back(coeff_y);
                    current_nnz++;
                }

                // -x neighbor (i-1)
                if (i > 0) {
                    int col = (i-1) * ny * nz + j * nz + k;
                    h_csrColInd.push_back(col);
                    h_csrValues.push_back(coeff_x);
                    current_nnz++;
                }

                // Diagonal
                h_csrColInd.push_back(row);
                h_csrValues.push_back(coeff_diag);
                h_diag[row] = coeff_diag;
                current_nnz++;

                // +x neighbor (i+1)
                if (i < nx - 1) {
                    int col = (i+1) * ny * nz + j * nz + k;
                    h_csrColInd.push_back(col);
                    h_csrValues.push_back(coeff_x);
                    current_nnz++;
                }

                // +y neighbor (j+1)
                if (j < ny - 1) {
                    int col = i * ny * nz + (j+1) * nz + k;
                    h_csrColInd.push_back(col);
                    h_csrValues.push_back(coeff_y);
                    current_nnz++;
                }

                // +z neighbor (k+1)
                if (k < nz - 1) {
                    int col = i * ny * nz + j * nz + (k+1);
                    h_csrColInd.push_back(col);
                    h_csrValues.push_back(coeff_z);
                    current_nnz++;
                }

                h_csrRowPtr[row + 1] = current_nnz;
            }
        }
    }

    nnz = current_nnz;

    // Allocate device memory
    cudaMalloc(&d_csrRowPtr, (n + 1) * sizeof(int));
    cudaMalloc(&d_csrColInd, nnz * sizeof(int));
    cudaMalloc(&d_csrValues, nnz * sizeof(double));
    cudaMalloc(&d_diag, n * sizeof(double));

    // Copy to device
    cudaMemcpy(d_csrRowPtr, h_csrRowPtr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrValues, h_csrValues.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_diag, h_diag.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    // Create sparse matrix descriptor
    cusparseCreateCsr(&matA, n, n, nnz,
                      d_csrRowPtr, d_csrColInd, d_csrValues,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    printf("Sparse matrix built: %d x %d with %d non-zeros (%.1f%% sparse)\n",
           n, n, nnz, 100.0 * (1.0 - (double)nnz / (n*n)));
}

// ====================================================================================
// CUDA Kernels
// ====================================================================================

// Extract interior points from field (with ghost layers) to vector (without ghost layers)
__global__ void extract_interior_kernel(const double *field_with_ghosts, double *interior,
                                       int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        // Field index with ghost layers
        int field_idx = (i+1) * (ny+2) * (nz+2) + (j+1) * (nz+2) + (k+1);
        // Vector index without ghost layers
        int vec_idx = i * ny * nz + j * nz + k;
        interior[vec_idx] = field_with_ghosts[field_idx];
    }
}

// Construct RHS with boundary contributions for implicit solve
// For interior points adjacent to boundaries, we need to add the boundary contribution to RHS
// RHS[i] = T^n[i] - coeffs * T_boundary (for boundary-adjacent cells)
__global__ void construct_rhs_with_boundaries_kernel(const double *field_with_ghosts, double *rhs,
                                                     double coeff_x, double coeff_y, double coeff_z,
                                                     int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        // Interior field index (with ghost layer offset)
        int ii = i + 1;
        int jj = j + 1;
        int kk = k + 1;

        int field_idx = ii * (ny+2) * (nz+2) + jj * (nz+2) + kk;
        int vec_idx = i * ny * nz + j * nz + k;

        // Start with interior value
        double rhs_value = field_with_ghosts[field_idx];

        // Add boundary contributions when moving from LHS to RHS
        // Matrix has: -coeff * T_boundary on LHS → +coeff * T_boundary on RHS
        // For cells at boundaries, add the fixed boundary value contribution

        // -z boundary (k=0)
        if (k == 0) {
            int ghost_idx = ii * (ny+2) * (nz+2) + jj * (nz+2) + 0;  // k ghost layer
            rhs_value += (-coeff_z) * field_with_ghosts[ghost_idx];  // -coeff_z from matrix
        }

        // +z boundary (k=nz-1)
        if (k == nz-1) {
            int ghost_idx = ii * (ny+2) * (nz+2) + jj * (nz+2) + (nz+1);  // k+1 ghost layer
            rhs_value += (-coeff_z) * field_with_ghosts[ghost_idx];
        }

        // -y boundary (j=0)
        if (j == 0) {
            int ghost_idx = ii * (ny+2) * (nz+2) + 0 * (nz+2) + kk;  // j ghost layer
            rhs_value += (-coeff_y) * field_with_ghosts[ghost_idx];
        }

        // +y boundary (j=ny-1)
        if (j == ny-1) {
            int ghost_idx = ii * (ny+2) * (nz+2) + (ny+1) * (nz+2) + kk;  // j+1 ghost layer
            rhs_value += (-coeff_y) * field_with_ghosts[ghost_idx];
        }

        // -x boundary (i=0)
        if (i == 0) {
            int ghost_idx = 0 * (ny+2) * (nz+2) + jj * (nz+2) + kk;  // i ghost layer
            rhs_value += (-coeff_x) * field_with_ghosts[ghost_idx];
        }

        // +x boundary (i=nx-1)
        if (i == nx-1) {
            int ghost_idx = (nx+1) * (ny+2) * (nz+2) + jj * (nz+2) + kk;  // i+1 ghost layer
            rhs_value += (-coeff_x) * field_with_ghosts[ghost_idx];
        }

        rhs[vec_idx] = rhs_value;
    }
}

// Insert interior points back into field (with ghost layers)
__global__ void insert_interior_kernel(double *field_with_ghosts, const double *interior,
                                      int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        // Field index with ghost layers
        int field_idx = (i+1) * (ny+2) * (nz+2) + (j+1) * (nz+2) + (k+1);
        // Vector index without ghost layers
        int vec_idx = i * ny * nz + j * nz + k;
        field_with_ghosts[field_idx] = interior[vec_idx];
    }
}

// Apply Jacobi preconditioner: z = diag(A)^(-1) * r
__global__ void jacobi_precondition_kernel(const double *r, double *z, const double *diag, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = r[i] / diag[i];
    }
}

// Vector operations
__global__ void axpy_kernel(double a, const double *x, double *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void copy_kernel(const double *src, double *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

// ====================================================================================
// PCG Solver with cuSPARSE and Jacobi Preconditioner
// ====================================================================================

void pcg_solve(double *d_x, const double *d_b, int n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Create vector descriptors (local to this function)
    cusparseDnVecDescr_t vecX, vecAp, vecP;
    cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vecAp, n, d_Ap, CUDA_R_64F);
    cusparseCreateDnVec(&vecP, n, d_p, CUDA_R_64F);

    // Allocate SpMV buffer if needed
    if (spmv_buffer_size == 0) {
        double alpha = 1.0, beta = 0.0;
        cusparseSpMV_bufferSize(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecAp,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                &spmv_buffer_size);
        cudaMalloc(&d_spmv_buffer, spmv_buffer_size);
    }

    // r = b - A*x (initial residual)
    double alpha_spmv = 1.0, beta_spmv = 0.0;
    cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha_spmv, matA, vecX, &beta_spmv, vecAp,
                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer);

    copy_kernel<<<numBlocks, blockSize>>>(d_b, d_r, n);
    axpy_kernel<<<numBlocks, blockSize>>>(-1.0, d_Ap, d_r, n);
    cudaDeviceSynchronize();

    // z = M^(-1) * r (apply Jacobi preconditioner)
    jacobi_precondition_kernel<<<numBlocks, blockSize>>>(d_r, d_z, d_diag, n);
    cudaDeviceSynchronize();

    // p = z
    copy_kernel<<<numBlocks, blockSize>>>(d_z, d_p, n);

    // rz_old = <r, z>
    double rz_old;
    cublasDdot(cublas_handle, n, d_r, 1, d_z, 1, &rz_old);

    double initial_residual = sqrt(rz_old);

    // PCG iteration
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Ap = A*p
        alpha_spmv = 1.0; beta_spmv = 0.0;
        cusparseSpMV(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha_spmv, matA, vecP, &beta_spmv, vecAp,
                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer);

        // pAp = <p, Ap>
        double pAp;
        cublasDdot(cublas_handle, n, d_p, 1, d_Ap, 1, &pAp);

        // alpha = rz_old / pAp
        double alpha_pcg = rz_old / pAp;

        // x = x + alpha*p
        cublasDaxpy(cublas_handle, n, &alpha_pcg, d_p, 1, d_x, 1);

        // r = r - alpha*Ap
        double neg_alpha = -alpha_pcg;
        cublasDaxpy(cublas_handle, n, &neg_alpha, d_Ap, 1, d_r, 1);

        // z = M^(-1) * r
        jacobi_precondition_kernel<<<numBlocks, blockSize>>>(d_r, d_z, d_diag, n);
        cudaDeviceSynchronize();

        // rz_new = <r, z>
        double rz_new;
        cublasDdot(cublas_handle, n, d_r, 1, d_z, 1, &rz_new);

        // Check convergence
        double residual = sqrt(rz_new);
        if (residual < TOLERANCE * initial_residual || residual < TOLERANCE) {
            printf("PCG converged in %d iterations (residual: %.2e)\n", iter + 1, residual);
            // Cleanup descriptors
            cusparseDestroyDnVec(vecX);
            cusparseDestroyDnVec(vecAp);
            cusparseDestroyDnVec(vecP);
            return;
        }

        // beta = rz_new / rz_old
        double beta_pcg = rz_new / rz_old;

        // p = z + beta*p
        cublasDscal(cublas_handle, n, &beta_pcg, d_p, 1);
        double one = 1.0;
        cublasDaxpy(cublas_handle, n, &one, d_z, 1, d_p, 1);

        rz_old = rz_new;
    }

    printf("PCG reached max iterations (%d)\n", MAX_ITER);

    // Cleanup descriptors
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecAp);
    cusparseDestroyDnVec(vecP);
}

// ====================================================================================
// Main Interface Functions
// ====================================================================================

void cuda_evolve(CUDAField* curr, CUDAField* prev, double alpha, double dt)
{
    int nx = curr->nx;
    int ny = curr->ny;
    int nz = curr->nz;
    int n = nx * ny * nz;

    // Compute stencil coefficients
    double dx = curr->dx;
    double dy = curr->dy;
    double dz = curr->dz;

    double inv_dx2 = 1.0 / (dx * dx);
    double inv_dy2 = 1.0 / (dy * dy);
    double inv_dz2 = 1.0 / (dz * dz);

    // Off-diagonal coefficients (negative, matching sparse matrix)
    double coeff_x = -dt * alpha * inv_dx2;
    double coeff_y = -dt * alpha * inv_dy2;
    double coeff_z = -dt * alpha * inv_dz2;

    // Build/update sparse matrix
    build_sparse_matrix(nx, ny, nz, curr->dx, curr->dy, curr->dz, alpha, dt);

    // Allocate workspace if needed
    static int allocated_n = 0;
    if (allocated_n != n) {
        if (d_r) cudaFree(d_r);
        if (d_p) cudaFree(d_p);
        if (d_Ap) cudaFree(d_Ap);
        if (d_z) cudaFree(d_z);

        cudaMalloc(&d_r, n * sizeof(double));
        cudaMalloc(&d_p, n * sizeof(double));
        cudaMalloc(&d_Ap, n * sizeof(double));
        cudaMalloc(&d_z, n * sizeof(double));

        allocated_n = n;
    }

    // Allocate temporary vectors for interior points
    double *d_x_interior, *d_b_interior;
    cudaMalloc(&d_x_interior, n * sizeof(double));
    cudaMalloc(&d_b_interior, n * sizeof(double));

    // Setup kernel launch parameters
    dim3 threads(8, 8, 8);
    dim3 blocks((nx + threads.x - 1) / threads.x,
                (ny + threads.y - 1) / threads.y,
                (nz + threads.z - 1) / threads.z);

    // Construct RHS with boundary contributions
    construct_rhs_with_boundaries_kernel<<<blocks, threads>>>(prev->d_temperature, d_b_interior,
                                                               coeff_x, coeff_y, coeff_z,
                                                               nx, ny, nz);

    // Extract interior points for initial guess
    extract_interior_kernel<<<blocks, threads>>>(curr->d_temperature, d_x_interior, nx, ny, nz);
    cudaDeviceSynchronize();

    // Solve: A*x = b
    pcg_solve(d_x_interior, d_b_interior, n);

    // Insert solution back into field
    insert_interior_kernel<<<blocks, threads>>>(curr->d_temperature, d_x_interior, nx, ny, nz);
    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_x_interior);
    cudaFree(d_b_interior);
}

int cuda_initialize(void)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }

    cudaSetDevice(0);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using CUDA device: %s (cuSPARSE implicit solver with Jacobi preconditioner)\n", prop.name);

    // Create cuSPARSE and cuBLAS handles
    cusparseCreate(&cusparse_handle);
    cublasCreate(&cublas_handle);

    return 0;
}

void cuda_finalize(void)
{
    // Cleanup cuSPARSE
    if (matA) cusparseDestroySpMat(matA);
    if (cusparse_handle) cusparseDestroy(cusparse_handle);
    if (cublas_handle) cublasDestroy(cublas_handle);

    // Cleanup device memory
    if (d_csrRowPtr) cudaFree(d_csrRowPtr);
    if (d_csrColInd) cudaFree(d_csrColInd);
    if (d_csrValues) cudaFree(d_csrValues);
    if (d_diag) cudaFree(d_diag);
    if (d_r) cudaFree(d_r);
    if (d_p) cudaFree(d_p);
    if (d_Ap) cudaFree(d_Ap);
    if (d_z) cudaFree(d_z);
    if (d_spmv_buffer) cudaFree(d_spmv_buffer);

    cusparse_handle = NULL;
    cublas_handle = NULL;
    matA = NULL;
    d_csrRowPtr = d_csrColInd = NULL;
    d_csrValues = d_diag = NULL;
    d_r = d_p = d_Ap = d_z = NULL;
    d_spmv_buffer = NULL;
    nnz = matrix_size = 0;
    spmv_buffer_size = 0;

    cudaDeviceReset();
}
