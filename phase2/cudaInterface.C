/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2025 CUDA Integration
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.
\*---------------------------------------------------------------------------*/

#include "cudaInterface.H"
#include <cstring>

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::CUDAHeatSolver::CUDAHeatSolver()
:
    initialized_(false)
{
    // Initialize field structures to null
    current_.h_temperature = nullptr;
    current_.d_temperature = nullptr;
    previous_.h_temperature = nullptr;
    previous_.d_temperature = nullptr;
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::CUDAHeatSolver::~CUDAHeatSolver()
{
    if (initialized_)
    {
        finalize();
    }
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

bool Foam::CUDAHeatSolver::initialize()
{
    int result = cuda_initialize();
    initialized_ = (result == 0);
    return initialized_;
}


void Foam::CUDAHeatSolver::allocateFields
(
    int nx,
    int ny,
    int nz,
    double dx,
    double dy,
    double dz
)
{
    // Allocate current field
    cuda_field_allocate(&current_, nx, ny, nz);
    cuda_field_set_spacing(&current_, dx, dy, dz);

    // Allocate previous field
    cuda_field_allocate(&previous_, nx, ny, nz);
    cuda_field_set_spacing(&previous_, dx, dy, dz);
}


void Foam::CUDAHeatSolver::evolve(double alpha, double dt)
{
    // Call CUDA kernel: updates current_ from previous_
    cuda_evolve(&current_, &previous_, alpha, dt);
}


void Foam::CUDAHeatSolver::copyCurrentToPrevious()
{
    // Copy current host data to previous
    size_t fieldSize = cuda_field_size(&current_);
    std::memcpy(previous_.h_temperature, current_.h_temperature, fieldSize);
}


void Foam::CUDAHeatSolver::swapFields()
{
    // Swap device pointers
    double* temp_d = current_.d_temperature;
    current_.d_temperature = previous_.d_temperature;
    previous_.d_temperature = temp_d;

    // Swap host pointers
    double* temp_h = current_.h_temperature;
    current_.h_temperature = previous_.h_temperature;
    previous_.h_temperature = temp_h;
}


void Foam::CUDAHeatSolver::copyToDevice()
{
    cuda_field_copy_to_device(&current_);
    cuda_field_copy_to_device(&previous_);
}


void Foam::CUDAHeatSolver::copyToHost()
{
    cuda_field_copy_to_host(&current_);
    cuda_field_copy_to_host(&previous_);
}


void Foam::CUDAHeatSolver::finalize()
{
    if (initialized_)
    {
        cuda_field_free(&current_);
        cuda_field_free(&previous_);
        cuda_finalize();
        initialized_ = false;
    }
}


// ************************************************************************* //
