/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2025 CUDA Integration
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    heatFoamCUDA

Description
    Solves the transient heat diffusion equation using CUDA GPU acceleration:
        ∂T/∂t = α∇²T

    Based on laplacianFoam but replaces OpenFOAM's matrix solver with
    CUDA finite-difference kernels for improved performance.

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "simpleControl.H"

#include "cudaInterface.H"
#include "meshMapper.H"

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nInitializing CUDA heat solver\n" << endl;

    // Initialize CUDA
    CUDAHeatSolver cudaSolver;
    if (!cudaSolver.initialize())
    {
        FatalErrorInFunction
            << "Failed to initialize CUDA"
            << exit(FatalError);
    }

    // Create mesh mapper
    Info<< "Creating mesh mapper..." << endl;
    MeshMapper mapper(mesh, cudaGridNx, cudaGridNy, cudaGridNz);

    Info<< "CUDA grid: " << cudaGridNx << " x "
        << cudaGridNy << " x " << cudaGridNz << endl;
    Info<< "Grid spacing: dx=" << mapper.dx()
        << " dy=" << mapper.dy()
        << " dz=" << mapper.dz() << endl;

    // Setup CUDA fields
    cudaSolver.allocateFields(cudaGridNx, cudaGridNy, cudaGridNz,
                              mapper.dx(), mapper.dy(), mapper.dz());

    // Map initial condition from OpenFOAM to CUDA
    Info<< "Mapping initial condition to CUDA..." << endl;
    mapper.mapToCUDA(T, cudaSolver.currentField());
    mapper.updateBoundaries(T, cudaSolver.currentField());

    // Copy previous field (for time stepping)
    cudaSolver.copyCurrentToPrevious();

    // Transfer to GPU
    cudaSolver.copyToDevice();

    Info<< "\nStarting time loop with CUDA acceleration\n" << endl;

    while (simple.loop(runTime))
    {
        Info<< "Time = " << runTime.userTimeName() << nl << endl;

        // Update boundary conditions from OpenFOAM
        mapper.updateBoundaries(T, cudaSolver.currentField());
        cudaSolver.copyToDevice();

        // Execute CUDA time step: ∂T/∂t = α∇²T
        cudaSolver.evolve(alpha.value(), runTime.deltaTValue());

        // Swap current and previous for next iteration
        cudaSolver.swapFields();

        // Copy solution back to host
        cudaSolver.copyToHost();

        // Map from CUDA grid back to OpenFOAM field
        mapper.mapFromCUDA(cudaSolver.previousField(), T);

        // Apply OpenFOAM boundary conditions
        T.correctBoundaryConditions();

        // Write output
        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    // Cleanup
    cudaSolver.finalize();

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
