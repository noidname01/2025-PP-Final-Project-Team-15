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

#ifndef cudaInterface_H
#define cudaInterface_H

#include "cudaField.h"
#include "cudaCore.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                      Class CUDAHeatSolver Declaration
\*---------------------------------------------------------------------------*/

class CUDAHeatSolver
{
    // Private data

        //- Current and previous temperature fields
        CUDAField current_;
        CUDAField previous_;

        //- Initialization flag
        bool initialized_;

public:

    // Constructors

        //- Construct null
        CUDAHeatSolver();

    //- Destructor
    ~CUDAHeatSolver();


    // Member Functions

        //- Initialize CUDA device
        bool initialize();

        //- Allocate CUDA fields
        void allocateFields
        (
            int nx,
            int ny,
            int nz,
            double dx,
            double dy,
            double dz
        );

        //- Perform one time step evolution
        void evolve(double alpha, double dt);

        //- Copy current to previous field
        void copyCurrentToPrevious();

        //- Swap current and previous fields
        void swapFields();

        //- Copy data from host to device
        void copyToDevice();

        //- Copy data from device to host
        void copyToHost();

        //- Finalize and cleanup CUDA
        void finalize();

        //- Access current field
        CUDAField& currentField()
        {
            return current_;
        }

        //- Access previous field
        CUDAField& previousField()
        {
            return previous_;
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
