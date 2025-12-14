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

#ifndef meshMapper_H
#define meshMapper_H

#include "fvMesh.H"
#include "volFields.H"
#include "cudaField.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                        Class MeshMapper Declaration
\*---------------------------------------------------------------------------*/

class MeshMapper
{
    // Private data

        //- Reference to mesh
        const fvMesh& mesh_;

        //- CUDA grid dimensions
        label nx_;
        label ny_;
        label nz_;

        //- Grid spacing
        scalar dx_;
        scalar dy_;
        scalar dz_;

        //- Domain bounds
        scalar xMin_;
        scalar xMax_;
        scalar yMin_;
        scalar yMax_;
        scalar zMin_;
        scalar zMax_;


public:

    // Constructors

        //- Construct from mesh and CUDA grid dimensions
        MeshMapper
        (
            const fvMesh& mesh,
            label nx,
            label ny,
            label nz
        );


    //- Destructor
    ~MeshMapper();


    // Member Functions

        //- Map OpenFOAM field to CUDA array
        void mapToCUDA
        (
            const volScalarField& field,
            CUDAField& cudaField
        ) const;

        //- Map CUDA array back to OpenFOAM field
        void mapFromCUDA
        (
            const CUDAField& cudaField,
            volScalarField& field
        ) const;

        //- Update CUDA ghost layers from OpenFOAM boundary conditions
        void updateBoundaries
        (
            const volScalarField& field,
            CUDAField& cudaField
        ) const;

        //- Return grid spacing in x direction
        scalar dx() const
        {
            return dx_;
        }

        //- Return grid spacing in y direction
        scalar dy() const
        {
            return dy_;
        }

        //- Return grid spacing in z direction
        scalar dz() const
        {
            return dz_;
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
