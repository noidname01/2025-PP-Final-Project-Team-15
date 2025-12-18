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

#include "meshMapper.H"
#include "boundBox.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::MeshMapper::MeshMapper
(
    const fvMesh& mesh,
    label nx,
    label ny,
    label nz
)
:
    mesh_(mesh),
    nx_(nx),
    ny_(ny),
    nz_(nz)
{
    // Compute bounding box from mesh cell centers
    const volVectorField& C = mesh.C();
    boundBox bb(C.primitiveField());

    xMin_ = bb.min().x();
    xMax_ = bb.max().x();
    yMin_ = bb.min().y();
    yMax_ = bb.max().y();
    zMin_ = bb.min().z();
    zMax_ = bb.max().z();

    // Compute grid spacing
    dx_ = (xMax_ - xMin_) / nx_;
    dy_ = (yMax_ - yMin_) / ny_;
    dz_ = (zMax_ - zMin_) / nz_;

    Info<< "MeshMapper initialized:" << nl
        << "  Domain: [" << xMin_ << ", " << xMax_ << "] x "
        << "[" << yMin_ << ", " << yMax_ << "] x "
        << "[" << zMin_ << ", " << zMax_ << "]" << nl
        << "  Grid spacing: dx=" << dx_ << " dy=" << dy_ << " dz=" << dz_ << endl;
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::MeshMapper::~MeshMapper()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::MeshMapper::mapToCUDA
(
    const volScalarField& field,
    CUDAField& cudaField
) const
{
    // Phase 1: Simplified mapping for structured grids
    // Assumes OpenFOAM cells map directly to CUDA grid points

    const scalarField& fieldData = field.primitiveField();
    const volVectorField& C = mesh_.C();

    double* cudaData = cudaField.h_temperature;

    // Clear CUDA array
    label totalSize = (nx_ + 2) * (ny_ + 2) * (nz_ + 2);
    for (label idx = 0; idx < totalSize; idx++)
    {
        cudaData[idx] = 0.0;
    }

    // Debug: Track statistics
    scalar minT = GREAT;
    scalar maxT = -GREAT;
    label hotCells = 0;

    // Map each OpenFOAM cell to CUDA grid
    forAll(fieldData, cellI)
    {
        const vector& cellCenter = C[cellI];
        scalar cellTemp = fieldData[cellI];

        minT = min(minT, cellTemp);
        maxT = max(maxT, cellTemp);
        if (cellTemp > 400.0) hotCells++;

        // Compute CUDA grid indices
        label i = label((cellCenter.x() - xMin_) / dx_) + 1;  // +1 for ghost layer
        label j = label((cellCenter.y() - yMin_) / dy_) + 1;
        label k = label((cellCenter.z() - zMin_) / dz_) + 1;

        // Clamp to valid range
        i = max(1, min(i, nx_));
        j = max(1, min(j, ny_));
        k = max(1, min(k, nz_));

        // CUDA indexing: i*(ny+2)*(nz+2) + j*(nz+2) + k
        label idx = i * (ny_ + 2) * (nz_ + 2) + j * (nz_ + 2) + k;

        cudaData[idx] = fieldData[cellI];

        // Debug hot cells (sphere region)
        if (cellTemp > 400.0 && hotCells <= 10)
        {
            Info << "  Hot cell " << cellI << ": center=" << cellCenter
                 << ", T=" << cellTemp
                 << " -> CUDA[" << i << "," << j << "," << k << "]" << endl;
        }
    }

    Info << "mapToCUDA: " << fieldData.size() << " OpenFOAM cells mapped" << nl
         << "  Temperature range: [" << minT << ", " << maxT << "]" << nl
         << "  Hot cells (T>400K): " << hotCells << endl;

}


void Foam::MeshMapper::mapFromCUDA
(
    const CUDAField& cudaField,
    volScalarField& field
) const
{
    // Phase 1: Simplified mapping for structured grids

    scalarField& fieldData = field.primitiveFieldRef();
    const volVectorField& C = mesh_.C();

    const double* cudaData = cudaField.h_temperature;

    // Map each CUDA grid point back to nearest OpenFOAM cell
    forAll(fieldData, cellI)
    {
        const vector& cellCenter = C[cellI];

        // Compute CUDA grid indices
        label i = label((cellCenter.x() - xMin_) / dx_) + 1;
        label j = label((cellCenter.y() - yMin_) / dy_) + 1;
        label k = label((cellCenter.z() - zMin_) / dz_) + 1;

        // Clamp to valid range
        i = max(1, min(i, nx_));
        j = max(1, min(j, ny_));
        k = max(1, min(k, nz_));

        // CUDA indexing
        label idx = i * (ny_ + 2) * (nz_ + 2) + j * (nz_ + 2) + k;

        fieldData[cellI] = cudaData[idx];
    }
}


void Foam::MeshMapper::updateBoundaries
(
    const volScalarField& field,
    CUDAField& cudaField
) const
{
    // Phase 1: Update CUDA ghost layers for structured grids
    // For each patch, identify which CUDA boundary face it is and set all ghost cells

    double* cudaData = cudaField.h_temperature;

    const volScalarField::Boundary& bField = field.boundaryField();

    forAll(bField, patchi)
    {
        const fvPatchScalarField& pField = bField[patchi];
        const fvPatch& patch = pField.patch();

        // Get average normal to determine which CUDA face
        vector avgNormal = average(patch.nf());

        // Get boundary value (for fixedValue)
        bool isFixedValue = (pField.type() == "fixedValue");
        scalar bcValue = 0.0;
        if (isFixedValue && pField.size() > 0)
        {
            bcValue = pField[0];
        }

        // Determine which CUDA boundary face and set all ghost cells
        if (mag(avgNormal.z() - 1.0) < 0.1)  // Top face (z = zMax, k = nz+1)
        {
            for (label i = 1; i <= nx_; i++)
            {
                for (label j = 1; j <= ny_; j++)
                {
                    label idx = i * (ny_ + 2) * (nz_ + 2) + j * (nz_ + 2) + (nz_ + 1);
                    if (!isFixedValue)  // zeroGradient: copy from adjacent interior
                    {
                        label interiorIdx = i * (ny_ + 2) * (nz_ + 2) + j * (nz_ + 2) + nz_;
                        cudaData[idx] = cudaData[interiorIdx];
                    }
                    else
                    {
                        cudaData[idx] = bcValue;
                    }
                }
            }
        }
        else if (mag(avgNormal.z() + 1.0) < 0.1)  // Bottom face (z = zMin, k = 0)
        {
            for (label i = 1; i <= nx_; i++)
            {
                for (label j = 1; j <= ny_; j++)
                {
                    label idx = i * (ny_ + 2) * (nz_ + 2) + j * (nz_ + 2) + 0;
                    if (!isFixedValue)  // zeroGradient: copy from adjacent interior
                    {
                        label interiorIdx = i * (ny_ + 2) * (nz_ + 2) + j * (nz_ + 2) + 1;
                        cudaData[idx] = cudaData[interiorIdx];
                    }
                    else
                    {
                        cudaData[idx] = bcValue;
                    }
                }
            }
        }
        else if (mag(avgNormal.x() - 1.0) < 0.1)  // Right face (x = xMax, i = nx+1)
        {
            for (label j = 1; j <= ny_; j++)
            {
                for (label k = 1; k <= nz_; k++)
                {
                    label idx = (nx_ + 1) * (ny_ + 2) * (nz_ + 2) + j * (nz_ + 2) + k;
                    if (!isFixedValue)  // zeroGradient: copy from adjacent interior
                    {
                        label interiorIdx = nx_ * (ny_ + 2) * (nz_ + 2) + j * (nz_ + 2) + k;
                        cudaData[idx] = cudaData[interiorIdx];
                    }
                    else
                    {
                        cudaData[idx] = bcValue;
                    }
                }
            }
        }
        else if (mag(avgNormal.x() + 1.0) < 0.1)  // Left face (x = xMin, i = 0)
        {
            for (label j = 1; j <= ny_; j++)
            {
                for (label k = 1; k <= nz_; k++)
                {
                    label idx = 0 * (ny_ + 2) * (nz_ + 2) + j * (nz_ + 2) + k;
                    if (!isFixedValue)
                    {
                        label interiorIdx = 1 * (ny_ + 2) * (nz_ + 2) + j * (nz_ + 2) + k;
                        cudaData[idx] = cudaData[interiorIdx];
                    }
                    else
                    {
                        cudaData[idx] = bcValue;
                    }
                }
            }
        }
        else if (mag(avgNormal.y() - 1.0) < 0.1)  // Front face (y = yMax, j = ny+1)
        {
            for (label i = 1; i <= nx_; i++)
            {
                for (label k = 1; k <= nz_; k++)
                {
                    label idx = i * (ny_ + 2) * (nz_ + 2) + (ny_ + 1) * (nz_ + 2) + k;
                    if (!isFixedValue)
                    {
                        label interiorIdx = i * (ny_ + 2) * (nz_ + 2) + ny_ * (nz_ + 2) + k;
                        cudaData[idx] = cudaData[interiorIdx];
                    }
                    else
                    {
                        cudaData[idx] = bcValue;
                    }
                }
            }
        }
        else if (mag(avgNormal.y() + 1.0) < 0.1)  // Back face (y = yMin, j = 0)
        {
            for (label i = 1; i <= nx_; i++)
            {
                for (label k = 1; k <= nz_; k++)
                {
                    label idx = i * (ny_ + 2) * (nz_ + 2) + 0 * (nz_ + 2) + k;
                    if (!isFixedValue)
                    {
                        label interiorIdx = i * (ny_ + 2) * (nz_ + 2) + 1 * (nz_ + 2) + k;
                        cudaData[idx] = cudaData[interiorIdx];
                    }
                    else
                    {
                        cudaData[idx] = bcValue;
                    }
                }
            }
        }
    }

    // Debug: Check if hot cells still exist after boundary update
    label hotCount = 0;
    label totalSize = (nx_ + 2) * (ny_ + 2) * (nz_ + 2);
    scalar cudaMin = GREAT;
    scalar cudaMax = -GREAT;
    for (label idx = 0; idx < totalSize; idx++)
    {
        cudaMin = min(cudaMin, cudaData[idx]);
        cudaMax = max(cudaMax, cudaData[idx]);
        if (cudaData[idx] > 400.0) hotCount++;
    }
    Info << "After updateBoundaries:" << nl
         << "  CUDA array range: [" << cudaMin << ", " << cudaMax << "]" << nl
         << "  Hot cells (T>400K): " << hotCount << endl;
}


// ************************************************************************* //
