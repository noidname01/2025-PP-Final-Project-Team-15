Info<< "Reading field T\n" << endl;

volScalarField T
(
    IOobject
    (
        "T",
        runTime.name(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading physicalProperties\n" << endl;

IOdictionary physicalProperties
(
    IOobject
    (
        "physicalProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

Info<< "Reading thermal diffusivity alpha\n" << endl;

dimensionedScalar alpha
(
    "alpha",
    dimensionSet(0, 2, -1, 0, 0, 0, 0),  // m^2/s
    physicalProperties
);

Info<< "Reading CUDA grid parameters\n" << endl;

// CUDA grid resolution (user-specified or default)
label cudaGridNx = physicalProperties.lookupOrDefault<label>("cudaGridNx", 100);
label cudaGridNy = physicalProperties.lookupOrDefault<label>("cudaGridNy", 100);
label cudaGridNz = physicalProperties.lookupOrDefault<label>("cudaGridNz", 100);

Info<< "Thermal diffusivity alpha = " << alpha.value() << " m^2/s" << endl;
Info<< "CUDA grid: " << cudaGridNx << " x "
    << cudaGridNy << " x " << cudaGridNz << endl;
