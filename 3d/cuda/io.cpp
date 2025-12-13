// SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

/* I/O related functions for heat equation solver */

#include <string>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <sstream>
#include "matrix.hpp"
#include "heat.hpp"
#ifndef DISABLE_PNG
#include "pngwriter.h"
#endif

// Write a picture of the temperature field
void write_field(Field& field, const int iter)
{
    auto height = field.nx_full;
    auto width = field.ny_full;
    auto length = field.nz_full;

    // Copy the inner data
    auto full_data = Matrix<double>(height, width, length);
    for (int i = 0; i < field.nx; i++)
        for (int j = 0; j < field.ny; j++)
            for (int k = 0; k < field.nz; k++)
                full_data(i, j, k) = field(i + 1, j + 1, k + 1);

    // Write out the middle slice of data to a png file
    std::ostringstream filename_stream;
    filename_stream << "heat_" << std::setw(4) << std::setfill('0') << iter << ".png";
    std::string filename = filename_stream.str();
#ifdef DISABLE_PNG
    std::cout << "No libpng, file not written" << std::endl;
#else
    save_png(full_data.data(height / 2, 0, 0), width, length, filename.c_str(), 'c');
#endif
}

// Write a VTK file for 3D visualization (compatible with ParaView)
void write_vtk(Field& field, const int iter)
{
    auto nx = field.nx;
    auto ny = field.ny;
    auto nz = field.nz;

    // Create VTK filename
    std::ostringstream filename_stream;
    filename_stream << "heat_" << std::setw(4) << std::setfill('0') << iter << ".vtk";
    std::string filename = filename_stream.str();

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open VTK file " << filename << std::endl;
        return;
    }

    // Write VTK header (Legacy format - most compatible)
    file << "# vtk DataFile Version 3.0\n";
    file << "3D Heat Equation Solution - Iteration " << iter << "\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_POINTS\n";
    file << "DIMENSIONS " << nx << " " << ny << " " << nz << "\n";
    file << "ORIGIN 0.0 0.0 0.0\n";
    file << "SPACING " << field.dx << " " << field.dy << " " << field.dz << "\n";
    file << "POINT_DATA " << (nx * ny * nz) << "\n";
    file << "SCALARS temperature float 1\n";
    file << "LOOKUP_TABLE default\n";

    // Write temperature data (excluding ghost layers)
    // VTK expects data with X varying fastest, then Y, then Z
    for (int k = 1; k <= nz; k++) {
        for (int j = 1; j <= ny; j++) {
            for (int i = 1; i <= nx; i++) {
                file << field(i, j, k) << " ";
            }
            file << "\n";
        }
    }

    file.close();
    std::cout << "Wrote VTK file: " << filename << std::endl;
}

// Read the initial temperature distribution from a file
void read_field(Field& field, std::string filename)
{
    std::ifstream file;
    file.open(filename);
    // Read the header
    std::string line, comment;
    std::getline(file, line);

    // Parse header - can be either "# nx ny" (2D) or "# nx ny nz" (3D)
    std::stringstream ss(line);
    ss >> comment;  // Read the '#'

    int nx_full, ny_full, nz_full;
    ss >> nx_full >> ny_full;

    // Check if there's a third dimension
    bool is_3d = false;
    if (ss >> nz_full) {
        is_3d = true;
        std::cout << "Reading 3D input file: " << nx_full << " x " << ny_full << " x " << nz_full << std::endl;
    } else {
        nz_full = ny_full;  // Use ny for nz in 2D case
        std::cout << "Reading 2D input file: " << nx_full << " x " << ny_full << " (replicating in z)" << std::endl;
    }

    field.setup(nx_full, ny_full, nz_full);

    if (is_3d) {
        // Read 3D data directly
        for (int i = 0; i < nx_full; i++) {
            for (int j = 0; j < ny_full; j++) {
                for (int k = 0; k < nz_full; k++) {
                    double value;
                    file >> value;
                    field(i + 1, j + 1, k + 1) = value;
                }
            }
        }
    } else {
        // Read 2D data and replicate it for all z-slices
        for (int i = 0; i < nx_full; i++) {
            for (int j = 0; j < ny_full; j++) {
                double value;
                file >> value;
                // Copy this value to all z-slices
                for (int k = 0; k < nz_full; k++) {
                    field(i + 1, j + 1, k + 1) = value;
                }
            }
        }
    }

    file.close();

    // Set the boundary values
    for (int i = 0; i < field.nx + 2; i++) {
        for (int j = 0; j < field.ny + 2; j++) {
            // z boundaries
            field(i, j, 0) = field(i, j, 1);
            field(i, j, field.nz + 1) = field(i, j, field.nz);
        }
    }
    for (int i = 0; i < field.nx + 2; i++) {
        for (int k = 0; k < field.nz + 2; k++) {
            // y boundaries
            field(i, 0, k) = field(i, 1, k);
            field(i, field.ny + 1, k) = field(i, field.ny, k);
        }
    }
    for (int j = 0; j < field.ny + 2; j++) {
        for (int k = 0; k < field.nz + 2; k++) {
            // x boundaries
            field(0, j, k) = field(1, j, k);
            field(field.nx + 1, j, k) = field(field.nx, j, k);
        }
    }
}
