// SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

#include "heat.hpp"
#include "matrix.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include "error_checks.h"

void Field::setup(int nx_in, int ny_in, int nz_in)
{
    nx_full = nx_in;
    ny_full = ny_in;
    nz_full = nz_in;

    nx = nx_full;
    ny = ny_full;
    nz = nz_full;

    // matrix includes also ghost layers
    temperature = Matrix<double> (nx + 2, ny + 2, nz + 2);
}

void Field::generate() {

    // Radius of the source disc
    double radius = (nx_full + ny_full + nz_full) / 18.0;
    for (int i = 0; i < nx + 2; i++) {
        for (int j = 0; j < ny + 2; j++) {
            for (int k = 0; k < nz + 2; k++) {
                // Distance of point i, j, k from the origin
                auto dx = i - nx_full / 2;
                auto dy = j - ny_full / 2;
                auto dz = k - nz_full / 2;
                if (dx * dx + dy * dy + dz * dz < radius * radius) {
                    temperature(i, j, k) = 5.0;
                } else {
                    temperature(i, j, k) = 65.0;
                }
            }
        }
    }

    // Boundary conditions - fixed temperatures on all 6 faces
    // z boundaries
    for (int i = 0; i < nx + 2; i++) {
        for (int j = 0; j < ny + 2; j++) {
            temperature(i, j, 0) = 20.0;
            temperature(i, j, nz + 1) = 35.0;
        }
    }

    // y boundaries
    for (int i = 0; i < nx + 2; i++) {
        for (int k = 0; k < nz + 2; k++) {
            temperature(i, 0, k) = 35.0;
            temperature(i, ny + 1, k) = 20.0;
        }
    }

    // x boundaries
    for (int j = 0; j < ny + 2; j++) {
        for (int k = 0; k < nz + 2; k++) {
            temperature(0, j, k) = 20.0;
            temperature(nx + 1, j, k) = 35.0;
        }
    }
}
