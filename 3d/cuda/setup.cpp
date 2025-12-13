// SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

#include <string>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include "heat.hpp"
#include "functions.hpp"


void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps)
{
    /*
     * Following combinations of command line arguments are possible:
     * No arguments:    use default field dimensions and number of time steps
     * One argument:    read initial field from a 3D input file
     * Two arguments:   initial field from 3D file and number of time steps
     * Four arguments:  field dimensions (nx, ny, nz) and number of time steps
     */


    int height = 800;             //!< Field dimensions with default values
    int width = 800;
    int length = 800;

    std::string input_file;        //!< Name of the optional input file

    bool read_file = 0;

    nsteps = 500;

    switch (argc) {
    case 1:
        /* Use default values */
        break;
    case 2:
        /* Read initial field from a file */
        input_file = argv[1];
        read_file = true;
        break;
    case 3:
        /* Read initial field from a file */
        input_file = argv[1];
        read_file = true;

        /* Number of time steps */
        nsteps = std::atoi(argv[2]);
        break;
    case 5:
        /* Field dimensions */
        height = std::atoi(argv[1]);
        width = std::atoi(argv[2]);
        length = std::atoi(argv[3]);
        /* Number of time steps */
        nsteps = std::atoi(argv[4]);
        break;
    default:
        std::cout << "Unsupported number of command line arguments" << std::endl;
        exit(-1);
    }

    if (read_file) {
        std::cout << "Reading input from " + input_file << std::endl;
        read_field(current, input_file);
    } else {
        current.setup(height, width, length);
        current.generate();
    }

    // copy "current" field also to "previous"
    previous = current;

    std::cout << "Simulation parameters: "
              << "height: " << height << " width: " << width << " length: " << length
              << " time steps: " << nsteps << std::endl;

    int dev_count;
    cudaGetDeviceCount(&dev_count);
    std::cout << "Number of GPUs: " << dev_count << std::endl;
}
