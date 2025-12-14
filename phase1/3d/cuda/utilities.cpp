// SPDX-FileCopyrightText: 2021 CSC - IT Center for Science Ltd. <www.csc.fi>
//
// SPDX-License-Identifier: MIT

// Utility functions for heat equation solver
//    NOTE: This file does not need to be edited!

#include <chrono>
#include "heat.hpp"

// Calculate average temperature
double average(const Field& field)
{
     double average = 0.0;

     for (int i = 1; i < field.nx + 1; i++) {
       for (int j = 1; j < field.ny + 1; j++) {
         for (int k = 1; k < field.nz + 1; k++) {
           average += field.temperature(i, j, k);
         }
       }
     }

     average /= (field.nx_full * field.ny_full * field.nz_full);
     return average;
}

double timer()
{
    static auto start_time = std::chrono::high_resolution_clock::now();
    auto current_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = current_time - start_time;
    return elapsed.count();
}

