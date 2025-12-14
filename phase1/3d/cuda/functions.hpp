#pragma once
#include "heat.hpp"
#include <cuda_runtime_api.h>

// Function declarations
void initialize(int argc, char *argv[], Field& current,
                Field& previous, int& nsteps);

void evolve(Field& curr, Field& prev, const double a, const double dt);

void evolve_interior(Field& curr, Field& prev, const double a, const double dt, cudaStream_t *streams);

void evolve_edges(Field& curr, Field& prev, const double a, const double dt, cudaStream_t *streams);

void write_field(Field& field, const int iter);

void write_vtk(Field& field, const int iter);

void read_field(Field& field, std::string filename);

double average(const Field& field);

double timer();

void exit_data(Field& curr, Field& prev);

void enter_data(Field& curr, Field& prev);

void update_host(Field& temperature);

void update_device(Field& temperature);

void allocate_data(Field& field1, Field& field2);

void free_data(Field& field1, Field& field2);

