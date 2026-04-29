#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include <mpi.h>
#include "init.h"
extern double min_val, max_val;

// PIC operations
void interpolation(double *mesh_value, Points *points, int num_particles_local);
void normalization(double *mesh_value);
void mover(double *mesh_value, Points *points, int num_particles_local);
void denormalization(double *mesh_value);
long long int void_count(Points *points, int num_particles_local);
void save_mesh(double *mesh_value, int rank);

#endif
