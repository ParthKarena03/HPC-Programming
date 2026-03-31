#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

// Function to run the simulation for different thread counts and configurations
void run_simulation(int threads, int NX, int NY, int NUM_Points, int Maxiter) {
    // Set grid and particle size
    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    // Fix the number of threads
    omp_set_num_threads(threads);

    // Allocate memory for grid and Points (check for allocation failures)
    size_t mesh_elems = (size_t)GRID_X * (size_t)GRID_Y;
    double *mesh_value = (double *) calloc(mesh_elems, sizeof(double));
    if (!mesh_value) {
        fprintf(stderr, "Failed to allocate mesh_value: %zu elements (%zu bytes)\n", mesh_elems, mesh_elems * sizeof(double));
        return;
    }

    // check multiplication overflow for points allocation
    size_t points_elems = (size_t)NUM_Points;
    size_t points_bytes = points_elems * sizeof(Points);
    Points *points = NULL;
    if (points_elems == 0) {
        fprintf(stderr, "NUM_Points is zero, skipping run\n");
        free(mesh_value);
        return;
    }
    points = (Points *) calloc(points_elems, sizeof(Points));
    if (!points) {
        fprintf(stderr, "Failed to allocate points: %zu elements (%zu bytes). Skipping this configuration.\n", points_elems, points_bytes);
        free(mesh_value);
        return;
    }

    // Initialize the points (random initialization or provided)
    initializepoints(points);

    // Print header for results
    printf("Threads: %d, Nx: %d, Ny: %d, Particles: %d\n", threads, NX, NY, NUM_Points);
    printf("Iter\tInterp\t\tMover\t\tTotal\n");
    fflush(stdout);

    // Run for Maxiter iterations
    for (int iter = 0; iter < Maxiter; iter++) {

        // Interpolation timing
        clock_t start_interp = clock();
        interpolation(mesh_value, points);
        clock_t end_interp = clock();

        // Mover timing
        clock_t start_move = clock();
        mover_serial_diff(points, dx, dy);
        clock_t end_move = clock();

        // Calculate execution time
        double interp_time = (double)(end_interp - start_interp) / CLOCKS_PER_SEC;
        double move_time = (double)(end_move - start_move) / CLOCKS_PER_SEC;
        double total = interp_time + move_time;

        // Output results for this iteration
        printf("%d\t%lf\t%lf\t%lf\n", iter + 1, interp_time, move_time, total);
        fflush(stdout);
    }

    // Free memory after each simulation run
    // Save mesh output for this run (overwrites Mesh.out)
    save_mesh(mesh_value);
    printf("Wrote mesh to Mesh.out (GRID: %d x %d)\n", GRID_X, GRID_Y);
    fflush(stdout);

    free(mesh_value);
    free(points);
}

int main(int argc, char **argv) {
    // Configurations for grid and particle ranges
    int grid_configs[3][2] = {{250, 100}, {500, 200}, {1000, 400}};  // Nx, Ny configurations
    int particle_ranges[5] = {14000000}; // Particle ranges

    Maxiter = 10; // Maxiter fixed for all

    // Run the simulation for different configurations and threads
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 5; j++) {
            // Get the current grid size and number of particles
            NX = grid_configs[i][0];
            NY = grid_configs[i][1];
            NUM_Points = particle_ranges[j];

            // Run the simulation for 2, 4, 8, 16 threads
            for (int threads = 2; threads <= 16; threads *= 2) {
                run_simulation(threads, NX, NY, NUM_Points, Maxiter);
            }
        }
    }

    return 0;
}
