#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {

    // ================= MPI INITIALIZATION =================
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <input_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    FILE *file = NULL;
    if (rank == 0) {
        file = fopen(argv[1], "rb");
        if (!file) {
            printf("Error opening input file\n");
            MPI_Finalize();
            exit(1);
        }

        // Read grid dimensions
        fread(&NX, sizeof(int), 1, file);
        fread(&NY, sizeof(int), 1, file);

        // Read number of Points and max iterations
        fread(&NUM_Points, sizeof(int), 1, file);
        fread(&Maxiter, sizeof(int), 1, file);
    }

    // ================= BROADCAST SIMULATION PARAMETERS =================
    MPI_Bcast(&NX, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NY, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NUM_Points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Maxiter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Since Number of points will be 1 more than number of cells
    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    // ================= PARTICLE DECOMPOSITION =================
    int particles_per_proc = NUM_Points / num_procs;
    int remainder = NUM_Points % num_procs;
    
    int num_particles_local = particles_per_proc + (rank < remainder ? 1 : 0);
    int start_idx = rank * particles_per_proc + (rank < remainder ? rank : remainder);

    // Allocate memory for grid and local points
    double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    double *local_grid = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points *) calloc(num_particles_local, sizeof(Points));

    // ================= READ AND SCATTER PARTICLES =================
    if (rank == 0) {
        // Read all particles (on rank 0)
        Points *all_points = (Points *) calloc(NUM_Points, sizeof(Points));
        read_points(file, all_points);

        // Scatter particles to other processes
        for (int p = 1; p < num_procs; p++) {
            int p_particles_per_proc = particles_per_proc + (p < remainder ? 1 : 0);
            int p_start_idx = p * particles_per_proc + (p < remainder ? p : remainder);
            MPI_Send(&all_points[p_start_idx], p_particles_per_proc * sizeof(Points), 
                     MPI_BYTE, p, 0, MPI_COMM_WORLD);
        }

        // Copy rank 0's particles
        memcpy(points, all_points, num_particles_local * sizeof(Points));
        free(all_points);
        fclose(file);
    } else {
        // Receive particles on other processes
        MPI_Recv(points, num_particles_local * sizeof(Points), 
                 MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double total_int_time = 0.0;
    double total_norm_time = 0.0;
    double total_move_time = 0.0;
    double total_denorm_time = 0.0;

    for (int iter = 0; iter < Maxiter; iter++) {

        double t0 = omp_get_wtime();

        // Reset local grid
        #pragma omp parallel for
        for (int i = 0; i < GRID_X * GRID_Y; i++) {
            local_grid[i] = 0.0;
        }

        // Perform interpolation on local particles
        interpolation(local_grid, points, num_particles_local);

        // ================= MPI_ALLREDUCE: COMBINE LOCAL GRIDS =================
        MPI_Allreduce(local_grid, mesh_value, GRID_X * GRID_Y, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double t1 = omp_get_wtime();
        
        normalization(mesh_value);

        double t2 = omp_get_wtime();

        // Perform mover on local particles using global grid
        mover(mesh_value, points, num_particles_local);

        double t3 = omp_get_wtime();

        denormalization(mesh_value);

        double t4 = omp_get_wtime();

        total_int_time += (double)(t1 - t0);
        total_norm_time += (double)(t2 - t1);
        total_move_time += (double)(t3 - t2);
        total_denorm_time += (double)(t4 - t3);
    }

    // ================= ONLY RANK 0 SAVES OUTPUT =================
    save_mesh(mesh_value, rank);

    // ================= GATHER VOID COUNT =================
    long long int local_voids = void_count(points, num_particles_local);
    long long int total_voids = 0;
    MPI_Reduce(&local_voids, &total_voids, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total Interpolation Time = %lf seconds\n", total_int_time);
        printf("Total Normalization Time = %lf seconds\n", total_norm_time);
        printf("Total Mover Time = %lf seconds\n", total_move_time);
        printf("Total Denormalization Time = %lf seconds\n", total_denorm_time);
        printf("Total Algorithm Time = %lf seconds\n", total_int_time + total_norm_time + total_move_time + total_denorm_time);
        printf("Total Number of Voids = %lld\n", total_voids);
    }
    
    // Free memory
    free(mesh_value);
    free(local_grid);
    free(points);

    // ================= MPI FINALIZATION =================
    MPI_Finalize();

    return 0;
}
   