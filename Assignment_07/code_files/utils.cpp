#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

double min_val, max_val;

// ======================= INTERPOLATION (OPTIMIZED) =======================
// Thread-private grid to remove atomics
void interpolation(double *mesh_value, Points *points) {

    int num_threads = omp_get_max_threads();

    // Allocate thread-local grids
    double **local_grid = (double **)malloc(num_threads * sizeof(double *));
    for (int t = 0; t < num_threads; t++) {
        local_grid[t] = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *grid = local_grid[tid];

        #pragma omp for schedule(guided)
        for (int i = 0; i < NUM_Points; i++) {

            if (points[i].is_void) continue;

            double x = points[i].x;
            double y = points[i].y;

            int xi = (int)(x / dx);
            int yi = (int)(y / dy);

            // Clamp safely (IMPORTANT: avoid xi+1 overflow)
            if (xi < 0) xi = 0;
            if (xi >= GRID_X - 1) xi = GRID_X - 2;

            if (yi < 0) yi = 0;
            if (yi >= GRID_Y - 1) yi = GRID_Y - 2;

            double Xi = xi * dx;
            double Yj = yi * dy;

            double lx = x - Xi;
            double ly = y - Yj;

            double w_ij   = (dx - lx) * (dy - ly);
            double w_i1j  = lx * (dy - ly);
            double w_ij1  = (dx - lx) * ly;
            double w_i1j1 = lx * ly;

            int idx = yi * GRID_X + xi;

            grid[idx]                 += w_ij;
            grid[idx + 1]             += w_i1j;
            grid[idx + GRID_X]        += w_ij1;
            grid[idx + GRID_X + 1]    += w_i1j1;
        }
    }

    // Merge thread-local grids into global grid
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        for (int t = 0; t < num_threads; t++) {
            mesh_value[i] += local_grid[t][i];
        }
    }

    // Free memory
    for (int t = 0; t < num_threads; t++) {
        free(local_grid[t]);
    }
    free(local_grid);
}

// ======================= NORMALIZATION =======================
void normalization(double *mesh_value) {

    min_val = mesh_value[0];
    max_val = mesh_value[0];

    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        if (mesh_value[i] < min_val) min_val = mesh_value[i];
        if (mesh_value[i] > max_val) max_val = mesh_value[i];
    }

    double range = max_val - min_val;
    if (range == 0) range = 1;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        mesh_value[i] = 2.0 * (mesh_value[i] - min_val) / range - 1.0;
    }
}

// ======================= MOVER =======================
void mover(double *mesh_value, Points *points) {

    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < NUM_Points; i++) {

        if (points[i].is_void) continue;

        double x = points[i].x;
        double y = points[i].y;

        int xi = (int)(x / dx);
        int yi = (int)(y / dy);

        if (xi < 0) xi = 0;
        if (xi >= GRID_X - 1) xi = GRID_X - 2;

        if (yi < 0) yi = 0;
        if (yi >= GRID_Y - 1) yi = GRID_Y - 2;

        double Xi = xi * dx;
        double Yj = yi * dy;

        double lx = x - Xi;
        double ly = y - Yj;

        double w_ij   = (dx - lx) * (dy - ly);
        double w_i1j  = lx * (dy - ly);
        double w_ij1  = (dx - lx) * ly;
        double w_i1j1 = lx * ly;

        int idx = yi * GRID_X + xi;

        double Fi =
            w_ij   * mesh_value[idx] +
            w_i1j  * mesh_value[idx + 1] +
            w_ij1  * mesh_value[idx + GRID_X] +
            w_i1j1 * mesh_value[idx + GRID_X + 1];

        points[i].x = x + Fi * dx;
        points[i].y = y + Fi * dy;

        if (points[i].x < 0.0 || points[i].x > 1.0 ||
            points[i].y < 0.0 || points[i].y > 1.0) {
            points[i].is_void = true;
        }
    }
}

// ======================= DENORMALIZATION =======================
void denormalization(double *mesh_value) {

    double range = max_val - min_val;
    if (range == 0) range = 1;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        mesh_value[i] = (mesh_value[i] + 1.0) * range / 2.0 + min_val;
    }
}

// ======================= VOID COUNT =======================
long long int void_count(Points *points) {

    long long int voids = 0;

    #pragma omp parallel for reduction(+:voids)
    for (int i = 0; i < NUM_Points; i++) {
        voids += (int)points[i].is_void;
    }

    return voids;
}

// ======================= SAVE =======================
void save_mesh(double *mesh_value) {

    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }

    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}