#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include "utils.h"

double min_val, max_val;

void interpolation(double *mesh_value, Points *points, int num_particles_local) {
    const int size = GRID_X * GRID_Y;
    const int nt = omp_get_max_threads();

    static double **local_grid = NULL;
    static int cached_nt = 0;
    static int cached_size = 0;

    if (local_grid == NULL || cached_nt != nt || cached_size != size) {
        if (local_grid != NULL) {
            for (int t = 0; t < cached_nt; t++) free(local_grid[t]);
            free(local_grid);
        }

        local_grid = (double **)malloc(nt * sizeof(double *));
        for (int t = 0; t < nt; t++) {
            local_grid[t] = (double *)calloc(size, sizeof(double));
        }

        cached_nt = nt;
        cached_size = size;
    }

    memset(mesh_value, 0, size * sizeof(double));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *grid = local_grid[tid];

        memset(grid, 0, size * sizeof(double));

        #pragma omp for schedule(guided)
        for (int i = 0; i < num_particles_local; i++) {
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

            grid[idx]              += w_ij;
            grid[idx + 1]          += w_i1j;
            grid[idx + GRID_X]     += w_ij1;
            grid[idx + GRID_X + 1] += w_i1j1;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        double sum = 0.0;
        for (int t = 0; t < nt; t++) {
            sum += local_grid[t][i];
        }
        mesh_value[i] = sum;
    }
}

void normalization(double *mesh_value) {
    const int size = GRID_X * GRID_Y;

    double mn = mesh_value[0];
    double mx = mesh_value[0];

    #pragma omp parallel for reduction(min:mn) reduction(max:mx)
    for (int i = 0; i < size; i++) {
        double v = mesh_value[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }

    min_val = mn;
    max_val = mx;

    double range = mx - mn;
    if (range == 0.0) range = 1.0;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        mesh_value[i] = 2.0 * (mesh_value[i] - mn) / range - 1.0;
    }
}

void mover(double *mesh_value, Points *points, int num_particles_local) {
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < num_particles_local; i++) {
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
            points[i].is_void = 1;
        }
    }
}

void denormalization(double *mesh_value) {
    const int size = GRID_X * GRID_Y;
    double range = max_val - min_val;
    if (range == 0.0) range = 1.0;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
        mesh_value[i] = (mesh_value[i] + 1.0) * range / 2.0 + min_val;
    }
}

long long int void_count(Points *points, int num_particles_local) {
    long long int voids = 0;

    #pragma omp parallel for reduction(+:voids)
    for (int i = 0; i < num_particles_local; i++) {
        voids += (int)points[i].is_void;
    }

    return voids;
}

void save_mesh(double *mesh_value, int rank) {
    if (rank != 0) return;

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