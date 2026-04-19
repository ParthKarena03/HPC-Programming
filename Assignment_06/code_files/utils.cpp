#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

void interpolation(double *mesh_value, Points *points) {

    double inv_dx = (double)NX;
    double inv_dy = (double)NY;

    int total_size = GRID_X * GRID_Y;

    int nthreads = omp_get_max_threads();

    // Allocate thread-local meshes
    static double **all_mesh = NULL;
    static int allocated_threads = 0;

    if (all_mesh == NULL || allocated_threads != nthreads) {

        if (all_mesh != NULL) {
            for (int t = 0; t < allocated_threads; t++) {
                free(all_mesh[t]);
            }
            free(all_mesh);
        }

        all_mesh = (double **)malloc(nthreads * sizeof(double *));
        for (int t = 0; t < nthreads; t++) {
            all_mesh[t] = (double *)calloc(total_size, sizeof(double));
        }

        allocated_threads = nthreads;
    }

    // Reset local meshes
    #pragma omp parallel for
    for (int t = 0; t < nthreads; t++) {
        memset(all_mesh[t], 0, total_size * sizeof(double));
    }

    // Compute
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *local_mesh = all_mesh[tid];

        #pragma omp for schedule(static)
        for (int p = 0; p < NUM_Points; p++) {

            double x = points[p].x;
            double y = points[p].y;

            int i = (int)(x * inv_dx);
            int j = (int)(y * inv_dy);

            if (i >= NX) i = NX - 1;
            if (j >= NY) j = NY - 1;

            double lx = x - (i * dx);
            double ly = y - (j * dy);

            double wx_m = dx - lx;
            double wy_m = dy - ly;

            int base_idx = j * GRID_X + i;

            local_mesh[base_idx]              += wx_m * wy_m;
            local_mesh[base_idx + 1]          += lx * wy_m;
            local_mesh[base_idx + GRID_X]     += wx_m * ly;
            local_mesh[base_idx + GRID_X + 1] += lx * ly;
        }
    }

    int chunk = 1024;  // tuning parameter

    #pragma omp parallel for schedule(static)
    for (int start = 0; start < total_size; start += chunk) {

        int end = start + chunk;
        if (end > total_size) end = total_size;

        for (int idx = start; idx < end; idx++) {
            double sum = 0.0;
            for (int t = 0; t < nthreads; t++) {
                sum += all_mesh[t][idx];
            }
            mesh_value[idx] += sum;
        }
    }
}
// Write mesh to file
void save_mesh(double *mesh_value) {

    FILE *fd = fopen("Mesh1.out", "w");
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