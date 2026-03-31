#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include <stdint.h>
#include "utils.h"

// ================= RNG =================

static inline uint32_t xor_shift(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static inline double uniform_random(uint32_t *s) {
    return xor_shift(s) * (1.0 / 4294967295.0);
}

// ================= Interpolation =================
// (Keep serial — avoids heavy atomic overhead)

void interpolation(double *mesh_value, Points *points) {

    double inv_dx = 1.0 / dx;
    double inv_dy = 1.0 / dy;
    double scale  = 1.0 / (NX * NY);

    for (int i = 0; i < NUM_Points; i++) {

        double x = points[i].x;
        double y = points[i].y;

        int ix = (int)(x * inv_dx);
        int iy = (int)(y * inv_dy);

        if (ix < 0) ix = 0;
        else if (ix >= GRID_X - 1) ix = GRID_X - 2;

        if (iy < 0) iy = 0;
        else if (iy >= GRID_Y - 1) iy = GRID_Y - 2;

        double fx = x * inv_dx - ix;
        double fy = y * inv_dy - iy;

        double w1 = (1.0 - fx) * (1.0 - fy) * scale;
        double w2 = fx * (1.0 - fy) * scale;
        double w3 = (1.0 - fx) * fy * scale;
        double w4 = fx * fy * scale;

        int idx = iy * GRID_X + ix;

        mesh_value[idx]              += w1;
        mesh_value[idx + 1]          += w2;
        mesh_value[idx + GRID_X]     += w3;
        mesh_value[idx + GRID_X + 1] += w4;
    }
}

// ================= Save =================

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

// ================= SERIAL =================

void mover_serial_imm(Points *points, double deltaX, double deltaY) {

    uint32_t state = 987654321u;

    for (int i = 0; i < NUM_Points; i++) {

        double rx = uniform_random(&state);
        double ry = uniform_random(&state);

        points[i].x += (rx * 2.0 * deltaX - deltaX);
        points[i].y += (ry * 2.0 * deltaY - deltaY);

        if (points[i].x < 0.0 || points[i].x > 1.0 ||
            points[i].y < 0.0 || points[i].y > 1.0) {

            points[i].x = uniform_random(&state);
            points[i].y = uniform_random(&state);
        }
    }
}

// ================= PARALLEL IMM (FASTEST) =================

void mover_parallel_imm(Points *__restrict points, double deltaX, double deltaY) {

    #pragma omp parallel
    {
        uint32_t state = 1234567u * (omp_get_thread_num() + 1);

        #pragma omp for
        for (int i = 0; i < NUM_Points; i++) {

            double rx = uniform_random(&state);
            double ry = uniform_random(&state);

            double x = points[i].x + (rx * 2.0 * deltaX - deltaX);
            double y = points[i].y + (ry * 2.0 * deltaY - deltaY);

            if (x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0) {
                points[i].x = x;
                points[i].y = y;
            } else {
                points[i].x = uniform_random(&state);
                points[i].y = uniform_random(&state);
            }
        }
    }
}

// ================= PARALLEL DIFF (OPTIMIZED) =================

void mover_parallel_diff(Points *__restrict points, double deltaX, double deltaY) {

    static Points *temp = NULL;
    if (!temp)
        temp = (Points*) malloc(NUM_Points * sizeof(Points));

    int nthreads = omp_get_max_threads();
    int *count = (int*) calloc(nthreads, sizeof(int));

    // Step 1: count survivors
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t state = 1234567u * (tid + 1);

        int local_count = 0;

        #pragma omp for
        for (int i = 0; i < NUM_Points; i++) {

            double rx = uniform_random(&state);
            double ry = uniform_random(&state);

            double x = points[i].x + (rx * 2.0 * deltaX - deltaX);
            double y = points[i].y + (ry * 2.0 * deltaY - deltaY);

            if (x >= 0.0 && x <= 1.0 && y >= 0.0 && y <= 1.0)
                local_count++;
        }

        count[tid] = local_count;
    }

    // prefix sum
    int *offset = (int*) malloc(nthreads * sizeof(int));
    offset[0] = 0;
    for (int i = 1; i < nthreads; i++)
        offset[i] = offset[i-1] + count[i-1];

    int survivors = offset[nthreads-1] + count[nthreads-1];
    int deleted = NUM_Points - survivors;

    // Step 2: fill temp
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint32_t state = 1234567u * (tid + 1);

        int pos = offset[tid];

        #pragma omp for
        for (int i = 0; i < NUM_Points; i++) {

            double rx = uniform_random(&state);
            double ry = uniform_random(&state);

            Points p = points[i];
            p.x += (rx * 2.0 * deltaX - deltaX);
            p.y += (ry * 2.0 * deltaY - deltaY);

            if (p.x >= 0.0 && p.x <= 1.0 && p.y >= 0.0 && p.y <= 1.0)
                temp[pos++] = p;
        }
    }

    free(count);
    free(offset);

    // Step 3: insert new
    #pragma omp parallel
    {
        uint32_t state = 7654321u * (omp_get_thread_num() + 1);

        #pragma omp for
        for (int i = 0; i < deleted; i++) {
            temp[survivors + i].x = uniform_random(&state);
            temp[survivors + i].y = uniform_random(&state);
        }
    }

    memcpy(points, temp, NUM_Points * sizeof(Points));
}

// ================= NO INSERT/DELETE =================

void mover_parallel_no_ins_del(Points *points, double deltaX, double deltaY) {

    #pragma omp parallel
    {
        uint32_t state = 13579u * (omp_get_thread_num() + 1);

        #pragma omp for
        for (int i = 0; i < NUM_Points; i++) {

            double rx = uniform_random(&state);
            double ry = uniform_random(&state);

            points[i].x += (rx * 2.0 * deltaX - deltaX);
            points[i].y += (ry * 2.0 * deltaY - deltaY);
        }
    }
}