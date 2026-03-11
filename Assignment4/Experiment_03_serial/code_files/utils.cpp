#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

// Interpolation (Serial Code)
void interpolation(double *mesh_value, Points *points) {
        double y0 = 1.0 / dx;
    double y1 = 1.0 / dy;
    double y2 = 1.0 / (NX * NY);

    for (int y3 = 0; y3 < NUM_Points; y3++)
    {
        double y4 = points[y3].x;
        double y5 = points[y3].y;

        int y6 = (int)(y4 * y0);
        int y7 = (int)(y5 * y1);

        if (y6 < 0) y6 = 0;
        else if (y6 >= GRID_X - 1) y6 = GRID_X - 2;

        if (y7 < 0) y7 = 0;
        else if (y7 >= GRID_Y - 1) y7 = GRID_Y - 2;

        double y8 = y4 * y0 - y6;
        double y9 = y5 * y1 - y7;

        double y10 = (1.0 - y8) * (1.0 - y9) * y2;
        double y11 = y8 * (1.0 - y9) * y2;
        double y12 = (1.0 - y8) * y9 * y2;
        double y13 = y8 * y9 * y2;

        int y14 = y7 * GRID_X + y6;

        mesh_value[y14]              += y10;
        mesh_value[y14 + 1]          += y11;
        mesh_value[y14 + GRID_X]     += y12;
        mesh_value[y14 + GRID_X + 1] += y13;
    }
}

// Stochastic Mover (Serial Code) 
void mover_serial(Points *points, double deltaX, double deltaY) {
    for (int i = 0; i < NUM_Points; i++) {
        points[i].x += deltaX;
        points[i].y += deltaY;
    }
}

// Stochastic Mover (Parallel Code) 
void mover_parallel(Points *points, double deltaX, double deltaY) {
    #pragma omp parallel for
    for (int i = 0; i < NUM_Points; i++) {
        points[i].x += deltaX;
        points[i].y += deltaY;
    }
}

// Write mesh to file
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