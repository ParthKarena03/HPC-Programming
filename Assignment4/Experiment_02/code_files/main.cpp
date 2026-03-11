#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
long NUM_Points;
int Maxiter;
double dx, dy;

int main()
{
    int NX_list[3] = {250,500,1000};
    int NY_list[3] = {100,200,400};

    NUM_Points = 1e8;
    Maxiter = 10;

    omp_set_num_threads(4);

    printf("Problem\tTotal Interpolation Time\n");

    for(int cfg=0; cfg<3; cfg++)
    {
        NX = NX_list[cfg];
        NY = NY_list[cfg];

        GRID_X = NX + 1;
        GRID_Y = NY + 1;

        dx = 1.0 / NX;
        dy = 1.0 / NY;

        double *mesh_value = (double*)calloc(GRID_X * GRID_Y, sizeof(double));
        Points *points = (Points*)calloc(NUM_Points, sizeof(Points));

        initializepoints(points);

        double total_interp = 0.0;

        for(int iter=0; iter<Maxiter; iter++)
        {
            clock_t start = clock();

            interpolation(mesh_value, points);

            clock_t end = clock();

            double time = (double)(end - start) / CLOCKS_PER_SEC;
            total_interp += time;
        }

        printf("%d\t%lf\n", cfg+1, total_interp);

        free(mesh_value);
        free(points);
    }

    return 0;
}