#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main()
{
    int NX_list[3] = {250,500,1000};
    int NY_list[3] = {100,200,400};

    long points_range[5] = {1e2,1e4,1e6,1e8,1e9};

    Maxiter = 10;

    omp_set_num_threads(4);

    for(int cfg=0; cfg<3; cfg++)
    {
        NX = NX_list[cfg];
        NY = NY_list[cfg];

        GRID_X = NX + 1;
        GRID_Y = NY + 1;

        dx = 1.0/NX;
        dy = 1.0/NY;

        printf("\n===== Configuration %d (NX=%d NY=%d) =====\n",cfg+1,NX,NY);

        for(int p=0; p<5; p++)
        {
            NUM_Points = points_range[p];

            double *mesh_value = (double*)calloc(GRID_X*GRID_Y,sizeof(double));
            Points *points = (Points*)calloc(NUM_Points,sizeof(Points));

            initializepoints(points);

            double total_interp = 0.0;

            for(int iter=0; iter<Maxiter; iter++)
            {
                clock_t start = clock();

                interpolation(mesh_value,points);

                clock_t end = clock();

                double time = (double)(end-start)/CLOCKS_PER_SEC;
                total_interp += time;
            }

            printf("Points = %ld   Total Interp Time = %lf\n",points_range[p],total_interp);

            free(mesh_value);
            free(points);
        }
    }

    return 0;
}