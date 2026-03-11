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

void mover_parallel(Points *points)
{
#pragma omp parallel for
    for(long i=0;i<NUM_Points;i++)
    {
        double rx, ry;

        do{
            rx=((double)rand()/RAND_MAX)*2*dx-dx;
            ry=((double)rand()/RAND_MAX)*2*dy-dy;

        }while(points[i].x+rx<0 || points[i].x+rx>1 ||
               points[i].y+ry<0 || points[i].y+ry>1);

        points[i].x+=rx;
        points[i].y+=ry;
    }
}

int main()
{
    NX=1000;
    NY=400;

    NUM_Points=14000000;
    Maxiter=10;

    GRID_X=NX+1;
    GRID_Y=NY+1;

    dx=1.0/NX;
    dy=1.0/NY;

    omp_set_num_threads(4);

    double *mesh_value=(double*)calloc(GRID_X*GRID_Y,sizeof(double));
    Points *points=(Points*)calloc(NUM_Points,sizeof(Points));

    initializepoints(points);

    printf("Iter\tInterp\tMover\tTotal\n");

    for(int iter=0;iter<Maxiter;iter++)
    {
        clock_t s1=clock();
        interpolation(mesh_value,points);
        clock_t e1=clock();

        clock_t s2=clock();
        mover_parallel(points);
        clock_t e2=clock();

        double interp=(double)(e1-s1)/CLOCKS_PER_SEC;
        double mover=(double)(e2-s2)/CLOCKS_PER_SEC;
        double total=interp+mover;

        printf("%d\t%lf\t%lf\t%lf\n",iter+1,interp,mover,total);
    }

    free(mesh_value);
    free(points);

    return 0;
}