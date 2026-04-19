#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "init.h"

// Random particle initialization (optional)
void initializepoints(Points *points) {

    // Seed once (important for randomness)
    static int seeded = 0;
    if (!seeded) {
        srand(time(NULL));
        seeded = 1;
    }

    for (int i = 0; i < NUM_Points; i++) {
        points[i].x = (double)rand() / RAND_MAX;
        points[i].y = (double)rand() / RAND_MAX;
    }
}

// Read particle positions from binary file (safe version)
void read_points(FILE *file, Points *points) {

    for (int i = 0; i < NUM_Points; i++) {

        if (fread(&points[i].x, sizeof(double), 1, file) != 1) {
            printf("Error reading x at index %d\n", i);
            exit(1);
        }

        if (fread(&points[i].y, sizeof(double), 1, file) != 1) {
            printf("Error reading y at index %d\n", i);
            exit(1);
        }
    }
}