#ifndef INIT_H
#define INIT_H

void init_matrices(int Np, double ***m1, double ***m2, double ***mt, double ***result);
void free_matrices(int Np, double** m1, double** m2, double** mt, double** result);

#endif