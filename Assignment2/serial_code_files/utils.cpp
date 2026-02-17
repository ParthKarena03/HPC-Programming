#include <math.h>
#include <stdlib.h>
#include "utils.h"

// Problem 01

// ijk
void matmul_ijk(double **m1, double **m2, double **result, int N)
{
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            for(int k = 0; k < N; k++)
                result[i][j] += m1[i][k] * m2[k][j];
}

// ikj
void matmul_ikj(double **m1, double **m2, double **result, int N)
{
    for(int i = 0; i < N; i++)
        for(int k = 0; k < N; k++)
            for(int j = 0; j < N; j++)
                result[i][j] += m1[i][k] * m2[k][j];
}

// jik
void matmul_jik(double **m1, double **m2, double **result, int N)
{
    for(int j = 0; j < N; j++)
        for(int i = 0; i < N; i++)
            for(int k = 0; k < N; k++)
                result[i][j] += m1[i][k] * m2[k][j];
}

// jki
void matmul_jki(double **m1, double **m2, double **result, int N)
{
    for(int j = 0; j < N; j++)
        for(int k = 0; k < N; k++)
            for(int i = 0; i < N; i++)
                result[i][j] += m1[i][k] * m2[k][j];
}

// kij
void matmul_kij(double **m1, double **m2, double **result, int N)
{
    for(int k = 0; k < N; k++)
        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++)
                result[i][j] += m1[i][k] * m2[k][j];
}

// kji

void matmul_kji(double **m1, double **m2, double **result, int N)
{
    for(int k = 0; k < N; k++)
        for(int j = 0; j < N; j++)
            for(int i = 0; i < N; i++)
                result[i][j] += m1[i][k] * m2[k][j];
}
// Problem 02
void transpose(double** m, double** mt, int N) {
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            mt[i][j] = m[j][i];
        }
    }
}

void transposed_matrix_multiplication(double** m1, double** m2, double** mt, double** result, int N) {
    transpose(m2, mt, N);

    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            for(int k=0;k<N;k++)
            {
                result[i][j]+=(m1[i][k]*mt[j][k]);
            }
        }
    }
}

// Problem 03
void block_matrix_multiplication(double** m1, double** m2, double** result, int B, int N) {
    int n = N/B;
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            for (int k=0;k<n;k++)
            {
                for(int l=(i*B);l<(i+1)*B;l++)
                {
                    for(int m=(j*B);m<(j+1)*B;m++)
                    {
                        for(int q=(k*B);q<(k+1)*B;q++)
                        {
                            result[l][m]+=(m1[l][q]*m2[q][m]);
                        }
                    }
                }
            }
            
         
        }
    }
}