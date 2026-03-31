#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "init.h"

void interpolation(double *mesh_value, Points *points);
void mover_serial_diff(Points *points, double deltaX, double deltaY);
void mover_serial_imm(Points *points, double deltaX, double deltaY);
void mover_parallel_diff(Points *points, double deltaX, double deltaY);
void mover_parallel_imm(Points *points, double deltaX, double deltaY);
void save_mesh(double *mesh_value);
void mover_parallel_no_ins_del(Points *points, double deltaX, double deltaY);

#endif
 
