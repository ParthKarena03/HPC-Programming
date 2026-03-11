# HPC Assignment 04: Particle Interpolation and Mover Performance Analysis

This repository contains the skeleton code for Assignment 04. In this lab, you will analyze the performance of the Cloud-In-Cell bilinear interpolation scheme, introduce a particle tracking phase (the Mover), and parallelize the Mover operation using OpenMP.

## Code Structure

The codebase provided here acts as a starting point. **Please note that the core physics and interpolation functions are provided as empty placeholders. You must implement the logic yourself.**

*   `init.h` / `init.cpp`: Contains the point data structure, global simulation variables, and the `initializepoints()` function which randomly initializes particles within the domain. 
*   `utils.h` / `utils.cpp`: Contains the placeholders for `interpolation()`, `mover_serial()`, and `mover_parallel()`. **You need to write your implementations here.** It also includes a working `save_mesh()` function for outputting the grid.
*   `main.cpp`: The main driver code. The simulation parameters (like `NX`, `NY`, `NUM_Points`, and `Maxiter`) are hardcoded directly in this file. You will need to modify these parameters manually to conduct the required experiments.

## Your Tasks

1. **Interpolation:** Implement the serial bilinear interpolation logic inside the `interpolation()` function.
2. **Serial Mover:** Implement the particle tracking logic inside `mover_serial()`. Make sure to apply boundary checks so that the particles remain strictly within the $1 \times 1$ grid.
3. **Parallel Mover:** Implement an OpenMP parallelized version of your mover inside `mover_parallel()`.
4. **Experiments:** Modify `main.cpp` (or split it into separate files for each experiment) to adjust the hardcoded variables and perform the three experiments detailed in your lab manual:
    *   **Experiment 01:** Scaling with Number of Particles.
    *   **Experiment 02:** Iteration Consistency.
    *   **Experiment 03:** The Mover Operation and OpenMP Parallelization.

## Compilation and Execution Commands

Since the code utilizes OpenMP (`<omp.h>`), you must compile it using the `-fopenmp` flag. 

**1. Compile the code:**
```bash
g++ main.cpp init.cpp utils.cpp -lm -fopenmp -o main
```
**12. Execute the compiled binary:**
```bash
./main
```