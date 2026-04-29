#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

/* ======================= GLOBAL VARIABLES ======================= */
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;
double min_val, max_val;

/* ======================= DATA STRUCTURES ======================= */
typedef struct
{
    double x, y;
    int is_void;
} Points;

/* ======================= INITIALIZATION FUNCTIONS ======================= */
void initializepoints(Points *points)
{
    int i;
    for (i = 0; i < NUM_Points; i++)
    {
        points[i].x = (double)rand() / RAND_MAX;
        points[i].y = (double)rand() / RAND_MAX;
        points[i].is_void = 0;
    }
}

void read_points(FILE *file, Points *points)
{
    int i;
    for (i = 0; i < NUM_Points; i++)
    {
        fread(&points[i].x, sizeof(double), 1, file);
        fread(&points[i].y, sizeof(double), 1, file);
        points[i].is_void = 0;
    }
}

/* ======================= INTERPOLATION (OPTIMIZED) ======================= */

void interpolation(double *mesh_value, Points *points, int num_particles_local)
{
    const int size = GRID_X * GRID_Y;
    const int nt = omp_get_max_threads();

    static double **local_grid = NULL;
    static int cached_nt = 0;
    static int cached_size = 0;

    if (local_grid == NULL || cached_nt != nt || cached_size != size)
    {
        if (local_grid != NULL)
        {
            for (int t = 0; t < cached_nt; t++)
                free(local_grid[t]);
            free(local_grid);
        }

        local_grid = (double **)malloc(nt * sizeof(double *));
        for (int t = 0; t < nt; t++)
        {
            local_grid[t] = (double *)calloc(size, sizeof(double));
        }

        cached_nt = nt;
        cached_size = size;
    }

    memset(mesh_value, 0, size * sizeof(double));

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *grid = local_grid[tid];

        memset(grid, 0, size * sizeof(double));

#pragma omp for schedule(guided)
        for (int i = 0; i < num_particles_local; i++)
        {
            if (points[i].is_void)
                continue;

            double x = points[i].x;
            double y = points[i].y;

            int xi = (int)(x / dx);
            int yi = (int)(y / dy);

            if (xi < 0)
                xi = 0;
            if (xi >= GRID_X - 1)
                xi = GRID_X - 2;
            if (yi < 0)
                yi = 0;
            if (yi >= GRID_Y - 1)
                yi = GRID_Y - 2;

            double Xi = xi * dx;
            double Yj = yi * dy;

            double lx = x - Xi;
            double ly = y - Yj;

            double w_ij = (dx - lx) * (dy - ly);
            double w_i1j = lx * (dy - ly);
            double w_ij1 = (dx - lx) * ly;
            double w_i1j1 = lx * ly;

            int idx = yi * GRID_X + xi;

            grid[idx] += w_ij;
            grid[idx + 1] += w_i1j;
            grid[idx + GRID_X] += w_ij1;
            grid[idx + GRID_X + 1] += w_i1j1;
        }
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++)
    {
        double sum = 0.0;
        for (int t = 0; t < nt; t++)
        {
            sum += local_grid[t][i];
        }
        mesh_value[i] = sum;
    }
}

void normalization(double *mesh_value)
{
    const int size = GRID_X * GRID_Y;

    double mn = mesh_value[0];
    double mx = mesh_value[0];

#pragma omp parallel for reduction(min : mn) reduction(max : mx)
    for (int i = 0; i < size; i++)
    {
        double v = mesh_value[i];
        if (v < mn)
            mn = v;
        if (v > mx)
            mx = v;
    }

    min_val = mn;
    max_val = mx;

    double range = mx - mn;
    if (range == 0.0)
        range = 1.0;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++)
    {
        mesh_value[i] = 2.0 * (mesh_value[i] - mn) / range - 1.0;
    }
}

/* ======================= MOVER ======================= */

void mover(double *mesh_value, Points *points, int num_particles_local)
{
#pragma omp parallel for schedule(guided)
    for (int i = 0; i < num_particles_local; i++)
    {
        if (points[i].is_void)
            continue;

        double x = points[i].x;
        double y = points[i].y;

        int xi = (int)(x / dx);
        int yi = (int)(y / dy);

        if (xi < 0)
            xi = 0;
        if (xi >= GRID_X - 1)
            xi = GRID_X - 2;
        if (yi < 0)
            yi = 0;
        if (yi >= GRID_Y - 1)
            yi = GRID_Y - 2;

        double Xi = xi * dx;
        double Yj = yi * dy;

        double lx = x - Xi;
        double ly = y - Yj;

        double w_ij = (dx - lx) * (dy - ly);
        double w_i1j = lx * (dy - ly);
        double w_ij1 = (dx - lx) * ly;
        double w_i1j1 = lx * ly;

        int idx = yi * GRID_X + xi;

        double Fi =
            w_ij * mesh_value[idx] +
            w_i1j * mesh_value[idx + 1] +
            w_ij1 * mesh_value[idx + GRID_X] +
            w_i1j1 * mesh_value[idx + GRID_X + 1];

        points[i].x = x + Fi * dx;
        points[i].y = y + Fi * dy;

        if (points[i].x < 0.0 || points[i].x > 1.0 ||
            points[i].y < 0.0 || points[i].y > 1.0)
        {
            points[i].is_void = 1;
        }
    }
}

/* ======================= DENORMALIZATION ======================= */

void denormalization(double *mesh_value)
{
    const int size = GRID_X * GRID_Y;
    double range = max_val - min_val;
    if (range == 0.0)
        range = 1.0;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++)
    {
        mesh_value[i] = (mesh_value[i] + 1.0) * range / 2.0 + min_val;
    }
}

/* ======================= VOID COUNT ======================= */
long long int void_count(Points *points, int num_particles_local)
{
    long long int voids = 0;

#pragma omp parallel for reduction(+ : voids)
    for (int i = 0; i < num_particles_local; i++)
    {
        voids += (int)points[i].is_void;
    }

    return voids;
}

/* ======================= SAVE ======================= */

void save_mesh(double *mesh_value, int rank)
{
    if (rank != 0)
        return;

    FILE *fd = fopen("Mesh.out", "w");
    if (!fd)
    {
        printf("Error creating Mesh.out\n");
        exit(1);
    }

    for (int i = 0; i < GRID_Y; i++)
    {
        for (int j = 0; j < GRID_X; j++)
        {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}

/* ======================= MAIN ======================= */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc != 2)
    {
        if (rank == 0)
            printf("Usage: %s <input_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    FILE *file = NULL;
    if (rank == 0)
    {
        file = fopen(argv[1], "rb");
        if (!file)
        {
            printf("Error opening input file\n");
            MPI_Finalize();
            exit(1);
        }

        // Read grid dimensions
        fread(&NX, sizeof(int), 1, file);
        fread(&NY, sizeof(int), 1, file);

        // Read number of Points and max iterations
        fread(&NUM_Points, sizeof(int), 1, file);
        fread(&Maxiter, sizeof(int), 1, file);
    }

    // ================= BROADCAST SIMULATION PARAMETERS =================
    MPI_Bcast(&NX, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NY, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NUM_Points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Maxiter, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Since Number of points will be 1 more than number of cells
    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    // ================= PARTICLE DECOMPOSITION =================
    int particles_per_proc = NUM_Points / num_procs;
    int remainder = NUM_Points % num_procs;

    int num_particles_local = particles_per_proc + (rank < remainder ? 1 : 0);
    int start_idx = rank * particles_per_proc + (rank < remainder ? rank : remainder);

    // Allocate memory for grid and local points
    double *mesh_value = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
    double *local_grid = (double *)calloc(GRID_X * GRID_Y, sizeof(double));
    Points *points = (Points *)calloc(num_particles_local, sizeof(Points));

    // ================= READ AND SCATTER PARTICLES =================
    if (rank == 0)
    {
        // Read all particles (on rank 0)
        Points *all_points = (Points *)calloc(NUM_Points, sizeof(Points));
        read_points(file, all_points);

        // Scatter particles to other processes
        for (int p = 1; p < num_procs; p++)
        {
            int p_particles_per_proc = particles_per_proc + (p < remainder ? 1 : 0);
            int p_start_idx = p * particles_per_proc + (p < remainder ? p : remainder);
            MPI_Send(&all_points[p_start_idx], p_particles_per_proc * sizeof(Points),
                     MPI_BYTE, p, 0, MPI_COMM_WORLD);
        }

        // Copy rank 0's particles
        memcpy(points, all_points, num_particles_local * sizeof(Points));
        free(all_points);
        fclose(file);
    }
    else
    {
        // Receive particles on other processes
        MPI_Recv(points, num_particles_local * sizeof(Points),
                 MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    double total_int_time = 0.0;
    double total_norm_time = 0.0;
    double total_move_time = 0.0;
    double total_denorm_time = 0.0;

    for (int iter = 0; iter < Maxiter; iter++)
    {

        double t0 = omp_get_wtime();

// Reset local grid
#pragma omp parallel for
        for (int i = 0; i < GRID_X * GRID_Y; i++)
        {
            local_grid[i] = 0.0;
        }

        // Perform interpolation on local particles
        interpolation(local_grid, points, num_particles_local);

        // ================= MPI_ALLREDUCE: COMBINE LOCAL GRIDS =================
        MPI_Allreduce(local_grid, mesh_value, GRID_X * GRID_Y, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double t1 = omp_get_wtime();

        normalization(mesh_value);

        double t2 = omp_get_wtime();

        // Perform mover on local particles using global grid
        mover(mesh_value, points, num_particles_local);

        double t3 = omp_get_wtime();

        denormalization(mesh_value);

        double t4 = omp_get_wtime();

        total_int_time += (double)(t1 - t0);
        total_norm_time += (double)(t2 - t1);
        total_move_time += (double)(t3 - t2);
        total_denorm_time += (double)(t4 - t3);
    }

    // ================= ONLY RANK 0 SAVES OUTPUT =================
    save_mesh(mesh_value, rank);

    // ================= GATHER VOID COUNT =================
    long long int local_voids = void_count(points, num_particles_local);
    long long int total_voids = 0;
    MPI_Reduce(&local_voids, &total_voids, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Total Interpolation Time = %lf seconds\n", total_int_time);
        printf("Total Normalization Time = %lf seconds\n", total_norm_time);
        printf("Total Mover Time = %lf seconds\n", total_move_time);
        printf("Total Denormalization Time = %lf seconds\n", total_denorm_time);
        printf("Total Algorithm Time = %lf seconds\n", total_int_time + total_norm_time + total_move_time + total_denorm_time);
        printf("Total Number of Voids = %lld\n", total_voids);
    }

    // Free memory
    free(mesh_value);
    free(local_grid);
    free(points);

    // ================= MPI FINALIZATION =================
    MPI_Finalize();

    return 0;
}
