#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>



static void print_time(double const seconds);


float _dist(float* a, float* b, int n);

void _add_to_index(float* a, float* b, int idx, int dim);

void _add_vec(float* acc, float* b, int dim);

void _copy_vec(float* dest, float* source, int dim);

void _initialize_vec(float* a, int n);

void _initialize_vec(float* a, int n, float val);

void _initialize_vec(int* a, int n);

void _initialize_vec(int* a, int n, int val);

bool _vec_equal(int* a, int* b, int dim);

void _print_array(float *a, int n);

void _print_array(int *a, int n);

void km_main(int grid_size, int block_size,
             float *points, int *classes, float *clusters, int *cluster_indexes, int *new_cluster_indexes,
             int num_pts, int num_cl, int dim, int max_iter);




