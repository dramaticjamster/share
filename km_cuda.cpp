#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#include "km_cuda_functions.h"

#define MAX_ITER 2

inline int norm_xy_to_idx(int pt_idx, int dim_idx, int dim){
  return dim_idx + pt_idx*dim;
}

inline int xy_to_idx(int pt_idx, int dim_idx, int num_pts){
  return pt_idx+dim_idx*num_pts;
}

inline int idx_to_x(int idx, int num_pts){
  return idx % num_pts;
}

inline int idx_to_y(int idx, int num_pts){
  return idx/num_pts;
}

int main ( int argc, char *argv[] )
{
  int id;
  if (argc < 5) {
    printf("Error: missing argument, format [filename num_clusters grid_size block_size]\n");
    return 1;
  }

  // printf("Argument 1: %s\n", argv[1]);

  int num_cl = strtod(argv[2], NULL);
  int grid_size = strtod(argv[3], NULL);
  int block_size = strtod(argv[4], NULL);


  // printf("Num clusters, threads = %d, %d\n", num_cl, num_thread);

  // FILE *testfile = fopen("test.txt", "w");
  // if (testfile == NULL) {
  //     printf("Error: could not open file\n");
  //     return 1;
  // }
  // for (int i = 0; i < 500000; i++) {
  //     fprintf(testfile, "%d %d %d %d %d %d\n", rand()%1001, rand()%1001, rand()%1001, rand()%1001, rand()%1001, rand()%1001);
  // }
  // fclose(testfile);



  // Open the file specified by argv[1]
  FILE *file = fopen(argv[1], "r");
  if (file == NULL) {
    printf("Error: could not open file\n");
    return 1;
  }

  // Print the first line of the file
  // char* line = (char*) malloc(sizeof(char) * 64 * 1024);
  char line[64*1024];
  fgets(line, sizeof(line), file);
  // printf("First line of file: %s", line);

  // Parse first line of format %i %i into two ints, int a and int b.
  int num_pts, dim;
  if (sscanf(line, "%i %i", &num_pts, &dim) != 2) {
    printf("Error: could not parse line\n");
    return 1;
  }

  // free(line);

  // Dynamically allocate array named points for n-by-d floats.
  // line = (char*) malloc(dim * 64 * sizeof(char));
  float *points = (float *)malloc(num_pts * dim * sizeof(float));
  int *classes = (int *)malloc(num_pts * sizeof(int));
  float *clusters = (float *)malloc(num_cl * dim * sizeof(float));
  int *classes_copy = (int *)malloc(num_pts * sizeof(int));
  float *clusters_copy = (float *)malloc(num_cl * dim * sizeof(float));
  int *cluster_indexes = (int *)malloc(num_cl*sizeof(int));
  int *new_cluster_indexes = (int *)malloc(num_cl*sizeof(int));
  double *time_ptr = (double *)malloc(sizeof(double));
  // line = (char *)malloc(128*dim*sizeof(char));

  

  _initialize_vec(classes, 0);
  _initialize_vec(classes_copy, 0);


  // Loop through remaining lines in the file.
  for (int i = 0; i < num_pts; i++) {
    // Read a line from the file.
    fgets(line, sizeof(line), file);

    // Convert the line to a string.
    char *str = strdup(line);
    // printf("String: %s\n", str);

    char* token;
    char delimiter[] = " ";
    token = strtok(str, delimiter);

    for (int j=0; j<dim; j++){
      float val = strtod(token, NULL);
      // printf("%f, ", val);
      // Normal ordering, (a0, a1, a2), (b0, b1, b2), etc.
      // points[dim*i+j] = val;
      // Ordered by axis, e.g. (x0, x1, ..., xn), (y0, ..., yn), etc. for n+1 points
      int store_at_idx = norm_xy_to_idx(i, j, dim);
      points[store_at_idx] = val;
      // printf("[%d -> %f]\n", store_at_idx, val);
      token = strtok(NULL, delimiter);
    }
  }

  // Close the file
  fclose(file);
  // free(line);

  // Initialize clusters
  for (int i = 0; i < num_cl; i++){
    for (int j = 0; j < dim; j++){
      clusters[norm_xy_to_idx(i, j, dim)] = points[norm_xy_to_idx(i, j, dim)];
      clusters_copy[norm_xy_to_idx(i, j, dim)] = points[norm_xy_to_idx(i, j, dim)];
    }
  }

  for (int i = 0; i < num_cl; i++){
    cluster_indexes[i] = i;
  }



float* cluster_avgs = (float*)malloc(num_cl * dim * sizeof(float));
_initialize_vec(cluster_avgs, num_cl*dim);
float* cluster_count = (float*)malloc(num_cl * sizeof(float));
_initialize_vec(cluster_count, num_cl);
bool converge = false;
int iter = 0;

// double start_time = monotonic_seconds();

/*
  INSIDE THE PARALLEL REGION
*/

// km_main(grid_size, block_size, points, classes, clusters, cluster_indexes, new_cluster_indexes,
//         num_pts, num_cl, dim, MAX_ITER);


int _grid_size_n = 9;
int _grid_size[_grid_size_n] = {16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32};
int _block_size_n = 3;
int _block_size[_block_size_n] = {1024, 512, 256};
double cur_best_time = std::numeric_limits<double>::max();
int cur_best[] = {0, 0};

for (int b_idx = 0; b_idx < _block_size_n; b_idx++){
  for (int g_idx = 0; g_idx < _grid_size_n; g_idx++){
    int g = _grid_size[0];
    int b = _block_size[b_idx];

    km_main(g, b, points, classes, clusters, cluster_indexes, new_cluster_indexes,
        num_pts, num_cl, dim, MAX_ITER, time_ptr);

    printf("\n  Current run (g, b: %d, %d), time elapsed = %.04f\n", g, b, *time_ptr);
    if (*time_ptr < cur_best_time){
      cur_best_time = *time_ptr;
      cur_best[0] = g;
      cur_best[1] = b;
    }
    printf("    Current best: (%d, %d) at %.04f sec\n", cur_best[0], cur_best[1], cur_best_time);
  }
}
/*
  Terminate.
*/

  // double end_time = monotonic_seconds();
  // print_time(end_time-start_time);

  // printf("\n\nHost environment:\n");
  // printf("  classes array for clusters_file:\n");
  // _print_array(classes, num_pts);
  // printf("  clusters array for medoids_file:\n");
  // _print_array(clusters, num_cl*dim);

  // Open file for writing
  FILE *clusters_file = fopen("clusters.txt", "w");
  if (clusters_file == NULL) {
      printf("Error: could not open output file\n");
      return 1;
  }

  // Write cluster indexes to the file
  for (int i = 0; i < num_pts; i++) {
      fprintf(clusters_file, "%d\n", classes[i]);
      // if ((i + 1) % dim == 0) {
      //     fprintf(output_file, "\n");
      // }
  }

  // Close the output file
  fclose(clusters_file);


  // Open file for writing
  FILE *medoids_file = fopen("medoids.txt", "w");
  if (medoids_file == NULL) {
      printf("Error: could not open output file\n");
      return 1;
  }

  // Write cluster indexes to the file
  for (int i = 0; i < num_cl; i++) {
    float* curr = clusters + i*dim;
    for (int j = 0; j < dim; j++) {
      fprintf(medoids_file, "%.9f", curr[j]);
      if (j < dim-1) {
        fprintf(medoids_file, " ");
      } else {
        fprintf(medoids_file, "\n");
      }
    }
  }

  // Close the output file
  fclose(medoids_file);


  // Free allocated memory
  free(points);
  free(classes);
  free(clusters);
  free(cluster_indexes);
  free(new_cluster_indexes);
  free(cluster_avgs);
  free(cluster_count);




  return 0;
}
