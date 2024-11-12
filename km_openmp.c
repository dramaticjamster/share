#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

#define MAX_ITER 20

/* Gives us high-resolution timers. */
#define _POSIX_C_SOURCE 199309L
#include <time.h>

/* OSX timer includes */
#ifdef __MACH__
  #include <mach/mach.h>
  #include <mach/mach_time.h>
#endif

/**
* @brief Return the number of seconds since an unspecified time (e.g., Unix
*        epoch). This is accomplished with a high-resolution monotonic timer,
*        suitable for performance timing.
*
* @return The number of seconds.
*/
static inline double monotonic_seconds()
{
#ifdef __MACH__
  /* OSX */
  static mach_timebase_info_data_t info;
  static double seconds_per_unit;
  if(seconds_per_unit == 0) {
    mach_timebase_info(&info);
    seconds_per_unit = (info.numer / info.denom) / 1e9;
  }
  return seconds_per_unit * mach_absolute_time();
#else
  /* Linux systems */
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/**
* @brief Output the seconds elapsed while clustering.
*
* @param seconds Seconds spent on k-medoids clustering, excluding IO.
*/
static void print_time(double const seconds)
{
  printf("k-medoids clustering time: %0.04fs\n", seconds);
}


float _dist(float* a, float* b, int n)
{ 
  float result = 0.0;
  for (int i=0; i<n; i++){
    result += pow(a[i]-b[i], 2.);
  }

  result = pow(result, 0.5);
  return result;
}

void _add_to_index(float* a, float* b, int idx, int dim){
  float* curr = a + (idx * dim);
  for (int i = 0; i < dim; i++){
    curr[i] += b[i];
  }
}

void _add_vec(float* acc, float* b, int dim){
  for (int i = 0; i < dim; i++){
    acc[i] += b[i];
  }
}

void _copy_vec(float* dest, float* source, int dim){
  for (int i = 0; i < dim; i++){
    dest[i] = source[i];
  }
}

void _initialize_vec(float* a, int n){
  for (int i = 0; i < n; i++){
    a[i] = 0.;
  }
}

void _initialize_vec(float* a, int n, float val){
  for (int i = 0; i < n; i++){
    a[i] = val;
  }
}

void _initialize_vec(int* a, int n){
  for (int i = 0; i < n; i++){
    a[i] = 0;
  }
}

void _initialize_vec(int* a, int n, int val){
  for (int i = 0; i < n; i++){
    a[i] = val;
  }
}

bool _vec_equal(int* a, int* b, int dim){
  for (int i = 0; i < dim; i++){
    if (a[i] != b[i]){
      return false;
    }
  }
  return true;
}

void _print_array(float *a, int n){
  printf("\n[");
  for (int i = 0; i<n; i++){
    printf("%f ", a[i]);
  }
  printf("]\n");
}

void _print_array(int *a, int n){
  printf("\n[");
  for (int i = 0; i<n; i++){
    printf("%d ", a[i]);
  }
  printf("]\n");
}


int main ( int argc, char *argv[] )
{
  int id;
  if (argc < 4) {
    printf("Error: missing argument, format [filename num_clusters num_threads]\n");
    return 1;
  }

  // printf("Argument 1: %s\n", argv[1]);

  int num_cl = strtod(argv[2], NULL);
  int num_thread = strtod(argv[3], NULL);

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
  char line[1024];
  fgets(line, sizeof(line), file);
  // printf("First line of file: %s", line);

  // Parse first line of format %i %i into two ints, int a and int b.
  int num_pts, dim;
  if (sscanf(line, "%i %i", &num_pts, &dim) != 2) {
    printf("Error: could not parse line\n");
    return 1;
  }

  // Dynamically allocate array named points for n-by-d floats.
  float *points = (float *)malloc(num_pts * dim * sizeof(float));
  int *classes = (int *)malloc(num_pts * sizeof(int));
  float *clusters = (float *)malloc(num_cl * dim * sizeof(float));
  int *cluster_indexes = (int *)malloc(num_cl*sizeof(int));
  int *new_cluster_indexes = (int *)malloc(num_cl*sizeof(int));

  _initialize_vec(classes, 0);

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
      points[dim*i+j] = val;
      token = strtok(NULL, delimiter);
    }
  }

  // Close the file
  fclose(file);

  // Initialize clusters
  for (int i = 0; i < num_cl; i++){
    for (int j = 0; j < dim; j++){
      clusters[dim*i + j] = points[dim*i + j];
    }
    cluster_indexes[i] = i;
  }

  // _print_array(clusters, num_cl * dim);



float* cluster_avgs = (float*)malloc(num_cl * dim * sizeof(float));
_initialize_vec(cluster_avgs, num_cl*dim);
float* cluster_count = (float*)malloc(num_cl * sizeof(float));
_initialize_vec(cluster_count, num_cl);
bool converge = false;
int iter = 0;

double start_time = monotonic_seconds();

/*
  INSIDE THE PARALLEL REGION
*/
omp_set_num_threads(num_thread);
# pragma omp parallel \
  private ( id )
  {
    id = omp_get_thread_num ( );
    int i = id;

    float* accumulate_clusters = (float*)malloc(num_cl * dim * sizeof(float));
    float* accumulate_cls_count = (float*)malloc(num_cl * sizeof(float));

    int divisible_num_pts = num_thread* (num_pts/num_thread);
    int remainder = num_pts - divisible_num_pts;
    if (remainder > 0) {
      divisible_num_pts += num_thread;
    }

    while (!converge && (iter < MAX_ITER)){
      _initialize_vec(accumulate_clusters, num_cl * dim);
      _initialize_vec(accumulate_cls_count, num_cl);
      
      i = id;
      // Classify points using existing clusters
      while (i < divisible_num_pts){
        if (i < num_pts) {
          int curr_class = 0;
          float* curr_pt = points + (i*dim);
          float* curr_cluster = clusters;
          float curr_min_dist = _dist(curr_pt, clusters, dim);
          for (int j = 1; j < num_cl; j++){
            curr_cluster = clusters + (j*dim);
            float d = _dist(curr_pt, curr_cluster, dim);

            if (d < curr_min_dist){
              curr_class = j;
              curr_min_dist = d;
            }
          }

          classes[i] = curr_class;

          // Add current point to accumulated sum of assigned cluster
          _add_to_index(accumulate_clusters, curr_pt, curr_class, dim);

          // Update point count of clusters
          accumulate_cls_count[curr_class] += 1.;
        }

        i += num_thread;
        // Barrier to ensure cache locality
        #pragma omp barrier
      }


      // Find average
      #pragma omp critical
      {
        _add_to_index(cluster_avgs, accumulate_clusters, 0, num_cl*dim);
        _add_vec(cluster_count, accumulate_cls_count, num_cl);
      }
      #pragma omp barrier
      #pragma omp master
      {
        for (int i = 0; i < num_cl; i++){
          for (int j = 0; j < dim; j++){
            cluster_avgs[i*dim + j] /= cluster_count[i];
          }
        }
      }
      

      // Find closest point to each cluster and update cluster
      float* accumulate_min_dist = accumulate_cls_count;
      _initialize_vec(accumulate_min_dist, num_cl, -1.);
      int* accumulate_min_index = (int*) accumulate_clusters;
      _initialize_vec(accumulate_min_index, num_cl, -1);

      int pt = id;
      #pragma omp barrier
      while (pt < divisible_num_pts){
        if (pt < num_pts){
          int cls = classes[pt];
          float* curr_pt = points + (pt*dim);
          float* cluster_avg = cluster_avgs + (cls*dim);
          float d = _dist(curr_pt, cluster_avg, dim);
          if ((accumulate_min_dist[cls] < 0.) || (d < accumulate_min_dist[cls])){
            accumulate_min_dist[cls] = d;
            accumulate_min_index[cls] = pt;
          }
        }

        

        pt += num_thread;
        #pragma omp barrier
      }


      // Collate potential cluster centers
      float *cluster_dist = cluster_count;
      #pragma omp master
      {
        _initialize_vec(cluster_dist, num_cl, -1.);
      }
      #pragma omp barrier

      #pragma omp critical
      {
        for (int cid = 0; cid < num_cl; cid++){
          if (cluster_dist[cid] < 0 ||
              (cluster_dist[cid] > accumulate_min_dist[cid] && accumulate_min_dist[cid] >= 0)){
            cluster_dist[cid] = accumulate_min_dist[cid];
            new_cluster_indexes[cid] = accumulate_min_index[cid];
          }
        }
      }
      #pragma omp barrier

      #pragma omp master
      {
        iter++;
        if (_vec_equal(new_cluster_indexes, cluster_indexes, num_cl)){
          converge = true;
        } else {
          int *temp = new_cluster_indexes;
          for (int i = 0; i < num_cl; i++){
            _copy_vec(clusters + i*dim, points + new_cluster_indexes[i]*dim, dim);
          }
          new_cluster_indexes = cluster_indexes;
          cluster_indexes = temp;
        }

      }

      #pragma omp barrier
    }

  }
/*
  Terminate.
*/

  double end_time = monotonic_seconds();
  print_time(end_time-start_time);

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
      fprintf(medoids_file, "%f", curr[i]);
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