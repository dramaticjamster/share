#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <math.h>
#include <stdbool.h>

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

void _initialize_vec_val(float* a, int n, float val){
  for (int i = 0; i < n; i++){
    a[i] = val;
  }
}

void _initialize_ivec(int* a, int n){
  for (int i = 0; i < n; i++){
    a[i] = 0;
  }
}

void _initialize_ivec_val(int* a, int n, int val){
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

void _print_iarray(int *a, int n){
  printf("\n[");
  for (int i = 0; i<n; i++){
    printf("%d ", a[i]);
  }
  printf("]\n");
}


typedef struct {
    int tid;
    float *pts;
    float *curr_clusters;
    int n;
    int num_cl;
    int dim;
    int* classes;
} ThreadInput1;


void* _thread_classify_pts(void* s){
  ThreadInput1* a = (ThreadInput1 *) s;
  int pt_offset = a->tid*a->n;
  int curr_class = -1;
  float* init_pt = a->pts + (pt_offset*a->dim);
  for (int pt_idx = 0; pt_idx < a->n; pt_idx++){
    float min_dist = -1.;
    float* curr_pt = init_pt + pt_idx*a->dim;
    for (int cls = 0; cls < a->num_cl; cls++){
      float* curr_cls = a->curr_clusters + cls*a->dim;
      float d = _dist(curr_pt, curr_cls, a->dim);
      
      // If distance is smaller, update
      if ((min_dist<0) || ((d<min_dist) && (min_dist>=0))) {
        min_dist = d;
        // a->classes[pt_idx] = cls;
        curr_class = cls;
      }
      // printf("\ntid=%d\npt[%d], cls[%d] dist = %f\nmin_dist = %f\na->classes[%d] = %d, curr_class = %d\n", 
      // a->tid, pt_idx + pt_offset, cls, d, min_dist, pt_idx, a->classes[pt_idx + pt_offset], curr_class);
      
    }
    a->classes[pt_idx + pt_offset] = curr_class;
  }

  free(a);
  return NULL;
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
  float *min_dist_to_cluster = (float *)malloc(num_cl*sizeof(float));
  
  _initialize_ivec(classes, 0);

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

pthread_t thread_id[num_thread];
while (!converge && (iter < MAX_ITER))
{
  // Classify points using existing clusters
  int pts_per_thread = num_pts/num_thread;
  int remainder = num_pts - (pts_per_thread*num_thread);
  for (int tid = 0; tid < num_thread; tid++){
    ThreadInput1* inp = (ThreadInput1*)malloc(sizeof(ThreadInput1));
    inp->tid = tid;
    inp->classes = classes;
    inp->pts = points;
    inp->curr_clusters = clusters;
    inp->dim = dim;
    if (tid == num_thread-1){
      inp->n = pts_per_thread + remainder;
    } else {
      inp->n = pts_per_thread;
    }
    inp->num_cl = num_cl;
    pthread_create( &thread_id[tid], NULL, &_thread_classify_pts, inp );
  }

  // Join all threads
  for (int tid = 0; tid < num_thread; tid++){
      pthread_join(thread_id[tid], NULL);
  }



  // Calculate mean of each cluster
  // printf("\nclasses =");
  // _print_iarray(classes, num_pts);

  _initialize_vec(cluster_avgs, num_cl*dim);
  for (int pt = 0; pt < num_pts; pt++){
    int cls_id = classes[pt];
    cluster_count[cls_id]++;
    _add_vec(cluster_avgs + (cls_id*dim), points + (pt*dim), dim);
  }

  // _print_array(cluster_avgs, num_cl*dim);
  // _print_array(cluster_count, num_cl);

  for (int pt = 0; pt < num_pts; pt++){
    for (int x = 0; x < dim; x++){
      cluster_avgs[pt*dim + x] /= cluster_count[classes[pt]];

    }
  }

  // Within cluster, calculate distance to mean and keep minimum
  _initialize_vec_val(min_dist_to_cluster, num_cl, -1.);
  for (int pt = 0; pt < num_pts; pt++){
    int cls = classes[pt];
    float d = _dist(points + (pt*dim), cluster_avgs + (cls*dim), dim);
    float prev_d = min_dist_to_cluster[cls];
    if ((prev_d < 0) || (d < prev_d)){
      min_dist_to_cluster[cls] = d;
      new_cluster_indexes[cls] = pt;
    }
  }

  if (_vec_equal(new_cluster_indexes, cluster_indexes, num_cl)){
    converge = true;
  } else {
    int *temp = cluster_indexes;
    cluster_indexes = new_cluster_indexes;
    new_cluster_indexes = temp;
    iter++;
  }
}

// _print_iarray(new_cluster_indexes, num_cl);




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
  free(min_dist_to_cluster);
  free(cluster_avgs);

  return 0;
}