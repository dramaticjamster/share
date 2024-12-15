#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <iostream>


// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

inline int xy_to_idx(int pt_idx, int dim_idx, int num_pts){
  return pt_idx+dim_idx*num_pts;
}

inline int idx_to_x(int idx, int num_pts){
  return idx % num_pts;
}

inline int idx_to_y(int idx, int num_pts){
  return idx/num_pts;
}

__device__ float fsquare(float x){
  return x * x;
}

__device__ void _print_cuda_array(float* x, int n){
  printf("\n[");
  for (int i = 0; i < n; i++){
    printf("%e ", x[i]);
  };
  printf("]\n");
}

__device__ void _print_cuda_array(int* x, int n){
  printf("\n[");
  for (int i = 0; i < n; i++){
    printf("%d ", x[i]);
  };
  printf("]\n");
}

__global__ void _htod_print_array(int* x, int n){
  if (threadIdx.x+blockIdx.x == 0){
    printf("\n[");
    for (int i = 0; i < n; i++){
      printf("%d ", x[i]);
    };
    printf("]\n");
  }
}

__global__ void _htod_print_array(float* x, int n){
  if (threadIdx.x+blockIdx.x == 0){
    printf("\n[");
    for (int i = 0; i < n; i++){
      printf("%f ", x[i]);
    };
    printf("]\n");
  }
}

__global__ void _cuda_dtod_memcpy(float* dest, float* src, int n){
  int const_idx = threadIdx.x + blockDim.x*blockIdx.x;
  while (const_idx < n){
    dest[const_idx] = src[const_idx];
    const_idx += blockDim.x;
    __syncthreads();
  }
}

__global__ void _cuda_dtod_memcpy(int* dest, int* src, int n){
  int const_idx = threadIdx.x + blockDim.x*blockIdx.x;
  while (const_idx < n){
    dest[const_idx] = src[const_idx];
    const_idx += blockDim.x;
    __syncthreads();
  }
}

inline int sharedmem_to_batchsize(int sharedmem, int dim){
  return sharedmem/(sizeof(float)*(dim+2)) - 1;
}

void swap_pointers(void **ptr1, void **ptr2) {
    void *temp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = temp;
}

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

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

__device__ float _dist(float* a, float* b, int n)
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

__global__ void _arrays_equal(int* a, int* b, int n, bool* result){
  if (threadIdx.x + blockIdx.x == 0){
    for (int i = 0; i < n; i++){
      if (a[i] != b[i]){
        *result = false;
        return;
      }
    }
    *result = true;
    return;
  }
}

__global__ void _arrays_equal(float* a, float* b, int n, bool* result){
  if (threadIdx.x + blockIdx.x == 0){
    for (int i = 0; i < n; i++){
      if (fabs(a[i] - b[i]) > 1.0){
        *result = false;
        return;
      }
    }
    *result = true;
    return;
  }
}




__global__ void classify_points(float *points, float* clusters, int* classes,
      int num_pts, int dim, int num_cls, int sharedmem_per_block, int batchsize,
      float fmax){
  // int const_idx = threadIdx.x+blockDim.x*blockIdx.x; // create typical 1D thread index from built-in variables
  int idx = threadIdx.x;
  int block_offset = blockIdx.x*batchsize;
  float cur_dist;
  // while ...
  // printf("\nBlock %d, thread %d active, block_offset = %d\n", blockIdx.x, threadIdx.x, block_offset);

  // Initialize shared memory
  extern __shared__ char s[];
  float* pts = (float *) s;
  float* cur_min_dist = (float*)&pts[batchsize*dim];
  float* cls = (float*)&cur_min_dist[batchsize];
  int* cur_min_cls = (int*)&cls[dim];
  // int cur_min_dist_offset = batchsize*dim;
  // __syncthreads();
  // int cls_offset = cur_min_dist_offset+batchsize;

  while (block_offset < num_pts){

    // copy batch of points to shared
    int num_pts_in_batch = min(batchsize, num_pts - block_offset);
    while (idx < num_pts_in_batch*dim){
        pts[idx] = points[block_offset*dim + idx];
        idx += blockDim.x;
      }
      idx = threadIdx.x;
    
    // Initialize cur_min_dist
    while (idx < num_pts_in_batch){
        cur_min_dist[idx] = fmax;
        idx += blockDim.x;
      }

    for (int c = 0; c < num_cls; c++) {
    // for (int c = 0; c < 2; c++) {
      // Initialize a single cluster point
      idx = threadIdx.x;
      while (idx < dim){
        cls[idx] = clusters[c*dim + idx];
        idx += blockDim.x;
      }
      idx = threadIdx.x;

      // __syncthreads();
      // if (const_idx == 0){
      //   printf("Doing c = %d:\n", c);
      //   printf("Current cluster target: [");
      //   for (int k = 0; k < dim; k++) {
      //     printf("%f ", cls[k]);
      //   }
      //   printf("]\n");
      //   printf("Correct cluster target: [");
      //   for (int k = 0; k < dim; k++) {
      //     printf("%f ", clusters[c*dim + k]);
      //   }
      //   printf("]\n");
      // }


      __syncthreads();

      // Calculate distance to current cluster
      cur_dist = 0.;
      while (idx < num_pts_in_batch){
        for (int d = 0; d < dim; d++) {
          cur_dist += fsquare(pts[idx*dim+d] - cls[d]);
        }
        // printf("\n(block, thread, point_x, cur_dist) = (%d, %d: %f, %f)\n", 
        //     blockIdx.x, threadIdx.x, pts[idx*dim], cur_dist);

        // Update min dist, cls index of min dist
        if (cur_dist < cur_min_dist[idx]){
          cur_min_dist[idx] = cur_dist;
          cur_min_cls[idx] = c;
        }
        cur_dist = 0.;
        // Increment to next point
        idx += blockDim.x;
      }

      // __syncthreads();
      // if (const_idx == 0){
      //   printf("\npts in shared: [");
      //   for (int i = 0; i < num_pts_in_batch*dim; i++){
      //     printf("%f ", pts[i]);
      //   }
      //   printf("]\n");
      //   printf("\npts in cls: [");
      //   for (int i = 0; i < dim; i++){
      //     printf("%f ", cls[i]);
      //   }
      //   printf("]\n");
      //   printf("\npts in cur_min_dist: [");
      //   for (int i = 0; i < num_pts_in_batch; i++){
      //     printf("%f ", cur_min_dist[i]);
      //   }
      //   printf("]\n");
      //   printf("\npts in cur_min_cls: [");
      //   for (int i = 0; i < num_pts_in_batch; i++){
      //     printf("%i ", cur_min_cls[i]);
      //   }
      //   printf("]\n");
      // }

      __syncthreads();

    }

    // write classifications from shared mem to global, increment batch
    idx = threadIdx.x;
    while (idx < num_pts_in_batch){
      classes[block_offset + idx] = cur_min_cls[idx];
      idx += blockDim.x;
    }
    idx = threadIdx.x;

    block_offset += gridDim.x*batchsize;
  }
}



__global__ void ref_avg_dist_to_self_cluster(float *points, float* avg_dist, int* cluster_indices, int* classes,
  int num_pts, int dim, int num_cls, int sharedmem_per_block){
    int const_idx = threadIdx.x+blockDim.x*blockIdx.x;
    while (const_idx < num_pts){
      avg_dist[const_idx] = 0.0;

      const_idx += gridDim.x*blockDim.x;
    }
    const_idx = threadIdx.x + blockDim.x*blockIdx.x;

    __syncthreads();


    while (const_idx < num_pts){
      int self_cls = classes[const_idx];
      float* self_pt = &points[const_idx*dim];
      for (int p = 0; p < num_pts; p++){
        int target_cls = classes[p];
        
        if (self_cls == target_cls){
          float* target_pt = &points[p*dim];
          float dist = _dist(self_pt, target_pt, dim);
          atomicAdd(&avg_dist[const_idx], dist);
        }
      }

      const_idx += blockDim.x*gridDim.x;
    }
  }



// Given points and classes, calculates for each point the sum of distance to 
// all points within same cluster, and stores distance to avg_dist
__global__ void avg_dist_to_self_cluster(float *points, float* avg_dist, int* cluster_indices, int* classes,
  int num_pts, int dim, int num_cls, int sharedmem_per_block,
  int suggested_batchsize){
  
  
  // calculate batch size (points held by block at a time) to maximize shared mem use
  int batchsize = suggested_batchsize;
  if (suggested_batchsize <= 0){
  int batchsize = (sharedmem_per_block - 2*num_cls*sizeof(int) - dim*sizeof(float))
                  / ((dim+2)*sizeof(float) + sizeof(int));
  }
  // int batchsize = 9;

  int const_idx = threadIdx.x+blockDim.x*blockIdx.x; // create typical 1D thread index from built-in variables
  int idx = threadIdx.x;
  int block_offset = blockIdx.x*batchsize;
  int points_idx;

  

  // float* avg_dist = clusters;

  // int batchsize = (sharedmem_per_block - );
  // // while ...
  // // printf("\nBlock %d, thread %d active, block_offset = %d\n", blockIdx.x, threadIdx.x, block_offset);

  // Define shared memory
  extern __shared__ char s[];
  float* pts = (float *) s;
  float* dist_sum = (float*)&pts[batchsize*dim];
  float* presqrt_dist_terms = (float*)&dist_sum[batchsize];
  int* prev_points_idx = (int*)&presqrt_dist_terms[batchsize];
  int* cls_to_idx_ncls = (int*)&prev_points_idx[batchsize];  // cls index -> (start idx of cls in pts, num el in cls)
  float* cur_pt = (float*)&cls_to_idx_ncls[2*num_cls];




  // while (block_offset < num_pts){
  while (block_offset < num_pts){
    int num_pts_in_batch = min(batchsize, num_pts - block_offset);
    // if (blockIdx.x == 0 && threadIdx.x == 0){
    //   printf("  block offset, num_pts_in_batch = %d, %d\n", block_offset, num_pts_in_batch);
    // }

    // Initialize shared memory
    // Count number of pts in batch belonging to each cluster
    while (idx < num_cls){
      cls_to_idx_ncls[2*idx] = 0;
      cls_to_idx_ncls[2*idx+1] = 0;
      idx += blockDim.x;
    }
    idx = threadIdx.x;
    __syncthreads();

    while (idx < num_pts_in_batch){
      points_idx = block_offset + idx;
      int target_pt_cls = classes[points_idx];
      atomicAdd(&cls_to_idx_ncls[2*(target_pt_cls)+1], 1);

      idx += blockDim.x;
    }
    idx = threadIdx.x;
    __syncthreads();

    while (idx < num_pts_in_batch){
      dist_sum[idx] = 0.0;

      idx += blockDim.x;
    }
    idx = threadIdx.x;


    // For each cluster, set pts index where pts of that cluster begins
    if (threadIdx.x == 0){
      int cur_idx_sum = 0;
      for (int i = 0; i < num_cls; i++){
        cls_to_idx_ncls[2*i] = cur_idx_sum;
        cur_idx_sum += cls_to_idx_ncls[2*i+1];
        
        // if (blockIdx.x == 1){
        //   printf("\ncur_idx_sum = %d, increment by %d\n", cur_idx_sum, cls_to_idx_ncls[2*i+1]);
        // }
      }
    }

    __syncthreads();

    // if (blockIdx.x == 0 && threadIdx.x == 0){
    //   printf("\nFinal cls_to_idx_ncls: [(%d, %d) (%d, %d) (%d, %d)]\n",
    //         cls_to_idx_ncls[0], cls_to_idx_ncls[1],
    //         cls_to_idx_ncls[2], cls_to_idx_ncls[3],
    //         cls_to_idx_ncls[4], cls_to_idx_ncls[5]);
    // }

    // Store points in pts based on reordering, saving previous index
    while (idx < num_cls){
      int save_at_pts_idx = cls_to_idx_ncls[2*idx];

      // Look through all points in batch
      for (int i = 0; i < num_pts_in_batch; i++){
        points_idx = block_offset + i;

        // If current target point belongs to cls under current thread
        if (idx == classes[points_idx]){
          // save original index
          // prev_points_idx[save_at_pts_idx] = points_idx;
          prev_points_idx[i] = save_at_pts_idx;
          // save point to shared mem
          for (int j = 0; j < dim; j++){
            pts[save_at_pts_idx*dim+j] = points[points_idx*dim+j];
          }
          save_at_pts_idx++;
        }
      }

      idx += blockDim.x;
    }
    idx = threadIdx.x;
    __syncthreads();


    // if (blockIdx.x == 0 && threadIdx.x == 0){
    //   printf("\nbreak 1:\n");
    //   _print_cuda_array(cls_to_idx_ncls, 2*num_cls);
    //   _print_cuda_array(pts, num_pts_in_batch*dim);
    // }

    // Go through points, summing distance for pts with matching cls
    for (int p = 0; p < num_pts; p++){
      int point_cls = classes[p];

      // If there are els in pts matching the class
      int cls_idx = cls_to_idx_ncls[2*point_cls];
      int ncls = cls_to_idx_ncls[2*point_cls+1];
      if (ncls > 0){
        // Initialize presqrt_dist_terms scratchpad
        while (idx < num_pts_in_batch){
          presqrt_dist_terms[idx] = 0.0;
          idx += blockDim.x;
        }
        idx = threadIdx.x;
        

        // Save target point to shared mem
        while (idx < dim){
          cur_pt[idx] = points[p*dim+idx];
          idx += blockDim.x;
        }
        idx = threadIdx.x;

        __syncthreads();

        // if (blockIdx.x == 1 && threadIdx.x == 0){
        //   printf("points %d retrieved as [%0.04f, %0.04f]\n", p, points[p*dim], points[p*dim+1]);
        // }


        // calculate pre sqrt dist term for each pts and target point
        for (int ci = 0; ci < ncls; ci++){
          int _pts_offset = cls_idx + ci;
          int _term;
          // calculate distance b/t cur_pt[_] and pts[(cls_idx + ci)*dim + _]
          // if (threadIdx.x == 0){
          //   float d = _dist(cur_pt, &pts[_pts_offset*dim], dim);
          //   atomicAdd(&dist_sum[_pts_offset], d);
          //   if (blockIdx.x == 0){
          //     printf("dist (<%f, %f>, <%f, %f>) = %f\n  dist_sum = [%f, %f, %f]\n\n", 
          //           cur_pt[idx], cur_pt[idx+1],
          //           pts[_pts_offset*dim + idx], pts[_pts_offset*dim + idx + 1],
          //           d,
          //           dist_sum[0], dist_sum[1], dist_sum[2]);
          //   }
          // }



          while (idx < dim){
            _term = fsquare(cur_pt[idx] - pts[_pts_offset*dim + idx]);
            atomicAdd(&presqrt_dist_terms[_pts_offset], _term);

            idx += blockDim.x;
          }
          idx = threadIdx.x;
        }

        __syncthreads();




        // calc squareroot and sum to dist_sum
        while (idx < ncls){
          int _pts_offset = cls_idx + idx;
          dist_sum[_pts_offset] += sqrt(presqrt_dist_terms[_pts_offset]);
          
          idx += blockDim.x;
        }
        idx = threadIdx.x;
      }
    }

    __syncthreads();
    // if (blockIdx.x == 0 && threadIdx.x == 0){
    //     printf("\n  final dist_sum (blockidx, blockoffset: %d, %d) = [%f, %f, %f]\n\n",
    //           blockIdx.x, block_offset,
    //           dist_sum[0], dist_sum[1], dist_sum[2]);
    // }


    // Save dist_sum to original index in avg_dist
    while (idx < num_pts_in_batch){
      // if (blockIdx.x == 1){
      //   printf("\n[idx, prev_points_idx: %d, %d], dist_sum[%d] = %f\n  Saved to avg_dist[%d]\n",
      //         idx, prev_points_idx[idx], prev_points_idx[idx], dist_sum[prev_points_idx[idx]],
      //         block_offset*batchsize + idx);
      // }
      avg_dist[block_offset + idx] = dist_sum[prev_points_idx[idx]];

      // prev_points_idx[save_at_pts_idx] = points_idx;
      // int original_idx = prev_points_idx[idx];
      // points[block_offset*dim + idx]
      idx += blockDim.x;
    }
    idx = threadIdx.x;

    // if (blockIdx.x == 1 && threadIdx.x == 0){
    //   printf("current block_offset = %d, batchsize = %d, num_pts_in_batch = %d\n",
    //     block_offset, batchsize, num_pts_in_batch);
    // }
    block_offset += gridDim.x*batchsize;
  }
  block_offset = blockIdx.x*batchsize;


  __syncthreads();

  

  // if (blockIdx.x == 1 && threadIdx.x == 0){
  //   printf("final block_offset = %d, batchsize = %d, num_pts_in_batch = %d\n", 
  //   block_offset, batchsize, min(batchsize, num_pts - block_offset));
  //   _print_cuda_array(cls_to_idx_ncls, num_cls*2);
  //   _print_cuda_array(prev_points_idx, min(num_pts, batchsize));
    
  //   _print_cuda_array(dist_sum, batchsize);
  //   _print_cuda_array(avg_dist, num_pts);
  // }
}


__global__ void ref_select_new_clusters(float *points, float* avg_dist, int* cluster_indices, int* classes,
  int num_pts, int dim, int num_cls, int sharedmem_per_block, float fmax){
  int const_idx = threadIdx.x+blockDim.x*blockIdx.x;
  int idx = threadIdx.x;
  float* clusters = avg_dist;

  if (threadIdx.x + blockIdx.x == 0){

    for (int c = 0; c < num_cls; c++){
      float cur_min = fmax;
      int cur_min_idx = -1;

      for (int p = 0; p < num_pts; p++){
        if (classes[p] == c && avg_dist[p] < cur_min){
          cur_min = avg_dist[p];
          cur_min_idx = p;
        }
      }

      cluster_indices[c] = cur_min_idx;
    }

    for (int c = 0; c < num_cls; c++){
      int medoid_idx_of_cluster_c = cluster_indices[c];

      // copy medoid data from point to clusters
      for (int d = 0; d < dim; d++){
        clusters[c*dim + d] = points[medoid_idx_of_cluster_c*dim + d];
      }
    }
  }
}


__global__ void select_new_clusters(float *points, float* avg_dist, int* cluster_indices, int* classes,
  int num_pts, int dim, int num_cls, int sharedmem_per_block, float fmax){

  int const_idx = threadIdx.x+blockDim.x*blockIdx.x; // create typical 1D thread index from built-in variables
  int idx = threadIdx.x;
  int block_cls_idx = blockIdx.x;
  float* clusters = avg_dist;

  // Define shared memory, assumes assigned shared can hold 1 float and int for each thread
  extern __shared__ char s[];
  float* cur_min_dist = (float *) s;
  int* cur_min_dist_idx = (int*) &cur_min_dist[blockDim.x];

  // Assign a class for each block
  while (block_cls_idx < num_cls){
    int cur_block_cls = block_cls_idx;

    // Initialize shared memory
    cur_min_dist[threadIdx.x] = fmax;
    cur_min_dist_idx[threadIdx.x] = -1;

    // Iterate over points to find min of current class (cur_cls)
    while (idx < num_pts){
      // if point class == assigned class, update min if necessary
      int target_cls = classes[idx];
      float target_dist = avg_dist[idx];
      if ((cur_block_cls == target_cls) &&
          (target_dist < cur_min_dist[threadIdx.x])){
        cur_min_dist[threadIdx.x] = target_dist;
        cur_min_dist_idx[threadIdx.x] = idx;
        
      }

      idx += blockDim.x;
    }
    __syncthreads();
    idx = threadIdx.x;
    

    // Find min within shared memory
    float cls_min;
    int cls_min_idx;
    if (threadIdx.x == 0){

      cls_min = cur_min_dist[0];
      cls_min_idx = cur_min_dist_idx[0];
      
      for (int i = 1; i < blockDim.x; i++){
        if (cur_min_dist[i] < cls_min){
          cls_min = cur_min_dist[i];
          cls_min_idx = cur_min_dist_idx[i];
        }
      }

      // Save new medoid into cluster_indices
      cluster_indices[cur_block_cls] = cls_min_idx;
    }
    __syncthreads();
    block_cls_idx += gridDim.x;
  }
  block_cls_idx = blockIdx.x;
  

  // Using new cluster_indices, save new medoids to clusters
  while (block_cls_idx < num_cls){
    int point_idx_of_cluster = cluster_indices[block_cls_idx];
    while (idx < dim){
      clusters[block_cls_idx*dim + idx] = points[point_idx_of_cluster*dim + idx];
      idx += blockDim.x;
    }
    idx = threadIdx.x;

    block_cls_idx += gridDim.x;
  }
  block_cls_idx = blockIdx.x;

}


void km_main(int grid_size, int block_size,
             float *points, int *classes, float *clusters, int *cluster_indexes, int *new_cluster_indexes,
             int num_pts, int num_cl, int dim, int max_iter){
  float *d_points, *d_clusters, *d_avg_dist;
  int *d_classes, *d_cluster_indexes, *d_new_cluster_indexes;

  double start_time, end_time;


  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0); // 0 is the default device
  size_t freeMem, totalMem;


  // Allocate memory for points, classes, clusters and cluster_indexes on the GPU
  cudaMalloc((void **)&d_points, num_pts * dim * sizeof(float));
  cudaMalloc((void **)&d_classes, num_pts * sizeof(int));
  cudaMalloc((void **)&d_clusters, num_cl * dim * sizeof(float));
  cudaMalloc((void **)&d_cluster_indexes, num_cl*sizeof(int));
  cudaMalloc((void **)&d_new_cluster_indexes, num_cl*sizeof(int));
  d_avg_dist = d_clusters;

  cudaDeviceSynchronize();
  

  // Copy data into GPU
  cudaMemcpy(d_points, points, num_pts * dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_classes, classes, num_pts * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_clusters, clusters, num_cl * dim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_cluster_indexes, cluster_indexes, num_cl*sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_new_cluster_indexes, cluster_indexes, num_cl * sizeof(int), cudaMemcpyHostToDevice);
  
  // float *temp, *d_temp;
  // cudaMalloc((void **)&d_temp, num_pts*sizeof(float));
  // temp = (float *) malloc(num_pts*sizeof(float));
  // column_sums<<<4, 16>>>(d_points, d_temp, dim);

  // cudaMemcpy(temp, d_temp, dim, cudaMemcpyDeviceToHost);

  // _print_array(temp, dim);
  // _print_array(points, num_pts*dim);
  // printf("\nd_points:");
  // _print_cuda_array<<<8, 8>>>(d_points, num_pts * dim );
  // cudaDeviceSynchronize();
  // printf("\nd_clusters:");
  // _print_cuda_array<<<8, 8>>>(d_clusters, num_cl * dim );
  // cudaDeviceSynchronize();
  // printf("\nd_classes:");
  // _print_cuda_array<<<8, 8>>>(d_classes, num_pts );
  // cudaDeviceSynchronize();



  // Find available shared memory to request
  cudaGetDeviceProperties(&prop, 0);
  int sharedMemBytes = prop.sharedMemPerBlock; // shared memory per block (32 bytes per thread)
  int sms = prop.multiProcessorCount;         // number of multiprocessors on the device
  int maxSharedMem = sharedMemBytes * sms;    // total shared memory size for the deviceint sharedMemBytes = prop.sharedMemPerBlock;
  
  // printf("mem required = %ld\n", sizeof(float)*512*500);
  // printf("sizeof(float) = %ld\n", sizeof(float));
  // printf("processor count, shared memory per block = %d, %d\n", sms, sharedMemBytes);
  // printf("batchsize = %d\n", sharedmem_to_batchsize(sharedMemBytes, dim));

  int iter = 0;
  bool converge, *d_converge_ptr;
  converge = false;
  cudaMalloc(&d_converge_ptr, sizeof(bool));

  printf("Starting parallel region...\n");
  start_time = monotonic_seconds();
  end_time = start_time;
  // Parallel region
  // Classify points using existing clusters
  // max_iter = 2;
  while (iter < max_iter && !converge){
    iter++;
    
    classify_points<<<grid_size, block_size, sharedMemBytes>>>(d_points, d_clusters, d_classes,
        num_pts, dim, num_cl, sharedMemBytes, sharedmem_to_batchsize(sharedMemBytes, dim),
        std::numeric_limits<float>::max());
    cudaDeviceSynchronize();

    // printf("\nOriginal cluster indexes:");
    // _htod_print_array<<<1,1>>>(d_cluster_indexes, num_cl);
    // cudaDeviceSynchronize();

    printf("  [Iter %d]: Classification time: %0.04f\n", iter, monotonic_seconds()-end_time);
    end_time = monotonic_seconds();

    int suggested_batchsize = -1;
    avg_dist_to_self_cluster<<<grid_size, block_size, sharedMemBytes>>>(
        d_points, d_avg_dist, d_new_cluster_indexes, d_classes,
        num_pts, dim, num_cl, sharedMemBytes,
        suggested_batchsize);
    cudaDeviceSynchronize();

    printf("  [Iter %d]: Sum dist calc time: %0.04f\n", iter, monotonic_seconds()-end_time);
    end_time = monotonic_seconds();


    // printf("\nOriginal sum dist:");
    // _htod_print_array<<<1,1>>>(d_avg_dist, num_pts);
    // cudaDeviceSynchronize();
    // printf("\nOriginal classes:");
    // _htod_print_array<<<1,1>>>(d_classes, num_pts);
    // cudaDeviceSynchronize();


    select_new_clusters<<<grid_size, block_size, sharedMemBytes>>>(
        d_points, d_avg_dist, d_new_cluster_indexes, d_classes,
        num_pts, dim, num_cl, sharedMemBytes,
        std::numeric_limits<float>::max());
    cudaDeviceSynchronize();

    printf("  [Iter %d]: Select medoid time: %0.04f\n", iter, monotonic_seconds()-end_time);
    end_time = monotonic_seconds();


    // Test convergence
    _arrays_equal<<<grid_size, block_size>>>(d_new_cluster_indexes, d_cluster_indexes, num_cl, d_converge_ptr);
    cudaMemcpy(&converge, d_converge_ptr, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("  [Iter %d/%d]: Total iter %d time: %0.04f\n", iter, max_iter, iter, monotonic_seconds()-end_time);
    end_time = monotonic_seconds();
    printf("  [Iter %d/%d]: converge = %d\n", iter, max_iter, converge);
    printf("  [Iter %d/%d]: stop? = %d\n", iter, max_iter, converge || (iter > max_iter));

    // Stop and copy data if converged, print duration
    if (converge || (iter >= max_iter)){
      cudaDeviceSynchronize();
      printf("\n\nCONVERGED, stopping and saving\n");
      end_time = monotonic_seconds();
      print_time(end_time-start_time);

      iter = max_iter+1;
      cudaMemcpy(classes, d_classes, num_pts * sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(clusters, d_clusters, num_cl * dim * sizeof(float), cudaMemcpyDeviceToHost);
    }


    // printf("\nclusters:");
    // _htod_print_array<<<1,1>>>(d_clusters, num_cl*dim);
    // cudaDeviceSynchronize();
    // printf("\ncluster indexes:");
    // _htod_print_array<<<1,1>>>(d_cluster_indexes, num_cl);
    // cudaDeviceSynchronize();
    // printf("new cluster indexes:");
    // _htod_print_array<<<1,1>>>(d_new_cluster_indexes, num_cl);
    // cudaDeviceSynchronize();

    swap_pointers((void**) &d_cluster_indexes, (void**) &d_new_cluster_indexes);
  }

  cudaDeviceSynchronize();
  

  cudaFree(d_converge_ptr);
  cudaFree(d_points);
  cudaFree(d_classes);
  cudaFree(d_clusters);
  cudaFree(d_cluster_indexes);
  cudaFree(d_new_cluster_indexes);
}

