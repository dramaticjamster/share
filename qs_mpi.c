#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <mpi.h>
#include <stdint.h>

#define MAX_RAND 9999
#define DEBUG_PRINT 0
#define LOCAL_SAMPLE 2048

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
* @brief Output the seconds elapsed while sorting. This excludes input and
*        output time. This should be wallclock time, not CPU time.
*
* @param seconds Seconds spent sorting.
*/
static void print_time(
    double const seconds)
{
  printf("Sort Time: %0.04fs\n", seconds);
}


/**
* @brief Write an array of integers to a file.
*
* @param filename The name of the file to write to.
* @param numbers The array of numbers.
* @param nnumbers How many numbers to write.
*/
static void print_numbers(
    char const * const filename,
    uint32_t const * const numbers,
    uint32_t const nnumbers,
    int rank,
    uint32_t total_nums)
{
  FILE * fout;
  char writemode[4];
  if (rank == 0) {
    strcpy(writemode, "w");
  } else {
    strcpy(writemode, "a");
  }

  /* open file */
  if((fout = fopen(filename, writemode)) == NULL) {
    fprintf(stderr, "error opening '%s'\n", filename);
    abort();
  }

  /* write the header */
  if (rank == 0) {
    fprintf(fout, "%d\n", total_nums);
  }

  /* write numbers to fout */
  for(uint32_t i = 0; i < nnumbers; ++i) {
    fprintf(fout, "%d\n", numbers[i]);
  }

  fclose(fout);
}

void append_files(const char* source, const char* destination) {
    FILE *src = fopen(source, "r");
    if (src == NULL) {
        perror("Error opening file for reading");
        exit(EXIT_FAILURE);
    }

    FILE *dst = fopen(destination, "a");
    if (dst == NULL) {
        fclose(src);
        perror("Error opening file for appending");
        exit(EXIT_FAILURE);
    }

    char ch;
    while ((ch = fgetc(src)) != EOF)
        fputc(ch, dst);

    fclose(src);
    fclose(dst);
}

int compare(const void* a, const void* b) {
  return (*(int*)a - *(int*)b);
}

void swap(uint32_t **a, uint32_t **b) {
    uint32_t *temp = *a;
    *a = *b;
    *b = temp;
}

void swap_index(uint32_t* lst, int a, int b) {
  uint32_t temp = lst[a];
  lst[a] = lst[b];
  lst[b] = temp;
}

void print_list(uint32_t* lst, int n) {
  printf("[");
  for(uint32_t i = 0; i < n-1; ++i) {
      printf("%u, ", lst[i]);
  }
  printf("%u]\n", lst[n-1]);
}


uint32_t sort_along_pivot(uint32_t* lst, int n, uint32_t pivot) {
  int low_idx = 0;
  int high_idx = n-1;

  while (high_idx > low_idx) {
    if (lst[low_idx] < pivot){
      low_idx++;
    } else if (lst[high_idx] >= pivot) {
      high_idx--;
    } else {
      swap_index(lst, low_idx, high_idx);
    }
  }

  return (uint32_t) low_idx;
}


void list_to_str(uint32_t* lst, int len, char* buff) {
    int buffer_index = 0;
    for (int i = 0; i < len; ++i) {
        if (i > 0) {
            sprintf(&buff[buffer_index], ", ");
            buffer_index += 2;
        }
        sprintf(&buff[buffer_index], "%u", lst[i]);
        buffer_index += snprintf(NULL, 0, "%u", lst[i]);
    }
}





uint32_t findMedian(uint32_t* arr, uint32_t n) {
    // Sort the array
    qsort(arr, n, sizeof(uint32_t), compare);

    // Return the middle element in sorted array
    return arr[n / 2];
}

void findSplit(int world_size, uint32_t* proc0_buffer) {
  int left = 0;
  int right = 0;

  for (int i = 0; i < world_size; i++){
    left += proc0_buffer[2*i];
    right += proc0_buffer[1+2*i];
  }

  
}

uint32_t* endpointer(uint32_t* buffer, int buffer_size) {
  uint32_t* pointer = buffer + buffer_size - 1;
  return pointer;
}

uint32_t uint_min(uint32_t a, uint32_t b) {
  if (a<b) {
    return a;
  }
  return b;
}


void store_contiguous(uint32_t* dest, int l_len, int r_len, 
                      uint32_t* src, int s_len) {
  memcpy(dest, src, uint_min(l_len, s_len) * sizeof(uint32_t));

  if (l_len < s_len) {
    // The initial open space does not have enough space
    int remainder = s_len - l_len;
    memcpy(endpointer(dest, l_len + r_len + 1), 
           endpointer(src, l_len + 1),
           remainder*sizeof(uint32_t));
  } else if (l_len > s_len) {
    // The initial open space has too much space
    int fill = l_len - s_len;
    memcpy(endpointer(dest, s_len + 1),
           endpointer(dest, l_len + r_len - fill + 1),
           fill*sizeof(uint32_t));
  }
}

// void in_place_split()


int main ( int argc, char *argv[] )
{
  uint32_t pivot;
  if (argc < 3) {
    printf("Error: missing argument, format [num_integers output_filename]\n");
    return 1;
  }

  uint32_t num_ints = strtod(argv[1], NULL);
  char output_filename[128];
  char* debug_str;
  strcpy(output_filename, argv[2]);

  



  


  MPI_Init(&argc, &argv);
  MPI_Comm curr_comm = MPI_COMM_WORLD;
  MPI_Status* status;
  int partner_rank;

  // Find out rank, size
  int world_rank;
  MPI_Comm_rank(curr_comm, &world_rank);
  int world_size;
  MPI_Comm_size(curr_comm, &world_size);

  int num_local_ints = (int) num_ints/world_size;
  if (world_rank == 0) {
    num_local_ints += num_ints % world_size;
  }

  int data_buffer_size = 2*num_local_ints * sizeof(uint32_t);
  int local_buffer_size = 2*num_local_ints * sizeof(uint32_t);
  int proc_0_buffer_size = 2*sizeof(uint32_t) * world_size;

  uint32_t *local_buffer = (uint32_t *) malloc(local_buffer_size);
  uint32_t *local_nums = (uint32_t *) malloc(data_buffer_size);
  uint32_t *proc_0_buffer = (uint32_t *) malloc(proc_0_buffer_size);
  srand(world_rank*1000); // seed random number generator
  for(int i = 0; i < num_local_ints; ++i) {
      // local_nums[i] = (uint32_t) rand() % MAX_RAND; // generate a random number between 0 and 99
      local_nums[i] = (uint32_t) rand();
  }

  
  double start_time = monotonic_seconds();
  if ((num_ints <= LOCAL_SAMPLE) || world_size == 1) {
    qsort(local_nums, num_local_ints, sizeof(uint32_t), compare);
  } else {
    while (world_size > 1) {

      if (num_local_ints >= LOCAL_SAMPLE) {
        local_buffer[0] = findMedian(local_nums, LOCAL_SAMPLE);
      } else {
        local_buffer[0] = local_nums[0];
      }
      // qsort(local_nums, num_local_ints, sizeof(uint32_t), compare);
      // print_list(local_nums, num_local_ints);


      // printf("I am MPI process %d.\nNum_ints = %d, num_local_ints = %d, filename = %s\n", 
      // world_rank, num_ints, num_local_ints, output_filename);

      // MPI_Gather(
      //   void* send_data,
      //   int send_count,
      //   MPI_Datatype send_datatype,
      //   void* recv_data,
      //   int recv_count,
      //   MPI_Datatype recv_datatype,
      //   int root,
      //   MPI_Comm communicator)
      MPI_Gather(
        local_buffer,
        1,
        MPI_UINT32_T,
        proc_0_buffer,
        1,
        MPI_UINT32_T,
        0,
        curr_comm);
      
      if (world_rank == 0) {
        local_buffer[0] = findMedian(proc_0_buffer, world_size);

        if (DEBUG_PRINT) {
          print_list(proc_0_buffer, world_size);
          char* debug_str = (char*) malloc (1024);
          list_to_str(proc_0_buffer, world_size, debug_str);
          printf("gathered string = %s\n", debug_str);
          free(debug_str);
          printf("pivot selected = %d\n", local_buffer[0]);
        }
      }

      // Broadcast pivot
      MPI_Bcast(
        local_buffer,
        1,
        MPI_UINT32_T,
        0,
        curr_comm);
      
      pivot = local_buffer[0];

      // printf("I am process %d and I have received pivot %d\n", 
      // world_rank, pivot);

      uint32_t split = sort_along_pivot(local_nums, num_local_ints, pivot);
      
      // debug_str = (char*) malloc (1024);
      // list_to_str(local_nums, num_local_ints, debug_str);
      // printf("Process %d: final data (split %d) after pivot: %s\n", 
      // world_rank, split, debug_str);
      // free(debug_str);

      local_buffer[0] = split;
      local_buffer[1] = num_local_ints - split;

      // MPI_Gather(
      //   void* send_data,
      //   int send_count,
      //   MPI_Datatype send_datatype,
      //   void* recv_data,
      //   int recv_count,
      //   MPI_Datatype recv_datatype,
      //   int root,
      //   MPI_Comm communicator)
      MPI_Gather(
        local_buffer,
        2,
        MPI_UINT32_T,
        proc_0_buffer,
        2,
        MPI_UINT32_T,
        0,
        curr_comm);
      
      // if (world_rank == 0) {
        // print_list(proc_0_buffer, world_size);
        // char* asdf = (char*) malloc (64*world_size);
        // list_to_str(proc_0_buffer, world_size * 2, asdf);
        // printf("gathered string = %s\n", asdf);
        // free(asdf);
        // local_buffer[0] = findMedian(proc_0_buffer, world_size);
        // printf("pivot selected = %d\n", local_buffer[0]);
      // }

      MPI_Bcast(
        proc_0_buffer,
        2*world_size,
        MPI_UINT32_T,
        0,
        curr_comm);
      

      // asdf = (char*) malloc (64*world_size);
      // list_to_str(proc_0_buffer, world_size * 2, asdf);
      // printf("Process %d: gathered proc0 data = %s\n", world_rank, asdf);
      // free(asdf);


      int offset = world_size/2;
      if (world_rank < offset) {
        // send upper half data
        // MPI_Send(
        // void* data,
        // int count,
        // MPI_Datatype datatype,
        // int destination,
        // int tag,
        // MPI_Comm communicator)
        partner_rank = world_rank + offset;
        MPI_Send(
        endpointer(local_nums, split+1),
        proc_0_buffer[2*world_rank + 1],
        MPI_UINT32_T,
        partner_rank,
        0,
        curr_comm);

        MPI_Recv(
        local_buffer,
        proc_0_buffer[2*partner_rank],
        MPI_UINT32_T,
        partner_rank,
        0,
        curr_comm,
        status);

        memcpy(endpointer(local_nums, split+1), 
              local_buffer, proc_0_buffer[2*partner_rank]*sizeof(uint32_t));
        num_local_ints = proc_0_buffer[2*world_rank] + proc_0_buffer[2*partner_rank];

        // debug_str = (char*) malloc (64*world_size);
        // list_to_str(local_buffer, proc_0_buffer[2*partner_rank], debug_str);
        // printf("\n  Process %d: gathered proc %d data = %s\n", world_rank, partner_rank, debug_str);
        // list_to_str(local_nums, num_local_ints, debug_str);
        // printf("\n  Process %d new local_ints: [%s]\n", world_rank, debug_str);
        // free(debug_str);

        MPI_Comm_split(
        curr_comm,
        0,
        -1,
        &curr_comm);

      } else {
        // receive upper half data
        // MPI_Recv(
        // void* data,
        // int count,
        // MPI_Datatype datatype,
        // int source,
        // int tag,
        // MPI_Comm communicator,
        // MPI_Status* status)
        partner_rank = world_rank - offset;
        MPI_Recv(
        local_buffer,
        proc_0_buffer[2*partner_rank+1],
        MPI_UINT32_T,
        partner_rank,
        0,
        curr_comm,
        status);

        MPI_Send(
        local_nums,
        proc_0_buffer[2*world_rank],
        MPI_UINT32_T,
        partner_rank,
        0,
        curr_comm);


        // debug_str = (char*) malloc (64*world_size);
        // list_to_str(local_buffer, proc_0_buffer[2*partner_rank+1], debug_str);
        // printf("\n  Process %d: gathered proc %d data = %s\n", world_rank, partner_rank, debug_str);
        
        memcpy(endpointer(local_buffer, 1+proc_0_buffer[2*partner_rank+1]),
              endpointer(local_nums, split+1),
              proc_0_buffer[2*world_rank+1]*sizeof(uint32_t));

        swap(&local_buffer, &local_nums);

        num_local_ints = proc_0_buffer[2*world_rank+1] + proc_0_buffer[2*partner_rank+1];

        
        // list_to_str(local_nums, num_local_ints, debug_str);
        // printf("\n  Process %d new local_ints: [%s]\n", world_rank, debug_str);
        // free(debug_str);

        MPI_Comm_split(
        curr_comm,
        1,
        -1,
        &curr_comm);
      }

      
      MPI_Comm_rank(curr_comm, &world_rank);
      MPI_Comm_size(curr_comm, &world_size);
      
    }

    qsort(local_nums, num_local_ints, sizeof(uint32_t), compare);
  }
  

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  curr_comm = MPI_COMM_WORLD;

  if (world_rank == 0) {
    double end_time = monotonic_seconds();
    print_time(end_time-start_time);
  }

  if (0) {
    char* s = (char*) malloc (num_local_ints*64);
    list_to_str(local_nums, num_local_ints, s);
    printf("\n  FINAL Process %d local_ints: [%s]\n", world_rank, s);
    free(s);
  }

  local_buffer[0] = (uint32_t) num_local_ints;
  // printf("proc %d ints = %d, localbuff = %d\n", world_rank, num_local_ints, local_buffer[0]);
  
  MPI_Gather(
        local_buffer,
        1,
        MPI_UINT32_T,
        proc_0_buffer,
        1,
        MPI_UINT32_T,
        0,
        MPI_COMM_WORLD);
  
  if (world_rank == 0) {
    int calculated_total = 0;
    for (int i = 0; i < world_size; i++){
      calculated_total += proc_0_buffer[i];
    }
    // printf("Total ints = %d, calculated total = %d\n", num_ints, calculated_total);
    // debug_str = (char*) malloc (64*world_size);
    // list_to_str(proc_0_buffer, world_size, debug_str);
    // printf("proc_0_buffer of proc %d: %s\n", world_rank, debug_str);
  
  }

  // Write to file
  for (int i = 0; i < world_size; i++) {
    if (world_rank == i) {
      print_numbers(output_filename, local_nums, num_local_ints, world_rank, num_ints);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  } 


  free(local_buffer);
  free(local_nums);
  free(proc_0_buffer);

  MPI_Finalize();


  return 0;
}