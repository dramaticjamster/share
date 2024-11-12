all: qs_mpi

km_pthreads: km_pthreads.c
	gcc km_pthreads.c -o km_pthreads -lm

km_openmp: km_openmp.c
	g++ km_openmp.c -o km_openmp -fopenmp

qs_mpi: qs_mpi.c
	mpicc qs_mpi.c -o qs_mpi -fopenmp

clean:
	rm km_openmp km_pthreads clusters.txt medoids.txt qs_mpi
	