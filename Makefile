all: km_cuda

km_cuda: km_cuda.cpp
	nvcc -c km_cuda.cpp
	nvcc -c km_cuda_functions.cu
	nvcc -o km_cuda km_cuda.o km_cuda_functions.o

debug: km_cuda.cpp
	nvcc -g -c km_cuda.cpp
	nvcc -g -c km_cuda_functions.cu
	nvcc -o km_cuda km_cuda.o km_cuda_functions.o

run_debug:
	cuda-gdb -ex "set auto-insert-breakpoints all" --args km_cuda clusters0.txt 3 4 4

clean_output:
	rm output.txt

clean:
	rm output.txt km_cuda km_cuda.o km_cuda_functions.o