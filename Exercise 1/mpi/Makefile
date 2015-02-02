.PHONY: all

all: jacobi-mpi gauss-seidel-mpi red-black-mpi

jacobi-mpi: jacobi-mpi.c utils.c
	mpicc -O3 -Wall -D PRINT_RESULTS -D TEST_CONV jacobi-mpi.c utils.c -o jacobi-mpi

gauss-seidel-mpi: gauss-seidel-mpi.c utils.c
	mpicc -O3 -Wall -D TEST_CONV -D PRINT_RESULTS gauss-seidel-mpi.c utils.c -o gauss-seidel-mpi

red-black-mpi: red-black-mpi.c utils.c
	mpicc -O3 -Wall -D TEST_CONV -D PRINT_RESULTS red-black-mpi.c utils.c -o red-black-mpi
