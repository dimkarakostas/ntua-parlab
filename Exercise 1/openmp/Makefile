.PHONY: all

all: jacobi-openmp jacobi-tiled-openmp gaussseidel-openmp redblack-openmp

jacobi-openmp: jacobi-openmp.c utils.c
	gcc -O3 -Wall -D TEST_CONV -D PRINT_RESULTS -fopenmp jacobi-openmp.c utils.c -o jacobi-openmp

jacobi-tiled-openmp: jacobi-tiled-openmp.c utils.c
	gcc -O3 -Wall -D TEST_CONV -D PRINT_RESULTS -fopenmp jacobi-tiled-openmp.c utils.c -o jacobi-tiled-openmp

gaussseidel-openmp: gaussseidel-openmp.c utils.c
	gcc -O3 -Wall -lm -D TEST_CONV -D PRINT_RESULTS -fopenmp gaussseidel-openmp.c utils.c -o gaussseidel-openmp

redblack-openmp: redblack-openmp.c utils.c
	gcc -O3 -Wall -lm -D PRINT_RESULTS -D TEST_CONV -fopenmp redblack-openmp.c utils.c -o redblack-openmp
