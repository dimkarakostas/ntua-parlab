#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

int converge(double ** u_previous, double ** u_current, int i_min, int i_max, int j_min, int j_max) {
	int i,j;
	for (i=i_min;i<=i_max;i++)
		for (j=j_min;j<=j_max;j++)
            if (fabs(u_current[i][j]-u_previous[i][j])>e) return 0;
    return 1;
}

double ** allocate2d ( int dimX, int dimY ) {
	double ** array, * tmp;
	int i;
	tmp = ( double * )calloc( dimX * dimY, sizeof( double ) );
	array = ( double ** )calloc( dimX, sizeof( double * ) );	
	for ( i = 0 ; i < dimX ; i++ )
		array[i] = tmp + i * dimY;
	if ( array == NULL || tmp == NULL) {
		fprintf( stderr,"Error in allocation\n" );
		exit( -1 );
	}
	return array;
}

void free2d( double ** array, int dimX, int dimY) {
    if (array==NULL) {
        fprintf(stderr,"Error in freeing matrix\n");
        exit(-1);
    }
    if (array[0])
        free(array[0]);
    if (array)
        free(array);
}

void init2d ( double ** array, int dimX, int dimY ) {
    int i,j;
    for ( i = 0 ; i < dimX ; i++ )
        for ( j = 0; j < dimY ; j++) 
            array[i][j]=(i==0 || i==dimX-1 || j==0 || j==dimY-1)?0.01*(i+1)+0.001*(j+1):0.0;
}

void zero2d ( double ** array, int dimX, int dimY ) {
    int i,j;
    for ( i = 0 ; i < dimX ; i++ )
        for ( j = 0; j < dimY ; j++) 
            array[i][j] = 0.0;
}

void print2d(double ** array, int dimX, int dimY) {
    int i,j;
    for (i=0;i<dimX;i++) {
        for (j=0;j<dimY;j++)
            printf("%lf ",array[i][j]);
        printf("\n");
    }
}

void fprint2d(char * s, double ** array, int dimX, int dimY) {
    int i,j;
    FILE * f=fopen(s,"w");
    for (i=0;i<dimX;i++) {
        for (j=0;j<dimY;j++)
            fprintf(f,"%lf ",array[i][j]);
        fprintf(f,"\n");
    }
    fclose(f);
}
