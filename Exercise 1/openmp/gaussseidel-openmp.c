#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "utils.h"
#include <omp.h>


void GaussSeidel(double ** u_previous, double ** u_current, int X_min, int X_max, int Y_min, int Y_max, double omega) {
	int i,j,k=0;

    #pragma omp parallel default(none) shared(X_min, X_max, Y_min, Y_max, u_current, u_previous, omega) private(i, j, k)
	{
	for (k=X_min;k<X_max+1;k++){
      #pragma omp for
		for (j=Y_min;j<=k-1;j++){
			i = k - j;
			u_current[i][j]=u_previous[i][j]+(u_current[i-1][j]+u_previous[i+1][j]+u_current[i][j-1]+u_previous[i][j+1]-4*u_previous[i][j])*omega/4.0;

		}
	}
	}
    #pragma omp parallel default(none) shared(X_min, X_max, Y_min, Y_max, u_current, u_previous, omega) private(i, j, k)
	{
	for (k=X_max-1;k>=X_min;k--) {
      #pragma omp for
		for (j=Y_min;j<=k-1;j++) {
			i = k - j;
			u_current[X_max-j][Y_max-i]=u_previous[X_max-j][Y_max-i]+(u_current[X_max-j-1][Y_max-i]+u_previous[X_max-j+1][Y_max-i]+u_current[X_max-j][Y_max-i-1]+u_previous[X_max-j][Y_max-i+1]-4*u_previous[X_max-j][Y_max-i])*omega/4.0;
		}
	}
	}
}


int main ( int argc, char ** argv ) {

	int X, Y;                           //2D-domain dimensions
	double ** u_current, ** u_previous;   //2D-domain
	double ** swap;
	struct timeval tts,ttf;
	double time=0;
	int t,converged=0;
	double omega;						//relaxation factor
	//read 2D-domain dimensions
	if (argc<2) {
		fprintf(stderr,"Usage: ./exec X (Y-optional)");
		exit(-1);
	}
	else if (argc==2) {
		X=atoi(argv[1]);
		Y=X;
	}
	else {
		X=atoi(argv[1]);
		Y=atoi(argv[2]);
	}

	//allocate domain and initialize boundary

	u_current=allocate2d(X,Y);
	u_previous=allocate2d(X,Y);
	init2d(u_current,X,Y);
	init2d(u_previous,X,Y);

	omega=2.0/(1+sin(3.14/X));


	//computational core
	#ifdef TEST_CONV
	for (t=0;t<T && !converged;t++) {
	#endif
	#ifndef TEST_CONV
	#undef T
	#define T 256
	for (t=0;t<T;t++) {
	#endif
		swap=u_previous;
		u_previous=u_current;
		u_current=swap;
		gettimeofday(&tts,NULL);

		GaussSeidel(u_previous,u_current,1,X-1,1,Y-1, omega);

		gettimeofday(&ttf,NULL);
		time+=(ttf.tv_sec-tts.tv_sec)+(ttf.tv_usec-tts.tv_usec)*0.000001;

		#ifdef TEST_CONV
		if (t%C==0)
			converged=converge(u_previous,u_current,X,Y);
		#endif
	}

	printf("GaussSeidelSOR X %d Y %d Iter %d Time %lf midpoint %lf\n",X,Y,t-1,time,u_current[X/2][Y/2]);

	#ifdef PRINT_RESULTS
	char * s=malloc(30*sizeof(char));
	sprintf(s,"resGaussSeidelSORNaive_%dx%d",X,Y);
	fprint2d(s,u_current,X,Y);
	free(s);
	#endif

	return 0;
}

