#define C 100
#define T 100000000

#define val 1.0
#define e 0.000001

int converge ( double ** u_previous, double ** u_current, int i_min, int i_max, int j_min, int j_max );
double ** allocate2d ( int dimX, int dimY );
void free2d( double ** array, int dimX, int dimY );
void init2d ( double ** array, int dimX, int dimY );
void zero2d ( double ** array, int dimX, int dimY );
void print2d ( double ** array, int dimX, int dimY );
void fprint2d ( char * s, double ** array, int dimX, int dimY );
