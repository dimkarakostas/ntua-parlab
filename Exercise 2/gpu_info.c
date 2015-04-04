/*
 *  gpuinfo.c -- Simple program for querying GPU properties
 *
 *  Copyright (C) 2008, Computing Systems Laboratory (CSLab)
 *  Copyright (C) 2008, Vasileios Karakasis
 */ 

#include	<cuda_runtime_api.h>
#include	<getopt.h>
#include	<inttypes.h>
#include	<stdio.h>
#include	<stdlib.h>
#include	<unistd.h>

#define	EXIT_SUCCESS	0
#define	EXIT_FAILURE	1

#define	NR_OPTIONS	18		/* total number of options */

/* Program short options */
#define	OPT_HELP		'h'
#define	OPT_ALL			'a'
#define	OPT_ID			'i'
#define	OPT_NAME		'n'
#define OPT_MEM_GLOBAL	'm'
#define	OPT_MEM_SHARED	's'
#define OPT_REG			'r'
#define OPT_MAX_THREAD	't'
#define	OPT_WARP		'w'
#define OPT_PITCH		'p'
#define OPT_MEM_CONST	'c'
#define OPT_DIM_BLOCK	'b'
#define	OPT_DIM_GRID	'g'
#define OPT_CLOCK		'z'
#define OPT_VERSION		'v'
#define	OPT_CUDA_VERSION 'V'
#define	OPT_NR_GPUS		'N'
#define	OPT_TEXTURE		'T'

static char	*pname = "gpuinfo";			/* program name */
static char	*version_str = "1.0";		/* program version */

static char	*short_opt = "hai:nmsrwpcbgztvVNT";
static struct option	long_opt[] = {
	{ "all", 0, NULL, OPT_ALL },
	{ "help", 0, NULL, OPT_HELP },
	{ "id", 1, NULL, OPT_ID },
	{ "number-of-gpus", 0, NULL, OPT_NR_GPUS },
	{ "name", 0, NULL, OPT_NAME },
	{ "global-memory", 0, NULL, OPT_MEM_GLOBAL },
	{ "shared-memory", 0, NULL, OPT_MEM_SHARED },
	{ "constant-memory", 0, NULL, OPT_MEM_CONST },
	{ "regs", 0, NULL, OPT_REG },
	{ "max-threads", 0, NULL, OPT_MAX_THREAD },
	{ "warp-size", 0, NULL, OPT_WARP },
	{ "memory-pitch", 0, NULL, OPT_PITCH},
	{ "block-dim", 0, NULL, OPT_DIM_BLOCK },
	{ "grid-dim", 0, NULL, OPT_DIM_GRID },
	{ "cuda-version", 0, NULL, OPT_CUDA_VERSION },
	{ "clock", 0, NULL, OPT_CLOCK },
	{ "texture-alignment", 0, NULL, OPT_TEXTURE },
	{ "version", 0, NULL, OPT_VERSION },
	{ 0, 0, 0, 0 }
};

typedef	uint32_t	optset_t;	/* the options set */

#define BIT_OPT_ALL				0x00001
#define	BIT_OPT_HELP			0x00002
#define	BIT_OPT_ID				0x00004
#define	BIT_OPT_NR_GPUS			0x00008
#define BIT_OPT_NAME			0x00010
#define BIT_OPT_MEM_GLOBAL		0x00020
#define	BIT_OPT_MEM_SHARED		0x00040
#define BIT_OPT_MEM_CONST		0x00080
#define	BIT_OPT_REG				0x00100
#define	BIT_OPT_WARP			0x00200
#define	BIT_OPT_PITCH			0x00400
#define BIT_OPT_DIM_BLOCK		0x00800
#define	BIT_OPT_DIM_GRID		0x01000
#define	BIT_OPT_CUDA_VERSION	0x02000
#define	BIT_OPT_CLOCK			0x04000
#define	BIT_OPT_TEXTURE			0x08000
#define	BIT_OPT_VERSION			0x10000
#define BIT_OPT_MAX_THREAD		0x20000

static __inline__ void
optset_set_option(optset_t *options, optset_t mask)
{
	*options |= mask;
	return;
}

static __inline__ void
optset_clear_option(optset_t *options, optset_t mask)
{
	*options &= ~mask;
	return;
}

static __inline__ void
optset_clear(optset_t *options)
{
	*options = 0;
	return;
}

static __inline__ int
optset_isset_only(optset_t *options, optset_t mask)
{
	return (*options == mask);
}

static __inline__ int
optset_isset(optset_t *options, optset_t mask)
{
	return ((*options & mask) == mask);
}

static __inline__ int
optset_isempty(optset_t *options)
{
	return (*options == 0);
}

static void
print_usage()
{
	fprintf(stderr, "Usage: %s [-i gpu_id] [-hanNmsrwpcbgztTvV]\n", pname);
	fprintf(stderr, "Try `%s --help' for more information\n", pname);
	return;
}

static void
print_help_msg()
{
	fprintf(stderr, "Usage: %s [-i gpu_id] [-hanNmsrwpcbgztTvV]\n", pname);
	fprintf(stderr, "Display information about gpu devices.\n");
	fprintf(stderr, "  -i [id] display information only for gpu `id'.\n");
	fprintf(stderr, "  -h      display this help message and exit.\n");
	fprintf(stderr, "  -a      display all available information for every "
			"gpu, unless the `-i' option is specified.\n");
	fprintf(stderr, "  -n      display the gpu name.\n");
	fprintf(stderr, "  -N      display the total number of gpus available "
			"and exit.\n");
	fprintf(stderr, "  -m      display the global memory size in bytes.\n");
	fprintf(stderr, "  -s      display the shared memory size per block in "
			"bytes.\n");
	fprintf(stderr, "  -r      display the number of available registers "
			"per multiprocessor.\n");
	fprintf(stderr, "  -w      display the warp size.\n");
	fprintf(stderr, "  -p      display the maximum memory pitch in bytes "
			"allowed for memory copies.\n");
	fprintf(stderr, "  -c      display the constant memory size in bytes.\n");
	fprintf(stderr, "  -b      display the maximum dimensions for a block.\n");
	fprintf(stderr, "  -g      display the maximum dimensions for a grid.\n");
	fprintf(stderr, "  -z      display the core clock rate in KHz.\n");
	fprintf(stderr, "  -t      display the maximum number of threads per "
			"block.\n");
	fprintf(stderr, "  -T      display the required texture alignment.\n");
	fprintf(stderr, "  -v      display version information and exit.\n");
	fprintf(stderr, "  -V      display the CUDA version supported.\n");
	return;
}

static __inline__ float
convert_to_mb(size_t nr_bytes)
{
	return (nr_bytes / (float) 1048576);
}

static __inline__ float
convert_to_kb(size_t nr_bytes)
{
	return (nr_bytes / (float) 1024);
}

static int
print_all_prop(int gpu_id)
{
	struct cudaDeviceProp	dev_prop;	/* gpu device properties */

	if (cudaGetDeviceProperties(&dev_prop, gpu_id) != cudaSuccess) {
		fprintf(stderr,
				"%s: cannot retrieve properties for gpu %d\n",
				pname, gpu_id);
		return -1;
	}
		
	printf("gpu                   : %d\n", gpu_id);
	printf("name                  : %s\n", dev_prop.name);

	printf("global memory         : %ld bytes (%.1f MB)\n",
		   dev_prop.totalGlobalMem, convert_to_mb(dev_prop.totalGlobalMem));

	printf("shared memory / block : %ld bytes (%.1f KB)\n",
		   dev_prop.sharedMemPerBlock,
		   convert_to_kb(dev_prop.sharedMemPerBlock));

	printf("regs / block          : %d\n", dev_prop.regsPerBlock);

	printf("warp size             : %d\n", dev_prop.warpSize);

	printf("max memory pitch      : %ld (bytes) (%.1f KB)\n",
		   dev_prop.memPitch, convert_to_kb(dev_prop.memPitch));

	printf("max threads / block   : %d\n", dev_prop.maxThreadsPerBlock);

	printf("max block dim         : (%d, %d, %d)\n",
		   dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1],
		   dev_prop.maxThreadsDim[2]);

	printf("max grid dim          : (%d, %d, %d)\n",
		   dev_prop.maxGridSize[0], dev_prop.maxGridSize[1],
		   dev_prop.maxGridSize[2]);

	printf("constant memory       : %ld bytes (%.1f KB)\n",
		   dev_prop.totalConstMem, convert_to_kb(dev_prop.totalConstMem));

	printf("compute capability    : %d.%d\n", dev_prop.major, dev_prop.minor);

	printf("clock rate            : %.2f MHz\n", dev_prop.clockRate	/ 1000.0);

	printf("texture alignment     : %ld bytes\n", dev_prop.textureAlignment);

	return 0;
}

int
main(int argc, char *argv[])
{
	int			nr_dev;
	int			gpu_id = -1;			/* no device explicitly specified */
	int			opt_sym;
	int			prop_seq[NR_OPTIONS];	/* sequence of property options */
	int			nr_prop = 0;			/* property options specified */
	optset_t	turned_on_options;
	struct cudaDeviceProp	dev_prop;	/* gpu device properties */
	
	optset_clear(&turned_on_options);
	while ( (opt_sym = getopt_long(argc, argv,
								   short_opt, long_opt, NULL)) >= 0) {
		switch (opt_sym) {
		case OPT_HELP:
			optset_set_option(&turned_on_options, BIT_OPT_HELP);
			break;
		case OPT_ALL:
			optset_set_option(&turned_on_options, BIT_OPT_ALL);
			break;
		case OPT_ID:
			optset_set_option(&turned_on_options, BIT_OPT_ID);
			gpu_id = atoi(optarg);	/* FIXME: it will set gpu_id to zero, even
									   when an invalid argument is given. */
			break;
		case OPT_NR_GPUS:
			optset_set_option(&turned_on_options, BIT_OPT_NR_GPUS);
			break;
		case OPT_NAME:
			if ( !optset_isset(&turned_on_options, BIT_OPT_NAME) ) {
				optset_set_option(&turned_on_options, BIT_OPT_NAME);
				prop_seq[nr_prop++] = OPT_NAME;
			}
			break;
		case OPT_MEM_GLOBAL:
			if ( !optset_isset(&turned_on_options, BIT_OPT_MEM_GLOBAL) ) {
				optset_set_option(&turned_on_options, BIT_OPT_MEM_GLOBAL);
				prop_seq[nr_prop++] = OPT_MEM_GLOBAL;
			}
			break;
		case OPT_MEM_SHARED:
			if ( !optset_isset(&turned_on_options, BIT_OPT_MEM_SHARED) ) {
				optset_set_option(&turned_on_options, BIT_OPT_MEM_SHARED);
				prop_seq[nr_prop++] = OPT_MEM_SHARED;
			}
			break;
		case OPT_REG:
			if ( !optset_isset(&turned_on_options, BIT_OPT_REG) ) {
				optset_set_option(&turned_on_options, BIT_OPT_REG);
				prop_seq[nr_prop++] = OPT_REG;
			}
			break;
		case OPT_MAX_THREAD:
			if ( !optset_isset(&turned_on_options, BIT_OPT_MAX_THREAD) ) {
				optset_set_option(&turned_on_options, BIT_OPT_MAX_THREAD);
				prop_seq[nr_prop++] = OPT_MAX_THREAD;
			}
			break;
		case OPT_WARP:
			if ( !optset_isset(&turned_on_options, BIT_OPT_WARP) ) {
				optset_set_option(&turned_on_options, BIT_OPT_WARP);
				prop_seq[nr_prop++] = OPT_WARP;
			}
			break;
		case OPT_PITCH:
			if ( !optset_isset(&turned_on_options, BIT_OPT_PITCH) ) {
				optset_set_option(&turned_on_options, BIT_OPT_PITCH);
				prop_seq[nr_prop++] = OPT_PITCH;
			}
			optset_set_option(&turned_on_options, BIT_OPT_PITCH);
			break;
		case OPT_MEM_CONST:
			if ( !optset_isset(&turned_on_options, BIT_OPT_MEM_CONST) ) {
				optset_set_option(&turned_on_options, BIT_OPT_MEM_CONST);
				prop_seq[nr_prop++] = OPT_MEM_CONST;
			}
			break;
		case OPT_DIM_BLOCK:
			if ( !optset_isset(&turned_on_options, BIT_OPT_DIM_BLOCK) ) {
				optset_set_option(&turned_on_options, BIT_OPT_DIM_BLOCK);
				prop_seq[nr_prop++] = OPT_DIM_BLOCK;
			}
			break;
		case OPT_DIM_GRID:
			if ( !optset_isset(&turned_on_options, BIT_OPT_DIM_GRID) ) {
				optset_set_option(&turned_on_options, BIT_OPT_DIM_GRID);
				prop_seq[nr_prop++] = OPT_DIM_GRID;
			}
			break;
		case OPT_CUDA_VERSION:
			if ( !optset_isset(&turned_on_options, BIT_OPT_CUDA_VERSION) ) {
				optset_set_option(&turned_on_options, BIT_OPT_CUDA_VERSION);
				prop_seq[nr_prop++] = OPT_CUDA_VERSION;
			}
			break;
		case OPT_CLOCK:
			if ( !optset_isset(&turned_on_options, BIT_OPT_CLOCK) ) {
				optset_set_option(&turned_on_options, BIT_OPT_CLOCK);
				prop_seq[nr_prop++] = OPT_CLOCK;
			}
			break;
		case OPT_TEXTURE:
			if ( !optset_isset(&turned_on_options, BIT_OPT_TEXTURE) ) {
				optset_set_option(&turned_on_options, BIT_OPT_TEXTURE);
				prop_seq[nr_prop++] = OPT_TEXTURE;
			}
			break;
		case OPT_VERSION:
			optset_set_option(&turned_on_options, BIT_OPT_VERSION);
			break;
		case '?':
			print_usage();
			exit(EXIT_FAILURE);
		}
	}

	if (optset_isset(&turned_on_options, BIT_OPT_HELP)) {
		/* print a short help message and exit */
		print_help_msg();
		exit(EXIT_SUCCESS);
	}

	if (optset_isset(&turned_on_options, BIT_OPT_VERSION)) {
		/* print program version and exit */
		printf("gpuinfo %s\n", version_str);
		exit(EXIT_SUCCESS);
	}

	if (cudaGetDeviceCount(&nr_dev) != cudaSuccess) {
		fprintf(stderr, "%s: no gpus found\n", pname);
		exit(EXIT_FAILURE);
	}

	if (optset_isset(&turned_on_options, BIT_OPT_ALL)		||
		optset_isset_only(&turned_on_options, BIT_OPT_ID)	||
		optset_isempty(&turned_on_options)) {
		/* print all info and exit */
		if (gpu_id == -1) {
			/* print info for every gpu found */
			printf("Number of gpus: %d\n", nr_dev);
			for (gpu_id = 0; gpu_id < nr_dev; gpu_id++)
				print_all_prop(gpu_id);
		} else
			print_all_prop(gpu_id);

		exit(EXIT_SUCCESS);
	}

	if (optset_isset(&turned_on_options, BIT_OPT_NR_GPUS)) {
		/* print number of gpus and exit */
		printf("%d\n", nr_dev);
		exit(EXIT_SUCCESS);
	}

	if (gpu_id == -1)
		/* no gpu specified; assume gpu 0 */
		gpu_id = 0;

	/* get the properties of the gpu */
	if (cudaGetDeviceProperties(&dev_prop, gpu_id) != cudaSuccess) {
		fprintf(stderr, "%s: cannot retrieve properties for gpu %d\n",
				pname, gpu_id);
		exit(EXIT_FAILURE);
	}

	/* print properties in the order specified from command line */
	int	i;
	for (i = 0; i < nr_prop; i++) {
		switch (prop_seq[i]) {
		case OPT_NAME:
			printf("%s\n", dev_prop.name);
			break;
		case OPT_MEM_GLOBAL:
			printf("%ld\n", dev_prop.totalGlobalMem);
			break;
		case OPT_MEM_SHARED:
			printf("%ld\n", dev_prop.sharedMemPerBlock);
			break;
		case OPT_REG:
			printf("%d\n", dev_prop.regsPerBlock);
			break;
		case OPT_MAX_THREAD:
			printf("%d\n", dev_prop.maxThreadsPerBlock);
			break;
		case OPT_WARP:
			printf("%d\n", dev_prop.warpSize);
			break;
		case OPT_PITCH:
			printf("%ld\n", dev_prop.memPitch);
			break;
		case OPT_MEM_CONST:
			printf("%ld\n", dev_prop.totalConstMem);
			break;
		case OPT_DIM_BLOCK:
			printf("%d %d %d\n", dev_prop.maxThreadsDim[0],
				   dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
			break;
		case OPT_DIM_GRID:
			printf("%d %d %d\n", dev_prop.maxGridSize[0],
				   dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
			break;
		case OPT_CUDA_VERSION:
			printf("%d.%d\n", dev_prop.major, dev_prop.minor);
			break;
		case OPT_CLOCK:
			printf("%d\n", dev_prop.clockRate);
			break;
		case OPT_TEXTURE:
			printf("%ld\n", dev_prop.textureAlignment);
			break;
		}
	}

	return 0;
}
