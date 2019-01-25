#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include <cuda.h>
#include <math.h>

/* Error Check Procedures */
inline cudaError_t checkCuda(cudaError_t result);

/* Element: 3 Dimensions Element in [0,1)^3 */
struct Element{
	float x;
	float y;
	float z;
};

/* Functions Declaration */
/* GPU Functions */
__global__ void nearestNeighbor( struct Element *devQ, struct Element *devC, int *devIndexArrayQ, int *devIndexArrayC, int *devNNindeces, float *devNNdistances,  int N,  int d);
__device__ float calculateDistance( struct Element a, struct Element b);

/* Main Procedures */
void binning( struct Element *elementsArray, int *a,  int N,  int d, int *blocksSize, int *oldIndex);
void initIndexArray( int *a, int *blocksSize,  int d);

/* Sub-Procedures */
void checkGpuMemory(  int N,  int d);
void printDeviceInfo(cudaDeviceProp *prop);
void printDeviceInfo(cudaDeviceProp *prop);
void print( struct Element *elementsArray,  int N);

/* File Operations Procedures */
void readElementsFromFile(struct Element *elements,  int N,  const char *s);
void readIntegersFromFile( int *a,  int N,  const char *s);
void readFloatsFromFile( float *a,  int N,  const char *s);

/* Validation Functions Declaration */
void validateBinning( struct Element *elemntsArray, int *a,  int N,  int d);
int validateResultsFromFile( struct Element *Q, struct Element *C, float *distances, int *NNindeces,  int N, int *oldIndexQ, int *oldIndexC);

/*********************/
/*** Main Function ***/
/*********************/

int main(int argc, char  *argv[])
{

  printf("Parallel Nearest Neighbor\n");
  /* Variables to hold the entire execetucion time */
	struct timeval startOfProgramTime, endOfProgramTime;
	float wholeTime;
	gettimeofday (&startOfProgramTime, NULL);

  int i;
	int d_power = atoi(argv[1]);
 	int N_power = 21;							             // N^N_power
  int l_power = N_power - d_power*3;	// Elements in each block are 2^l_power
  int N = pow( 2, N_power);
	int d = pow( 2, d_power); printf("N is 2^%d= %d and d is 2^%d\n", N_power, N, d_power);

  /* Print Device information*/
	cudaDeviceProp prop;
	printDeviceInfo(&prop);

	/* Check current State of GPU Memory */
	printf("\nChecking current state of GPU Memory:\n");
	checkGpuMemory( N, d); printf("\n");

  /*--- Host: Variables Initialization, Data Pre-Processing */
  /* Generate Random Elements [0,1) Uniformly Distributed */
  struct Element *hostQ, *hostC;
  hostQ = ( struct Element *)malloc( N*sizeof(struct Element));
  hostC = ( struct Element *)malloc( N*sizeof(struct Element));
	printf("Reading Elements for Q Set\n"); readElementsFromFile( hostQ, N,  "./test/MatlabQ21.txt");
	printf("Reading Elements for C Set\n"); readElementsFromFile( hostC, N,  "./test/MatlabC21.txt");
  //printf("Elements of Q Set:\n"); print( hostQ, N);
  //printf("Elements of C Set:\n"); print( hostC, N);

  /* Match Elements-Boxes */
	int *oldIndexQ = (int *)malloc( N*sizeof(int));
	int *oldIndexC = (int *)malloc( N*sizeof(int));
	int *hostBoxIndexC = (int *)malloc( N*sizeof(int)); //Array to store the index of the box where Element is matched
	int *hostBoxIndexQ = (int *)malloc( N*sizeof(int));
	int *blocksSizeC = (int *)malloc( d*d*d*sizeof(int)); 		//Array to Store the Number of Elements for each block
	int *blocksSizeQ = (int *)malloc( d*d*d*sizeof(int)); 		//Array to Store the Number of Elements for each block
	for( i = 0; i < d*d*d; i++)
	{	blocksSizeQ[i] = 0;	blocksSizeC[i] = 0;}

  printf("Binning Q Array:"); binning( hostQ, hostBoxIndexQ, N, d, blocksSizeQ, oldIndexQ);
	printf("Binning C Array:"); binning( hostC, hostBoxIndexC, N, d, blocksSizeC, oldIndexC);
	printf("Q Array:"); validateBinning( hostQ, hostBoxIndexQ, N, d);
	printf("C Array:"); validateBinning( hostC, hostBoxIndexC, N, d);
	//printf("Elements of Q Set:\n"); print( hostQ, N);
	//printf("Elements of C Set:\n"); print( hostC, N);

  /* Table Of starting Indeces for each block */
  int *indexArrayC;		//Elements of this array contain the starting index of Element's table for every block for Array C
  int *indexArrayQ;		//Elements of this array contain the starting index of Element's table for every block for Array Q
  indexArrayC = (int *)malloc( d*d*d*sizeof(int));
  indexArrayQ = (int *)malloc( d*d*d*sizeof(int));
  initIndexArray( indexArrayC, blocksSizeC, d); printf("Index Array C has been completed\n");
  initIndexArray( indexArrayQ, blocksSizeQ, d); printf("Index Array Q has been completed\n");

  /* Allocate Space to hold distnces and indeces of Nearest Neighbors */
  int *NNindeces = (int *)malloc(N*sizeof(int));
	float *NNdistances = (float *)malloc(N*sizeof(float));
  for( i = 0; i < N; i++)
		NNdistances[i] = (float)2;

  /*--- Data Initialization and Pre-Processing has been Completed ---*/

  /* Allocate Space in GPU */
  struct Element *devQ, *devC;
  float *devNNdistances;
  int *devNNindeces, *devIndexArrayC, *devIndexArrayQ;
	//int *devBitSearchArray;
	checkCuda( cudaMalloc( &devQ, N * sizeof(struct Element)));
  checkCuda( cudaMalloc( &devC, N * sizeof(struct Element)));
  checkCuda( cudaMalloc( &devNNdistances, N * sizeof(float)));
  checkCuda( cudaMalloc( &devNNindeces, N * sizeof(int)));
  checkCuda( cudaMalloc( &devIndexArrayQ, N* sizeof(int))); ;
  checkCuda( cudaMalloc( &devIndexArrayC, N * sizeof(int)));
	//checkCuda( cudaMalloc( &devBitSearchArray, (N*27/32)*sizeof(int)));
  printf("Space Allocated in GPU\n");
  /* Copy Arrays from HOST to DEVICE */
  checkCuda( cudaMemcpy( devQ, hostQ, N * sizeof( struct Element), cudaMemcpyHostToDevice));
  checkCuda( cudaMemcpy( devC, hostC, N * sizeof( struct Element), cudaMemcpyHostToDevice));
  checkCuda( cudaMemcpy( devIndexArrayQ, indexArrayQ, N * sizeof( int), cudaMemcpyHostToDevice));
  checkCuda( cudaMemcpy( devIndexArrayC, indexArrayC, N * sizeof( int), cudaMemcpyHostToDevice));
  checkCuda( cudaMemcpy( devNNdistances, NNdistances, N * sizeof( float), cudaMemcpyHostToDevice));
  checkCuda( cudaMemset( devNNindeces, 0, N*sizeof( int))); //Set All Elements to 0
	//checkCuda( cudaMemset( devBitSearchArray, 0, (N*27/32)*sizeof( int))); //Set All Elements to 0

  /* Initialize Dimensions */
	int dimX, dimY, dimZ;
	if( (1 << l_power) > prop.maxThreadsPerBlock){	//Case: number in each Block are more than the maximum number of threads per block.
		l_power = log(prop.maxThreadsPerBlock)/log(2); printf("Number of Elements in each Block are more than Maximum Number of Threads that can be allocated\n");
	}
	dimX = 1<<(l_power/3 + l_power%3); dimY = 1 << l_power/3; dimZ = 1 << l_power/3;
	if( l_power/3 == 0){	//Case: l_power < 3, then [x*1*1] threads per block.
		dimY = 1;	dimZ = 1;
	}
	dim3 numberOfBlocks( d, d, d); printf("Grid Size : [%d x %d x %d]\n", d, d, d);
	dim3 threadsPerBlock( dimX, dimY, dimZ); printf("Threads per Block: [%d %d %d]\n", dimX, dimY, dimZ);

  /* 1st Primary Nearest Neighbor */
  /* Variables to hold execution time */
	struct timeval startwtime, endwtime;
	float executionTime;
	gettimeofday (&startwtime, NULL);

  nearestNeighbor<<< numberOfBlocks, threadsPerBlock>>>( devQ, devC, devIndexArrayQ, devIndexArrayC, devNNindeces, devNNdistances, N, d);
	checkCuda(cudaDeviceSynchronize());
  gettimeofday (&endwtime, NULL);
	executionTime = (float)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
	printf("Nearest Neighbor wall clock time: %f sec\n\n\n", executionTime);

  /* End of Program	*/
	gettimeofday (&endOfProgramTime, NULL);
	wholeTime = (float)((endOfProgramTime.tv_usec - startOfProgramTime.tv_usec)/1.0e6 + endOfProgramTime.tv_sec - startOfProgramTime.tv_sec);
	/* Copy results to HOST */
  checkCuda( cudaMemcpy(  NNindeces, devNNindeces, N * sizeof( int), cudaMemcpyDeviceToHost));
  checkCuda( cudaMemcpy(  NNdistances, devNNdistances, N * sizeof( float), cudaMemcpyDeviceToHost));

  int flag = validateResultsFromFile( hostQ, hostC, NNdistances, NNindeces, N, oldIndexQ, oldIndexC);
  if(flag == 1){
    printf("Nearest Neighbor Validation: PASSed\n");
    printf("Nearest Neighbor wall clock time: %f sec\n\n\n", executionTime);
  }
  else{
    printf("Nearest Neighbor Validation: FAILed\n");
    printf("Failed Nearest Neighbor wall clock time: %f sec\n\n\n", executionTime);
  }

  /* End of Program	*/
	gettimeofday (&endOfProgramTime, NULL);
	wholeTime = (float)((endOfProgramTime.tv_usec - startOfProgramTime.tv_usec)/1.0e6 + endOfProgramTime.tv_sec - startOfProgramTime.tv_sec);
	printf("Entire program wall clock time: %f sec\n\n", wholeTime);

  /* Cleanup Device */
  cudaFree( devQ);
  cudaFree( devC);
  cudaFree( devIndexArrayQ);
  cudaFree( devIndexArrayC);
  cudaFree( devNNindeces);
  cudaFree( devNNdistances);

  /* Cleanup Host*/
  free(hostC);
  free(hostQ);
  free(oldIndexC);
  free(oldIndexQ);
  free(hostBoxIndexQ);
  free(hostBoxIndexC);
  free(blocksSizeC);
  free(blocksSizeQ);
  free(indexArrayQ);
  free(indexArrayC);
  free(NNdistances);
  free(NNindeces);

  return 0;
}

/**********************/
/*** GPU Procedures ***/
/**********************/

/* primaryNearestNeighbor(): For All Elements included in a box of Q
*  searches the respective block in C for the nearestNeighbor.*/
	//extern __shared__ int sharedMemory[];
__global__ void nearestNeighbor( struct Element *devQ, struct Element *devC, int *devIndexArrayQ, int *devIndexArrayC, int *devNNindeces, float *devNNdistances,  int N,  int d){
  int i, k, j, l;
  int limitQ, limitC, startQ, startC, index, nearestNeighborIndex, blocksSizeC;
  float minDistance, distance;
  /* Calculate 1D index of a thread relative to its block */
	int threadId_block = threadIdx.x + (threadIdx.y)*blockDim.x + (threadIdx.z)*(blockDim.x)*(blockDim.y);
	/* Calculate 1D index of a block relative to the grid */
	int blockId_grid = blockIdx.x + (blockIdx.y)*(gridDim.x) + (blockIdx.z)*(gridDim.x)*(gridDim.y);
	/* Calculate 1D index o a thread relative to the grid */
	//int threadId_grid = blockId_grid*(blockDim.x*blockDim.y*blockDim.z) + threadId_block;
  /* Calculate Block Stride */
  int blockStride = (blockDim.x)*(blockDim.y)*(blockDim.z); //In case Elements in Query Array are more than Threads
 	//printf("blockStride %d\n", blockStride);
  /* Each Block corresponds to the block of the previous pre-processing Clustering */
  /* Start point of a block is known in both arrays Q and C */
  /* Thread Id corresponding to the block represents an element of threadId_block position + start_position of the block in Array C and Q */
	/* Define Search Limits of Block*/


  if( blockId_grid == d*d*d - 1){//Check for Out of Bounds limit
		limitQ = N;
		limitC = N;
	}
	else{	//Stop when you reach 1st Element of next block
		limitQ = devIndexArrayQ[blockId_grid + 1];
		limitC = devIndexArrayC[blockId_grid + 1];
	}

  /* Define Start Index of Block*/
  startQ = devIndexArrayQ[blockId_grid];
  startC = devIndexArrayC[blockId_grid];

  /* Define corresponding index in the Array Q */
  index = startQ + threadId_block;

	blocksSizeC = limitC - startC;
	struct Element Query;
	/* move C to share memory */
	extern __shared__ int sharedMemory[];
	int count = 0;
	struct Element *s = ( struct Element*)(&(sharedMemory[0]));
	j = 0;
	while( startC + j*count < limitC){
		for( l = 0; l < 5120 || l < limitC; l++){ //Do not exceed Shared Memory
			s[l] = devC[startC + l + 5120*count];
		}
		/* Synchronize Threads to read simultaneously from shared memory */
		__syncthreads();


	  /* Primary Nearest Neighbor Set */
	  for( i = index; i < limitQ; i = i + blockStride){
			minDistance = 2;
			Query = devQ[i];
			for( k = 0 + 5120*count; k < 5120 || k < blocksSizeC; k++){
				distance = sqrt( pow(Query.x - s[k].x, 2) + pow(Query.y - s[k].y, 2) + pow(Query.z - s[k].z, 2));
				if( distance < minDistance){
					minDistance = distance;
	        nearestNeighborIndex = k;
	      }
	    }
	    /* Store the results */
	    devNNindeces[i] = nearestNeighborIndex;
	    devNNdistances[i] = minDistance;
		}
		count++;
  }

  /* Secondary Nearest Neighbor Set */
  int x, y, z, searchBoxId;
	float distanceZ, distanceX, distanceY, interval, blockDistance;

  /* Define Interval between boxes */
	interval = 1/((float)d);
	/* Reinitialize index */
  index = startQ + threadId_block;
	/* Choose Which Box to Search */
	int startZ, startX, startY, endZ, endY, endX;
	startZ = blockIdx.z - 1;
	endZ = blockIdx.z + 1;
	startX = blockIdx.x - 1;
	endX = blockIdx.x + 1;
	startY = blockIdx.y - 1;
	endY = blockIdx.y + 1;

	/* Define z */
  for( z = startZ; z <= endZ; z++ ){
		if( z >= 0 && z < d){
			/* Define y */
			for( y = startY; y <= endY; y++ ){
			  if( y >= 0 && y < d){
			    /* Define x*/
			    for( x = startX; x <= endX; x++){
		      	if( x >= 0 && x < d){
		        	/* Block to be searched has been defined */
			        searchBoxId = x + d*y +d*d*z;
			        /* Choose to search or not */
			        if(searchBoxId != blockId_grid){
								/* Set Limit of the Box */
								if( searchBoxId != d*d*d-1)
									limitC = devIndexArrayC[searchBoxId+1];
								else
									limitC = N;
								/* Set start Index of Box to be searched */
								startC = devIndexArrayC[searchBoxId];
								/* Size of searchBoxId */
								blocksSizeC = limitC - startC;
								/* move C to share memory */
								count = 0;
								j = 0;
								while( startC + j*count < limitC){
									for( l = 0; l < 5120 || l < limitC; l++){ //Do not exceed Shared Memory
										s[l] = devC[startC + l + 5120*count];
									}
									/* Synchronize Threads to read simultaneously from shared memory */
									__syncthreads();
				          for( i = index; i < limitQ; i += blockStride){
				            minDistance = devNNdistances[i];
				            nearestNeighborIndex = devNNindeces[i];
										Query = devQ[i];
				            /* Calculate shortest distance Of Query from Box on z axis */
				            if( z == startZ)
				              distanceZ = abs( Query.z - (z + 1)*interval);
				            else if( z == blockIdx.z)
				              distanceZ = 0;
				            else if( z == endZ)
				              distanceZ = abs( z*interval - Query.z);
				            else
				              printf("Z error: z = %d\n", z);
				            /* Calculate shortest distance Of Query from Box on y axis */
				            if( y == startY)
				              distanceY = abs( Query.y - (y + 1)*interval);
				            else if( y == blockIdx.y)
				              distanceY = 0;
				            else if( y == endY)
				              distanceY = abs( y*interval - Query.y);
				            else
				              printf("Y error: y = %d\n", y);
				            /* Calculate shortest distance Of Query from Box on y axis */
				            if( x == startX)
				              distanceX = abs( Query.x - (x + 1)*interval);
				            else if( x == blockIdx.x)
				              distanceX = 0;
				            else if( x == endX)
				              distanceX = abs( x*interval - Query.x);
				            else
				              printf("X error: x = %d\n", x);
				            /* Calculate Distance from Block */
				            blockDistance = sqrt( pow(distanceX, 2) + pow(distanceY, 2) + pow(distanceZ, 2));

				            if( blockDistance < minDistance){

				              /* Search All his Elements */
											for( k = 0 + 5120*count; k < 5120 || k < blocksSizeC; k++){
												distance = sqrt( pow(Query.x - s[k].x, 2) + pow(Query.y - s[k].y, 2) + pow(Query.z - s[k].z, 2));
												if( distance < minDistance){
													minDistance = distance;
									        nearestNeighborIndex = k;
									      }
				              }
				              devNNdistances[i] = minDistance;
				              devNNindeces[i] = nearestNeighborIndex;
				            }
				          }
									count++;
								}
			        }
			      }
			    }
			  }
			}
		}
  }
}

/* calculateDistance(): calculates euclidean distance between two elements */
__device__
float calculateDistance( struct Element a, struct Element b)
{
	float temp;
	temp = sqrt( pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
	return temp;
}


/***********************/
/*** Main Procedures ***/
/***********************/

/* binning(): Data Binning/Bucketing The original data values
*	 which fall in a given small interval, a bin, are assigned
*	 a value representative of that interval.
*	 This function rearranges the whole array based on  the bins.
*/
void binning( struct Element *elementsArray, int *a,  int N,  int d, int *blocksSize, int *oldIndex)
{
	/* Grid size is [d x d x d]
	x = [0,1), y = [0,1), z = [0,1)
	Therefore every direction's distance is divided by d.
	k intervals/bins are produced by division 1/d
	This function matches an Element to its id in the grid of blocks
	*/
	int i, k, dimX, dimY, dimZ;

	float interval = 1/((float)d); printf(" Interval is %f\n", interval);
	int blockIdInGrid;
	/* For Every Element find the coordinates of the block it belongs */
	for( k = 0; k < N; k++){
		/* find x block position */
		for( i = 0; i < d; i++){
			if( elementsArray[k].x >= i*interval && elementsArray[k].x < (i+1)*interval)
				dimX = i;
		}
		/* find y block position */
		for( i = 0; i < d; i++){
			if( elementsArray[k].y >= i*interval && elementsArray[k].y < (i+1)*interval)
				dimY = i;
		}
		/* find z block position */
		for( i = 0; i < d; i++){
			if( elementsArray[k].z >= i*interval && elementsArray[k].z < (i+1)*interval)
				dimZ = i;
		}

		/* block's id inside grid is idx + numXs*idy + numXs*numYs*idz */
		blockIdInGrid = dimX + d*dimY + d*d*dimZ;
		a[k] = blockIdInGrid;
		blocksSize[blockIdInGrid] = blocksSize[blockIdInGrid] + 1;
	}

	/* Since each Element's box is known, they will be clustered */
	struct Element *tempElementsArray = (struct Element *)malloc( N*sizeof(struct Element));	// temporary Array needed for swapping

	int *tempBoxIndexes = (int *)malloc( d*d*d*sizeof(int));	// temporary array containing start index of each box
	initIndexArray( tempBoxIndexes, blocksSize, d);

	/* Binning Rearrangement of Elements */
	for( i = 0; i < N; i++){
		tempElementsArray[tempBoxIndexes[a[i]]] = elementsArray[i];
		oldIndex[tempBoxIndexes[a[i]]] = i;
		tempBoxIndexes[a[i]] = tempBoxIndexes[a[i]] + 1;
	}

	/* Update hostBoxIndex Array */
	int tempIndex = 0;
	for( i = 0; i < d*d*d; i++){
		for( k = 0; k < blocksSize[i]; k++){
			a[tempIndex] = i;
			tempIndex++;
		}
	}

	/* Update elemntsArrayArray */
	for( i = 0; i < N; i++){
		elementsArray[i] = tempElementsArray[i];
	}

	free(tempElementsArray);
	free(tempBoxIndexes);
}

/* initIndexArray(): this function is used to Initialize an array with the starting indeces for each block in the Elements Array */
void initIndexArray( int *a, int *blocksSize,  int d)
{
	int i;
	a[0] = 0;
	for( i = 1; i < d*d*d; i++)
		a[i] = a[i-1] + blocksSize[i-1];
}

/**********************/
/*** Sub-Procedures ***/
/**********************/

/* print(): prints the Elements of the given array */
void print( struct Element *elementsArray,  int N)
{
  int i;
  for( i = 0; i < N; i++)
    printf("[%f,%f,%f]\n", elementsArray[i].x, elementsArray[i].y, elementsArray[i].z);
}

/* printDeviceInfo(): prints device's information */
void printDeviceInfo(cudaDeviceProp *prop)
{
	/* Get CUDA Device Properties */
  cudaGetDeviceProperties(prop, 0);
	checkCuda( cudaGetLastError());

	printf("Device Name: %s\n", prop->name);
	//printf("Device Global Memory %ld MBs \n", prop->totalGlobalMem/(1<<20));
	printf("Warp Size: %d\n", prop->warpSize);
	printf("Max Threads per Block: %d\n", prop->maxThreadsPerBlock);
	printf("Max Threads Dimensions per Block: [%d, %d, %d]\n",prop->maxThreadsDim[0],prop->maxThreadsDim[1],prop->maxThreadsDim[2]);
	//printf("Max Blocks Dimensions per Grid: [%d, %d, %d]\n",prop->maxGridSize[0], prop->maxGridSize[1], prop->maxGridSize[2]);
}

void checkGpuMemory(  int N,  int d)
{
	float free_m,total_m,used_m;
	size_t free_t,total_t;

	cudaMemGetInfo(&free_t,&total_t);
	free_m =(uint)free_t/1048576.0;
	total_m=(uint)total_t/1048576.0;
	used_m=total_m-free_m;
	printf("Available Memory: %f MBs\nTotal Memory %f MBs\nMemory in use: %f MB\n", free_m, total_m, used_m);

	float memoryToBeNeeded = 2*12*N + 2*d*d*d*4 + N*8 + N*4;
	memoryToBeNeeded = memoryToBeNeeded/(1<<20);

	if(memoryToBeNeeded > free_m){
		printf("\n***Warning: Memory to be needed is estimated to %.2f MBs***\n", memoryToBeNeeded);
		printf("***Warning: Available Memory: %.2f MBs***\n", free_m);
		printf("***Warning: Program probably will not be executed***\n\n");
	}
}

/*****************************/
/*** Validation Procedures ***/
/*****************************/

/* validateBinning(): this function validates that eavh Element has been matched
*  in the right Box.*/
void validateBinning( struct Element *elementsArray, int *a,  int N,  int d)
{
	int i, k, dimX, dimY, dimZ;
	int blockIdInGrid;
	float interval = 1/((float)d);

	/* For Every Element find the coordinates of the block it belongs */
	for( k = 0; k < N; k++){
		/* find x block position */
		for( i = 0; i < d; i++){
			if( elementsArray[k].x >= i*interval && elementsArray[k].x < (i+1)*interval)
				dimX = i;
		}
		/* find y block position */
		for( i = 0; i < d; i++){
			if( elementsArray[k].y >= i*interval && elementsArray[k].y < (i+1)*interval)
				dimY = i;
		}
		/* find z block position */
		for( i = 0; i < d; i++){
			if( elementsArray[k].z >= i*interval && elementsArray[k].z < (i+1)*interval)
				dimZ = i;
		}

		/* block's id inside grid is idx + numXs*idy + numXs*numYs*idz */
		blockIdInGrid = dimX + d*dimY + d*d*dimZ;

		/* Check that element is assigned in the right block */
		assert( a[k] == blockIdInGrid);
	}

	printf(" Binning Validation: Passed\n");
}

/* validateResultsFromFile(): reads nearest Neighbor's distance from a file.
*  Distance was computed in R. Script is include in Folder.
*  Error Margin has been set in 10^(-14)*/
int validateResultsFromFile( struct Element *Q, struct Element *C, float *distances, int *NNindeces,  int N, int *oldIndexQ, int *oldIndexC)
{

	float *calculatedDistances = (float*)malloc(N*sizeof(float));
	int *calculatedIndeces = (int*)malloc(N*sizeof(int));
	readFloatsFromFile( calculatedDistances, N, "./test/MatlabDistances21.txt");
  readIntegersFromFile( calculatedIndeces, N, "./test/MatlabIndeces21.txt");
	//for(int i = 0; i < N; i++ )
		//printf(" %i %f\n", calculatedIndeces[i], calculatedDistances[i]);
	/*for(int i = 0; i < N; i++){
		printf("%d:Element: [%f %f %f] NN: [%f %f %f] NNdistance:%f, calculatedDistance: %f\n", i, Q[i].x, Q[i].y, Q[i].z, C[Nindeces[i]].x, C[Nindeces[i]].y, C[Nindeces[i]].z, distances[i], calculatedDistances[oldIndex[i]]);
	}*/
	int error = 0;
  printf("Validation..\n");
	for(int i = 0; i < N; i++){
	  if( abs(distances[i] - calculatedDistances[oldIndexQ[i]]) > 0.0001){
    //if( oldIndexC[NNindeces[i]] + 1 != calculatedIndeces[oldIndexQ[i]]){
			//printf(" Error: C:%f Matlab:%f \n", distances[i], calculatedDistances[oldIndexQ[i]]);
			//printf("C:%d Matlab:%d\n", oldIndexC[Nindeces[i]]+1, calculatedIndeces[oldIndexQ[i]]);
			//printf("Failed [New,Old] Q position:[%d,%d] [%f %f %f] NNdistance:%f | NN: C [New,Old] Position:[%d,%d] [%f %f %f],  calculatedDistance: %f, calculated Matlab Q Index = %d(-1 in my Table)\n",	i, oldIndexQ[i], Q[i].x, Q[i].y, Q[i].z, distances[i], NNindeces[i], oldIndexC[NNindeces[i]], C[NNindeces[i]].x, C[NNindeces[i]].y, C[NNindeces[i]].z,  calculatedDistances[oldIndexQ[i]], calculatedIndeces[oldIndexQ[i]]);
			error++;
		}
	}
	printf("ERRORS: %d\n", error);
	free( calculatedDistances);
  free(calculatedIndeces);
	if(error > 0)
		return 0;

	return 1;
}

/***********************/
/* Cuda Error Function */
/***********************/

/*checkCuda(): Checks For cuda calls Errors
/* Check for Error */
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

/***********************/
/*** File Procedures ***/
/***********************/

/* readElementsFromFile(): reads Elements from a file */
void readElementsFromFile(struct Element *elements,  int N,  const char *s)
{
	FILE *myfile;
	int i;
	int j;

	if(myfile=fopen( s, "r")){
		for(i = 0; i < N; i++){
			for (j = 0 ; j < 3; j++){
				if( j == 0)
					fscanf( myfile,"%f",&(elements[i].x));
				else if( j == 1)
					fscanf( myfile,"%f",&(elements[i].y));
				else
					fscanf( myfile,"%f",&(elements[i].z));
				}
			}
	}
	fclose(myfile);
}

/* readIntegersFromFile(): reads Integers from a file */
void readIntegersFromFile( int *a,  int N,  const char *s)
{
	FILE *myfile;
	int i;

	if(myfile=fopen( s, "r")){
		for(i = 0; i < N; i++){
					fscanf(myfile,"%d",&a[i]);
		}
	}
	fclose(myfile);
}

/* readFloatsFromFile(): reads floats from a file */
void readFloatsFromFile( float *a,  int N,  const char *s)
{
	FILE *myfile;
	int i;

	if(myfile=fopen( s, "r")){
		for(i = 0; i < N; i++){
				fscanf(myfile,"%f",&a[i]);
		}
	}
	fclose(myfile);
}
