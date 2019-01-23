#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>


/* Element: 3 Dimensions Element in [0,1)^3 */
struct Element{
	float x;
	float y;
	float z;
};

/* Functions Declaration */
void binning( struct Element *elementsArray, int *a, int N, int d, int *blocksSize, int *oldIndex);
void nearestNeighbor( int *NNindeces, float *NNdistances, struct Element *hostC, struct Element *hostQ, int *indexArrayC, int *indexArrayQ, int N, int d);

/* Sub-Procedures Declaration */
void print( struct Element *elementsArray, int N);
void initIndexArray( int *a, int *blocksSize, int d);
float calculateDistance( struct Element *a, struct Element *b);

/* File Operation Functions */
void readFloatsFromFile( float *a, int N, const char *s);
void readIntegersFromFile( int *a, int N, const char *s);
void readElementsFromFile(struct Element *elements, int N, const char *s);

/* Validation Functions Declaration */
void validateBinning( struct Element *elemntsArray, int *a, int N, int d);
int validateResultsFromFile( struct Element *Q, struct Element *C, float *distances, int *Nindeces, int N, int *oldIndexQ, int *oldIndexC);

/*********************/
/*** Main Function ***/
/*********************/
int main(int argc, char const *argv[]) {

	/* Variables to hold the entire execetucion time */
	struct timeval startOfProgramTime, endOfProgramTime;
	float wholeTime;
	gettimeofday (&startOfProgramTime, NULL);

	int i;
 	int d_power = 5;
  int N_power = 21;							             // N^N_power
	//int l_power = N_power - d_power*3;         // grid size : 2^(3*d_power), Block size: 32*32 = 2^(2*5), so grid_power*block_power = 3*d_power + 10
	int N = 1 << N_power;
	int d = 1 << d_power;
	printf("Serial Nearest Neighbor\n");
	printf("N is 2^%d and d is 2^%d\n",  N_power, d_power);

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

	/* Variables to hold execution time */
	struct timeval startwtime, endwtime;
	float executionTime;

	/* Table Of starting Indeces for each block */
	int *indexArrayC;		//Elements of this array contain the starting index of Element's table for every block for Array C
	int *indexArrayQ;		//Elements of this array contain the starting index of Element's table for every block for Array Q
	indexArrayC = (int *)malloc( d*d*d*sizeof(int));
	indexArrayQ = (int *)malloc( d*d*d*sizeof(int));
	initIndexArray( indexArrayC, blocksSizeC, d);
	initIndexArray( indexArrayQ, blocksSizeQ, d);

	/* Find Nearest Neighbor */
	/* Allocate Space to hold distnces and indeces of Nearest Neighbors */
	int *NNindeces = (int *)malloc(N*sizeof(int));
	float *NNdistances = (float *)malloc(N*sizeof(float));
	for( i = 0; i < N; i++)
		NNdistances[i] = 2;
	gettimeofday (&startwtime, NULL);
	nearestNeighbor( NNindeces, NNdistances, hostC, hostQ, indexArrayC, indexArrayQ, N, d);
	gettimeofday (&endwtime, NULL);
	executionTime = (float)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
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

/***********************/
/*** Main Procedures ***/
/***********************/

/* binning(): Data Binning/Bucketing The original data values
*	 which fall in a given small interval, a bin, are assigned
*	 a value representative of that interval.
*	 This function rearranges the whole array based on  the bins.
*/
void binning( struct Element *elementsArray, int *a, int N, int d, int *blocksSize, int *oldIndex)
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


void nearestNeighbor( int *NNindeces, float *NNdistances, struct Element *hostC, struct Element *hostQ, int *indexArrayC, int *indexArrayQ, int N, int d)
{
	int limitC, limitQ, i, j, k, tempIndex;
	float distance, minDistance;

	/*Search for Primary Candidates */
	for( i = 0; i < d*d*d; i++){
		if( i != d*d*d-1)
			{	limitC = indexArrayC[i+1]; limitQ = indexArrayQ[i+1]; }
		else
			{	limitC = N; limitQ = N; }
		for( j = indexArrayQ[i]; j < limitQ; j++){
			minDistance = 2;
			tempIndex = -1;
			for( k = indexArrayC[i]; k < limitC; k++){
				distance = calculateDistance( &hostQ[j], &hostC[k]);
				if( distance < minDistance){
					minDistance = distance;
					tempIndex = k;
				}
			}
			NNdistances[j] = minDistance;
			NNindeces[j]	= tempIndex;
		}
	}

	int idx, idy, idz, searchBoxId;
	int x,y,z;
	float distanceZ, distanceX, distanceY, interval, blockDistance;
	interval = 1/((float)d);
	/* Search for Secondary Candidates */
	for(i = 0; i < d*d*d; i++){
		/* 1st Define Blocks id [x,y,z] */
		idx = i%d;							// A blockId is defined as: x + d*y + d*d*z
		idy = (i/d)%d;
		idz = (i/(d*d))%d;
		if( i != d*d*d-1)
			limitQ = indexArrayQ[i+1];
		else
			limitQ = N;
		/* For Every Element in box i search near boxes */
		for( j = indexArrayQ[i]; j < limitQ; j++){
			/* Define near Box to search */
			/* Define z */
			for( z = idz - 1; z <= idz + 1; z++ ){
				if( z >= 0 && z < d){
					if( z == idz - 1)
						distanceZ = abs( hostQ[j].z - idz*interval);
					else if( z == idz)
						distanceZ = 0;
					else
						distanceZ = abs( z*interval - hostQ[j].z);
					/* Define y */
					for( y = idy - 1; y <= idy + 1; y++){
						if( y >= 0 && y < d){
							if( y == idy - 1)
								distanceY = abs( hostQ[j].y - idy*interval);
							else if( y == idy)
								distanceY = 0;
							else
								distanceY = abs( y*interval - hostQ[j].y);
							/* Define x*/
							for( x = idx - 1; x <= idx + 1; x++){
								if( x >= 0 && x < d){
									if( x == idx - 1)
										distanceX = abs( hostQ[j].x - idx*interval);
									else if( x == idx)
										distanceX = 0;
									else
										distanceX = abs( x*interval - hostQ[j].x);
									/* Block to search has been defined */
									/* Calculate Distance from Block */
									blockDistance = sqrt( pow(distanceX, 2) + pow(distanceY, 2) + pow(distanceZ, 2));
									/* Set id of box to be searched */
									searchBoxId = x + d*y +d*d*z;
									if( !(idx == x && idy == y && idz ==z)){
										if( blockDistance < NNdistances[j]){

											/* Set Limit of the Box */
											if( searchBoxId != d*d*d-1)
												limitC = indexArrayC[searchBoxId+1];
											else
												limitC = N;
											/* Search All his Elements */
											for( k = indexArrayC[searchBoxId]; k < limitC; k++){
												distance = calculateDistance( &hostQ[j], &hostC[k]);
												if( distance < NNdistances[j]){
													NNdistances[j] = distance;
													NNindeces[j] = k;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}

/*****************************/
/*** Validation Procedures ***/
/*****************************/

/* validateBinning(): this function validates that eavh Element has been matched
*   in the right Box.
*/
void validateBinning( struct Element *elementsArray, int *a, int N, int d)
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

int validateResultsFromFile( struct Element *Q, struct Element *C, float *distances, int *Nindeces, int N, int *oldIndexQ, int *oldIndexC)
{
	float *calculatedDistances = (float*)malloc(N*sizeof(float));
	int *calculatedIndeces = (int*)malloc(N*sizeof(int));
	//int *calculatedIndeces = (int*)malloc(N*sizeof(int));
	readFloatsFromFile( calculatedDistances, N, "./test/MatlabDistances21.txt");
	readIntegersFromFile( calculatedIndeces, N, "./test/MatlabIndeces21.txt");
	//for( int i = 0; i < N; i++)
		//printf("%d: %d\n", i, oldIndex[i]);
	//for(int i = 0; i < N; i++)
		//printf("%d:Element: [%f %f %f] NN: [%f %f %f] NNdistance:%f, calculatedDistance: %f\n", i, Q[i].x, Q[i].y, Q[i].z, C[Nindeces[i]].x, C[Nindeces[i]].y, C[Nindeces[i]].z, distances[i], calculatedDistances[oldIndex[i]]);
		int error = 0;
	for(int i = 0; i < N; i++){
		if( abs(distances[i] - calculatedDistances[oldIndexQ[i]]) > 0.00001){
		//if( oldIndexC[Nindeces[i]] + 1 != calculatedIndeces[oldIndexQ[i]]){
			printf("Failed [New,Old] Q position:[%d,%d] [%f %f %f] NNdistance:%lf | NN: C [New,Old] Position:[%d,%d] [%f %f %f],  calculatedDistance: %lf, calculated Matlab Q Index = %d(-1 in my Table)\n",	i, oldIndexQ[i], Q[i].x, Q[i].y, Q[i].z, distances[i], Nindeces[i], oldIndexC[Nindeces[i]], C[Nindeces[i]].x, C[Nindeces[i]].y, C[Nindeces[i]].z,  calculatedDistances[oldIndexQ[i]], calculatedIndeces[oldIndexQ[i]]);
			//printf("C:%d Matlab:%d\n", oldIndexC[Nindeces[i]]+1, calculatedIndeces[oldIndexQ[i]]);
			error++;
		}
	}
	printf("ERRORS: %d\n", error);
	free( calculatedDistances);
	free( calculatedIndeces);
	if(error > 0)
		return 0;

	return 1;
}


/**********************/
/*** Sub-Procedures ***/
/**********************/

/* print(): prints the Elements of the given array */
void print( struct Element *elementsArray, int N)
{
  int i;
  for( i = 0; i < N; i++)
    printf("[%f,%f,%f]\n", elementsArray[i].x, elementsArray[i].y, elementsArray[i].z);
}

/* initIndexArray(): this function is used to Initialize an array with the starting indeces for each block in the Elements Array */
void initIndexArray( int *a, int *blocksSize, int d)
{
	int i;
	a[0] = 0;
	for( i = 1; i < d*d*d; i++)
		a[i] = a[i-1] + blocksSize[i-1];
}

/* calculateDistance(): calculates euclidean distance between two elements */
float calculateDistance( struct Element *a, struct Element *b)
{
	float temp;
	temp = sqrtf( powf(a->x - b->x, 2) + powf(a->y - b->y, 2) + powf(a->z - b->z, 2));
	return temp;
}

/***********************/
/*** File Procedures ***/
/***********************/

/* readElementsFromFile(): reads Elements from a file */
void readElementsFromFile(struct Element *elements, int N, const char *s)
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
void readIntegersFromFile( int *a, int N, const char *s)
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

/* readdoublesFromFile(): reads doubles from a file */
void readFloatsFromFile( float *a, int N, const char *s)
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
