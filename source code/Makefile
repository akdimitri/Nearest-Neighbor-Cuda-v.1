##############################################################################
# FILE: Makefile CUDA																												 #
# DESCRIPTION:																															 #
# Makefile for CUDA C Language																							 #
# AUTHOR: Dimitris Antoniadis																								 #
# LAST REVISED:  23/1/19																										 #
##############################################################################

#compiler
CC    =nvcc

#FLAGS   -Wconversion  used to show implicit conversion warnings
FLAGS   = -arch=sm_30

END_FLAGS =	-lm

all:parallel

clean:
	/bin/rm -rf	cudaParallel cudaSerial

parallel:knn-grid-cuda-parallel-file-float.cu
	$(CC) $(FLAGS) knn-grid-cuda-parallel-file-float.cu -o cudaParallel
serial:knn-grid-cuda-serial-file-float.cu
	$(CC) $(FLAGS) knn-grid-cuda-serial-file-float.cu -o cudaSerial

# CHECK Tabs Seperators cat -e -t -v makefile_name #
# All actions should start with a tab #
# Tab = ^I, End Of Line = $#
