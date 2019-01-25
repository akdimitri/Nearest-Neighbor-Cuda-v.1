# Nearest-Neighbor-Cuda-v.1

Nearest Neighbor Grid Implementation in CUDA


Author: Dimitris Antoniadis ~ akdimitri@auth.gr

************************************************************************************
***This Project has been implemented in the context of Academic Subject          
Parallel and Distributed Systems of Aristotle University of Thessaloniki(AUTH)***
************************************************************************************

*Project 3*

This repository includes a serial CPU implementation and a parallel GPU implementation.

*Compilation*
Makefile 
The following command creates the parallel program
>>$ make    
or
>>$ make parallel

The following command creates the parallel program using Shared Memory
>>$ make shared 

The following command creates the serial program
>>$make serial

Cleanup
>>$ make clean

*Execution*
>>$ ./cudaSerial argument1

>>$ ./cudaParallel argument1

>>$ ./cudaParallelShared argument1

where d = 2^argument1 and grid size is d*d*d

*** Important ***

1)Choose the power of the size of the Array N_power.

2)Open matlabScript.m and change N_power value.(If you wish change the name of the files and the 
output folder)

3)Run matlabScript

4)Open *.cu file and change the value of N_power and set the desired value of d_power(If you have changed
matlabScript's output files, make sure to change them in the folder too. You have to change 4 path strings
(2 in main function, 2 in validationFromFile function)

5)Compile the file you wish to run

6)Run the file

