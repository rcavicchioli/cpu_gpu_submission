/*
Copyright © 2019
Roberto Cavicchioli, Nicola Capodieci

See LICENSE.txt
*/

#ifndef KERNELS_H
#define KERNELS_H
#include "definitions.h"

#ifndef TYPE
#define TYPE float
#endif

__global__ void convolutionGPU(TYPE *d_Result, TYPE *d_Data, TYPE *d_Filter, int dataW, int dataH );
__global__ void activationGPU(TYPE *d_Result, TYPE *d_Data, int dataW, int dataH, int funct);
__global__ void matrixMulGPU(TYPE *d_Result, TYPE *d_Data, TYPE *d_Weights, int dataW, int dataH );
__global__ void dynParKernel(TYPE *dIn, TYPE *dOut, TYPE *dWeights, TYPE *dFilter, int funct);
#endif