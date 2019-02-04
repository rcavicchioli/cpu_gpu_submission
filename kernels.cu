/*
Copyright © 2019
Roberto Cavicchioli, Nicola Capodieci

See LICENSE.txt
*/

#include "kernels.h"
#include "definitions.h"
#include <stdio.h>

#ifndef TYPE
#define TYPE float
#endif

__global__ void convolutionGPU(TYPE *d_Result, TYPE *d_Data, TYPE *d_Filter, int dataW, int dataH )
{
    const int gLocX = threadIdx.x + blockIdx.x * blockDim.x; 
    const int gLocY = threadIdx.y + blockIdx.y * blockDim.y;
    TYPE sum = 0;
    TYPE value = 0;
    for (int i = 0; i < KERNEL_SIZE; i++)
        for (int j = 0; j < KERNEL_SIZE; j++)
        {
            if ((gLocX-1+i)<0)
                value = 0;
            else
            if ((gLocX-1+i)>=dataW)
                value = 0;
            else
            {
                if ((gLocY-1+j)<0)
                    value = 0;    
                else
                if ((gLocY-1+j)>=dataH)
                    value = 0;
                else
                    value = d_Data[(gLocY-1+j)*dataW+(gLocX-1+i)];
            }           

            sum+=value*d_Filter[i + KERNEL_SIZE * j];
        }
    d_Result[gLocX*dataW+gLocY]= sum;
}
__global__ void activationGPU(TYPE *d_Result, TYPE *d_Data, int dataW, int dataH, int funct)
{
    const int gLoc = threadIdx.x + blockIdx.x * blockDim.x + threadIdx.y * dataW + blockIdx.y * blockDim.y * dataW; 
    if (funct == ACT_RELU)
        d_Result[gLoc] = (d_Data[gLoc]>(TYPE)0.0)?d_Data[gLoc]:(TYPE)0.0;
    else
        d_Result[gLoc] = (TYPE)(1.0) / ((TYPE)(1.0) + __expf((float)d_Data[gLoc]));
}

__global__ void matrixMulGPU(TYPE *d_Result, TYPE *d_Data, TYPE *d_Weights, int dataW, int dataH )
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dataW && col < dataH) {
        TYPE sum = 0;
        for (int ii = 0; ii < dataW; ii++) {
            sum += d_Data[row * dataW + ii] * d_Weights[ii * dataH + col];
        }
    d_Result[row * dataW + col] = sum;
  }

}

__global__ void dynParKernel(TYPE *dIn, 
    TYPE *dOut, 
    TYPE *dWeights, 
    TYPE *dFilter, int funct)
{
    dim3 threadDim(TILE_WIDTH,TILE_WIDTH); 
    dim3 dimGrid(ceil(((float)INW) / threadDim.x),
                ceil(((float)INH) / threadDim.y));
    
	matrixMulGPU<<<dimGrid, threadDim, 0>>>(dIn,dOut,dWeights,INW,INH);
	convolutionGPU<<<dimGrid, threadDim, 0>>>(dOut,dIn,dFilter,OUTW,OUTH);
	activationGPU<<<dimGrid, threadDim, 0>>>(dIn, dOut, INW, INH, funct);
}