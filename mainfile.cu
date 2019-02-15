/*
Copyright © 2019
Roberto Cavicchioli, Nicola Capodieci

See LICENSE.txt
*/

#include <cuda_runtime.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
#include <thread>

#include "vkcomp/stdafx.h"
#include "definitions.h"
#include "kernels.h"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


#ifdef CUDA_VALIDATION
#define CUDA_ERR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_KERN_CHECK { cudaKernelCheck();  }
#else
#define CUDA_ERR_CHECK(ans) {ans;}
#define CUDA_KERN_CHECK {}
#endif

inline void cudaKernelCheck() { CUDA_ERR_CHECK( cudaPeekAtLastError() ); } 

//timing variables
std::chrono::high_resolution_clock::time_point tStart;
std::chrono::high_resolution_clock::time_point tEnd;

struct ccheck{
TYPE *hPtr0;
TYPE *hPtr1;
};

struct ccheck *dataPeek;

void CUDART_CB endStreamCallback(cudaStream_t event, cudaError_t status, void *data){
	tEnd = std::chrono::high_resolution_clock::now();
	double *t = (double*)data;
	*t = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd-tStart).count();
	std::cout << "ELAPSED (CB) : " << *t << std::endl;
}

int main(int argc, char* argv[]){

setFIFO99andCore(1);
	
//Methodology selected
int32_t method = BASELINE;
int iterations = ITERATIONS;
int samples = SAMPLES;

if(argc>1)
	method = atoi(argv[1]);
else 
	std::cout << "WARNING: no kernel invocation methodology is selected. Defaulting to baseline" << std::endl;

if(argc>2)
	iterations = atoi(argv[2]);

if(argc>3)
	samples = atoi(argv[3]);

std::cout << "Number of iterations " << iterations << std::endl;
std::cout << "Number of samples " << samples << std::endl;

dataPeek = (struct ccheck*) malloc(sizeof(struct ccheck));

//allocations:
TYPE* hIn; TYPE* dIn;
TYPE* hOut; TYPE* dOut;
TYPE* hWeights; TYPE* dWeights;
TYPE* hFilter; TYPE* dFilter;
double *times_sub, *times_end;
double elapsed_time;

//Host Side
CUDA_ERR_CHECK(cudaMallocHost((void**)&hIn, sizeof(TYPE)*INH*INW));
CUDA_ERR_CHECK(cudaMallocHost((void**)&hOut, sizeof(TYPE)*OUTH*OUTW));
CUDA_ERR_CHECK(cudaMallocHost((void**)&hWeights, sizeof(TYPE)*WH*WW));
CUDA_ERR_CHECK(cudaMallocHost((void**)&hFilter, sizeof(TYPE)*KERNEL_SIZE*KERNEL_SIZE));

dataPeek->hPtr0 = hIn;
dataPeek->hPtr1 = hOut;

times_sub = (double*)malloc(samples * sizeof(double));
times_end = (double*)malloc(samples * sizeof(double));

//Device Side
CUDA_ERR_CHECK(cudaMalloc((void**)&dIn, sizeof(TYPE)*INH*INW));
CUDA_ERR_CHECK(cudaMalloc((void**)&dOut, sizeof(TYPE)*OUTH*OUTW));
CUDA_ERR_CHECK(cudaMalloc((void**)&dWeights, sizeof(TYPE)*WH*WW));
CUDA_ERR_CHECK(cudaMalloc((void**)&dFilter, sizeof(TYPE)*KERNEL_SIZE*KERNEL_SIZE));

//Stream creation
cudaStream_t stream;
CUDA_ERR_CHECK(cudaStreamCreate(&stream));

CUDA_ERR_CHECK(cudaStreamSynchronize(stream));

cudaStream_t streamForGraph;
CUDA_ERR_CHECK(cudaStreamCreate(&streamForGraph));
CUDA_ERR_CHECK(cudaStreamSynchronize(streamForGraph));


//Initialize data
srand(static_cast <unsigned> (SEED));

for(size_t i=0; i<INH*INW; i++)
	hIn[i] = (TYPE)(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));

for(size_t i=0; i<OUTH*OUTW; i++)
	hOut[i] = (TYPE)(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));

for(size_t i=0; i<WH*WW; i++)
	hWeights[i] = (TYPE)(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));

for(size_t i=0; i<KERNEL_SIZE*KERNEL_SIZE; i++)
	hFilter[i] = (TYPE)(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));

dim3 threadDim(TILE_WIDTH,TILE_WIDTH); 

dim3 dimGrid(ceil(((float)INW) / threadDim.x),
			 ceil(((float)OUTH) / threadDim.y));
			 
std::cout << threadDim.x << " " << threadDim.y << " " << threadDim.z << std::endl;
std::cout << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z << std::endl;

for (int j=0; j<samples; j++)
{

switch(method){
	case BASELINE:
	
	std::cout << "Executing BASELINE methodology..." << std::endl;

	//time here!
	tStart = std::chrono::high_resolution_clock::now();
	
	CUDA_ERR_CHECK(cudaMemcpyAsync(dWeights, hWeights, sizeof(TYPE)*WH*WW, cudaMemcpyHostToDevice, stream));
	CUDA_ERR_CHECK(cudaMemcpyAsync(dFilter, hFilter, sizeof(TYPE)*KERNEL_SIZE*KERNEL_SIZE, cudaMemcpyHostToDevice, stream));
	CUDA_ERR_CHECK(cudaMemcpyAsync(dIn, hIn, sizeof(TYPE)*INH*INW, cudaMemcpyHostToDevice, stream));
	CUDA_ERR_CHECK(cudaMemcpyAsync(dOut, hOut, sizeof(TYPE)*OUTH*OUTW, cudaMemcpyHostToDevice, stream));
	
	for(size_t i=0; i<iterations; i++){


		if((i%2)==0){

			matrixMulGPU<<<dimGrid
		, threadDim, 0,stream>>>(dIn,dOut,dWeights,INW,INH); CUDA_KERN_CHECK
			convolutionGPU<<<dimGrid
		, threadDim, 0,stream>>>(dOut,dIn,dFilter,OUTW,OUTH); CUDA_KERN_CHECK
			activationGPU<<<dimGrid
		, threadDim, 0,stream>>>(dIn, dOut, INW, INH, ACT_RELU); CUDA_KERN_CHECK

		}else{
			matrixMulGPU<<<dimGrid
		, threadDim, 0,stream>>>(dOut,dIn,dWeights,INW,INH); CUDA_KERN_CHECK
			convolutionGPU<<<dimGrid
		, threadDim, 0,stream>>>(dIn,dOut,dFilter,OUTW,OUTH); CUDA_KERN_CHECK
			activationGPU<<<dimGrid
		, threadDim, 0,stream>>>(dOut, dIn, INW, INH, ACT_SIGMOID); CUDA_KERN_CHECK
		}

	}
	
	CUDA_ERR_CHECK(cudaMemcpyAsync(hIn, dIn, sizeof(TYPE)*INH*INW, cudaMemcpyDeviceToHost, stream));
	CUDA_ERR_CHECK(cudaMemcpyAsync(hOut, dOut, sizeof(TYPE)*OUTH*OUTW, cudaMemcpyDeviceToHost, stream));
	CUDA_ERR_CHECK(cudaStreamAddCallback (stream, endStreamCallback, &times_end[j], 0));
	
	//time here and show elapsed!
	tEnd = std::chrono::high_resolution_clock::now();
	elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd-tStart).count();
	std::cout << "ELAPSED : " << elapsed_time << std::endl;
	
	break;
	
	case DYNPAR:
	
	std::cout << "Executing DYNPAR methodology..." << std::endl;
		
	//time here
	tStart = std::chrono::high_resolution_clock::now();
	
	CUDA_ERR_CHECK(cudaMemcpyAsync(dIn, hIn, sizeof(TYPE)*INH*INW, cudaMemcpyHostToDevice, stream));
	CUDA_ERR_CHECK(cudaMemcpyAsync(dOut, hOut, sizeof(TYPE)*OUTH*OUTW, cudaMemcpyHostToDevice, stream));
	CUDA_ERR_CHECK(cudaMemcpyAsync(dWeights, hWeights, sizeof(TYPE)*WH*WW, cudaMemcpyHostToDevice, stream));
	CUDA_ERR_CHECK(cudaMemcpyAsync(dFilter, hFilter, sizeof(TYPE)*KERNEL_SIZE*KERNEL_SIZE, cudaMemcpyHostToDevice, stream));
	
	dynParKernel<<<1,1,0,stream>>>(dIn,dOut, dWeights, dFilter, iterations); CUDA_KERN_CHECK

	CUDA_ERR_CHECK(cudaMemcpyAsync(hIn, dIn, sizeof(TYPE)*INH*INW, cudaMemcpyDeviceToHost, stream));
	CUDA_ERR_CHECK(cudaMemcpyAsync(hOut, dOut, sizeof(TYPE)*OUTH*OUTW, cudaMemcpyDeviceToHost, stream));
	CUDA_ERR_CHECK(cudaStreamAddCallback (stream, endStreamCallback, &times_end[j], 0));

	//time here and show elapsed!
	tEnd = std::chrono::high_resolution_clock::now();
	elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd-tStart).count();
	std::cout << "ELAPSED : " << elapsed_time << std::endl;
	
	break;
	
	case CUDAGRAPH:
	
	std::cout << "Executing CUDAGRAPH methodology..." << std::endl;
	
	cudaGraph_t graph;
	cudaGraphExec_t graphExec;

	CUDA_ERR_CHECK(cudaStreamBeginCapture(stream));

	CUDA_ERR_CHECK(cudaMemcpyAsync(dWeights, hWeights, sizeof(TYPE)*WH*WW, cudaMemcpyHostToDevice, stream));
	CUDA_ERR_CHECK(cudaMemcpyAsync(dFilter, hFilter, sizeof(TYPE)*KERNEL_SIZE*KERNEL_SIZE, cudaMemcpyHostToDevice, stream));
	CUDA_ERR_CHECK(cudaMemcpyAsync(dIn, hIn, sizeof(TYPE)*INH*INW, cudaMemcpyHostToDevice, stream));
	CUDA_ERR_CHECK(cudaMemcpyAsync(dOut, hOut, sizeof(TYPE)*OUTH*OUTW, cudaMemcpyHostToDevice, stream));
	
	for(size_t i=0; i<iterations; i++){


		if((i%2)==0){
			matrixMulGPU<<<dimGrid
		, threadDim, 0,stream>>>(dIn,dOut,dWeights,INW,INH); 
			convolutionGPU<<<dimGrid
		, threadDim, 0,stream>>>(dOut,dIn,dFilter,OUTW,OUTH);
			activationGPU<<<dimGrid
		, threadDim, 0,stream>>>(dIn, dOut, INW, INH, ACT_RELU);
		}else{
			matrixMulGPU<<<dimGrid
		, threadDim, 0,stream>>>(dOut,dIn,dWeights,INW,INH); 
			convolutionGPU<<<dimGrid
		, threadDim, 0,stream>>>(dIn,dOut,dFilter,OUTW,OUTH);
			activationGPU<<<dimGrid
		, threadDim, 0,stream>>>(dOut, dIn, INW, INH, ACT_SIGMOID);
		}

		
	}
	
	CUDA_ERR_CHECK(cudaMemcpyAsync(hIn, dIn, sizeof(TYPE)*INH*INW, cudaMemcpyDeviceToHost, stream));
	CUDA_ERR_CHECK(cudaMemcpyAsync(hOut, dOut, sizeof(TYPE)*OUTH*OUTW, cudaMemcpyDeviceToHost, stream));
	CUDA_ERR_CHECK(cudaStreamEndCapture(stream, &graph));
	CUDA_ERR_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
	//set up graph-stream operations done.
	
	//set Timer here
	tStart = std::chrono::high_resolution_clock::now();

	//Launch!
	CUDA_ERR_CHECK(cudaGraphLaunch(graphExec, streamForGraph));
	CUDA_ERR_CHECK(cudaStreamAddCallback (streamForGraph, endStreamCallback, &times_end[j], 0));
	
	//time here and show elapsed, as this last launch is async.
	tEnd = std::chrono::high_resolution_clock::now();
	elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd-tStart).count();
	std::cout << "ELAPSED : " << elapsed_time << std::endl;

	CUDA_ERR_CHECK(cudaGraphDestroy(graph));
	CUDA_ERR_CHECK(cudaGraphExecDestroy(graphExec));


	break;
	
	default: break;
}

CUDA_ERR_CHECK(cudaDeviceSynchronize());

times_sub[j] = elapsed_time;

}

for (int i=0; i< samples; i++)
	std::cout<< times_sub[i] << std::endl;

for (int i=0; i< samples; i++)
	std::cout<< times_end[i] << std::endl;

CUDA_ERR_CHECK(cudaDeviceReset());

return 0;

}
