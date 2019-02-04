
/*
Copyright © 2019
Roberto Cavicchioli, Nicola Capodieci

See LICENSE.txt
*/

#include "../definitions.h"

#include "VulkanCompute.h"
#include "stdafx.h"

#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>

#ifndef TYPE
#define TYPE float
#endif

int main(int argc, char* argv[]){

setFIFO99andCore(1);

int iterations = 1;
int samples = 10;

if(argc>1)
	iterations = atoi(argv[1]);

if(argc>2)
	samples = atoi(argv[2]);

double *times_sub, *times_end;
double elapsed_time;

times_sub = (double*)malloc(samples * sizeof(double));
times_end = (double*)malloc(samples * sizeof(double));

//Vulkan context creation
VulkanCompute vk;
vk.createContext();
vk.printContextInformation();

char buff[FILENAME_MAX];
GetCurrentDir( buff, FILENAME_MAX );

CrossFileAdapter crossFileAdapter;
std::string cwd(buff);

crossFileAdapter.setAbsolutePath(cwd+FILE_SEPARATOR+"shaders"+FILE_SEPARATOR+"matrixmul.comp");

if (vk.loadAndCompileShader(crossFileAdapter, "matmul") == -1)
{
	std::cout << "Error in compiling shader matrixmul.comp" << std::endl;
	exit(-1);
}

crossFileAdapter.setAbsolutePath(cwd+FILE_SEPARATOR+"shaders"+FILE_SEPARATOR+"convolution.comp");
if (vk.loadAndCompileShader(crossFileAdapter, "conv") == -1)
{
	std::cout << "Error in compiling shader convolution.comp" << std::endl;
	exit(-1);
}

crossFileAdapter.setAbsolutePath(cwd+FILE_SEPARATOR+"shaders"+FILE_SEPARATOR+"activation.comp");
if (vk.loadAndCompileShader(crossFileAdapter, "activ") == -1)
{
	std::cout << "Error in compiling shader activation.comp" << std::endl;
	exit(-1);
}

//host and device allocations: returned ptr is always host side.
TYPE* in = (TYPE*) vk.deviceSideAllocation(sizeof(TYPE)*INH*INW, BufferUsage::BUF_INOUT);
TYPE* out = (TYPE*) vk.deviceSideAllocation(sizeof(TYPE)*OUTH*OUTW, BufferUsage::BUF_INOUT);
TYPE* weights = (TYPE*) vk.deviceSideAllocation(sizeof(TYPE)*WW*WH, BufferUsage::BUF_INOUT);
TYPE* filter = (TYPE*) vk.deviceSideAllocation(sizeof(TYPE)*KERNEL_SIZE*KERNEL_SIZE, BufferUsage::BUF_INOUT);

//init values:
srand(static_cast <unsigned> (SEED));

for(size_t i=0; i<INH*INW; i++)
	in[i] = (TYPE)(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));
	
for(size_t i=0; i<OUTH*OUTW; i++)
	out[i] = (TYPE)(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));

for(size_t i=0; i<WH*WW; i++)
	weights[i] = (TYPE)(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));

for(size_t i=0; i<KERNEL_SIZE*KERNEL_SIZE; i++)
	filter[i] = (TYPE)(static_cast <float> (rand()) / static_cast <float> (RAND_MAX));

//timing variables
std::chrono::high_resolution_clock::time_point tStart;
std::chrono::high_resolution_clock::time_point tEnd;

//launch configurations
ComputeWorkDistribution_t threadDim(TILE_WIDTH,TILE_WIDTH); 

ComputeWorkDistribution_t dimGrid(ceil(((float)INW) / threadDim.x),
                            ceil(((float)OUTH) / threadDim.y));

std::cout << threadDim.x << " " << threadDim.y << " " << threadDim.z << std::endl;
std::cout << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z << std::endl;

//Pipelines creation: 

vk.startCreatePipeline("matmul");
		vk.setArg(PPTR(in), "matmul",4);
		vk.setArg(PPTR(out), "matmul", 5);
		vk.setArg(PPTR(weights), "matmul", 6);
		vk.setSymbol(0, sizeof(int));
		vk.setSymbol(1, sizeof(int));
		vk.setLaunchConfiguration(dimGrid, threadDim);
PIPELINE_HANDLE hMatmulAB = vk.finalizePipeline();

vk.startCreatePipeline("matmul");
		vk.setArg(PPTR(out), "matmul",4);
		vk.setArg(PPTR(in), "matmul", 5);
		vk.setArg(PPTR(weights), "matmul", 6);
		vk.setSymbol(0, sizeof(int));
		vk.setSymbol(1, sizeof(int));
		vk.setLaunchConfiguration(dimGrid, threadDim);
PIPELINE_HANDLE hMatmulBA = vk.finalizePipeline();

vk.startCreatePipeline("conv");
		vk.setArg(PPTR(out), "conv",4);
		vk.setArg(PPTR(in), "conv", 5);
		vk.setArg(PPTR(filter), "conv", 6);
		vk.setSymbol(0, sizeof(int));
		vk.setSymbol(1, sizeof(int));
		vk.setLaunchConfiguration(dimGrid, threadDim);
PIPELINE_HANDLE hConvAB = vk.finalizePipeline();

vk.startCreatePipeline("conv");
		vk.setArg(PPTR(in), "conv",4);
		vk.setArg(PPTR(out), "conv", 5);
		vk.setArg(PPTR(filter), "conv", 6);
		vk.setSymbol(0, sizeof(int));
		vk.setSymbol(1, sizeof(int));
		vk.setLaunchConfiguration(dimGrid, threadDim);
PIPELINE_HANDLE hConvBA = vk.finalizePipeline();

vk.startCreatePipeline("activ");
		vk.setArg(PPTR(in), "activ",4);
		vk.setArg(PPTR(out), "activ", 5);
		vk.setSymbol(0, sizeof(int));
        vk.setSymbol(1, sizeof(int));
        vk.setSymbol(2, sizeof(int));
		vk.setLaunchConfiguration(dimGrid, threadDim);
PIPELINE_HANDLE hActivAB = vk.finalizePipeline();

vk.startCreatePipeline("activ");
		vk.setArg(PPTR(out), "activ",4);
		vk.setArg(PPTR(in), "activ", 5);
		vk.setSymbol(0, sizeof(int));
        vk.setSymbol(1, sizeof(int));
        vk.setSymbol(2, sizeof(int));
		vk.setLaunchConfiguration(dimGrid, threadDim);
PIPELINE_HANDLE hActivBA = vk.finalizePipeline();

vk.startCreateCommandList();

        vk.synchBuffer(PPTR(weights), HOST_TO_DEVICE);
        vk.synchBuffer(PPTR(filter), HOST_TO_DEVICE);
		vk.synchBuffer(PPTR(in), HOST_TO_DEVICE);
        vk.synchBuffer(PPTR(out), HOST_TO_DEVICE);

        for(size_t i=0; i<iterations; i++){
		    if((i%2)==0){
                vk.selectPipeline(hMatmulAB);
                vk.copySymbolInt(INW,"matmul",0);
                vk.copySymbolInt(INH,"matmul",1);
                vk.launchComputation("matmul");
                vk.selectPipeline(hConvAB);
                vk.copySymbolInt(OUTW,"conv",0);
                vk.copySymbolInt(OUTH,"conv",1);
                vk.launchComputation("conv");
                vk.selectPipeline(hActivAB);
                vk.copySymbolInt(INW,"activ",0);
                vk.copySymbolInt(INH,"activ",1);
                vk.copySymbolInt(ACT_RELU,"activ",2);
                vk.launchComputation("activ");
            }else{
                vk.selectPipeline(hMatmulBA);
                vk.copySymbolInt(INW,"matmul",0);
                vk.copySymbolInt(INH,"matmul",1);
                vk.launchComputation("matmul");
                vk.selectPipeline(hConvBA);
                vk.copySymbolInt(OUTW,"conv",0);
                vk.copySymbolInt(OUTH,"conv",1);
                vk.launchComputation("conv");
            	vk.selectPipeline(hActivBA);
                vk.copySymbolInt(INW,"activ",0);
                vk.copySymbolInt(INH,"activ",1);
                vk.copySymbolInt(ACT_SIGMOID,"activ",2);
                vk.launchComputation("activ");
            }
        }

		vk.synchBuffer(PPTR(in), DEVICE_TO_HOST);
        vk.synchBuffer(PPTR(out), DEVICE_TO_HOST);

vk.finalizeCommandList();

vk.deviceSynch();

for (int j=0; j<samples; j++)
{

//timer starts here.
tStart = std::chrono::high_resolution_clock::now();

vk.submitWork();

//and stops here after launch...
tEnd = std::chrono::high_resolution_clock::now();
times_sub[j] = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd-tStart).count();
std::cout << "ELAPSED : " << times_sub[j] << std::endl;

vk.deviceSynch();

tEnd = std::chrono::high_resolution_clock::now();
times_end[j] = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd-tStart).count();
std::cout << "ELAPSED (\"CB\"): " << times_end[j] << std::endl;

}

for (int i=0; i< samples; i++)
	std::cout<< times_sub[i] << std::endl;

for (int i=0; i< samples; i++)
	std::cout<< times_end[i] << std::endl;

#ifdef VK_INSHADER_VALIDATION
   std::cout << out[0] <<  "/" << out[1] << " " << out[2] << " " << out[3] << " " << out[4] << std::endl; 
   std::cout << in[0] <<  "/" << in[1] << " " << in[2] << " " <<  in[3] << " " << in[4] << std::endl;  
#endif

//freeing the context
vk.freeResources();

return 0;
}