/*
Copyright © 2019
Roberto Cavicchioli, Nicola Capodieci

See LICENSE.txt
*/

#ifndef DEFINITIONS
#define DEFINITIONS

//parametrize type
#define TYPE float

#ifndef DATA_SIZE
#define DATA_SIZE 1
#endif

//In matrix dimension (W,H)
#define INW (DATA_SIZE*16)
#define INH (DATA_SIZE*16)

//Out matrix dimension (W,H)
#define OUTW (DATA_SIZE*16)
#define OUTH (DATA_SIZE*16)

//Weight matrix dimension (W,H)
#define WW (DATA_SIZE*16)
#define WH (DATA_SIZE*16)

//Activation functions
#define ACT_RELU 0
#define ACT_SIGMOID 1
#define ACT_TANH 2

//CUDA launch methodologies
#define BASELINE 0
#define DYNPAR 1
#define CUDAGRAPH 2

//Tile size
#define TILE_WIDTH 16

//passes
#define ITERATIONS 100

//# of submission
#define SAMPLES 10

//filter size
#define KERNEL_RADIUS 1 //3*3
//#define KERNEL_RADIUS 2 //5*5
//#define KERNEL_RADIUS 3 //7*7

#define KERNEL_SIZE (KERNEL_RADIUS*2+1)

//For random host PTRs init.
#define SEED 186

//#define CUDA_VALIDATION

//#define VK_INSHADER_VALIDATION

#endif