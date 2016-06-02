/*
 * =====================================================================================
 *
 *       Filename:  cuda_sim.cu
 *
 *    Description:  Source file for Heat flow simulation for radiator class on the GPU.
 *
 *        Version:  1.0
 *        Created:  19/03/16 17:48:31
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "cuda_sim.hpp"
#include "stdio.h"

// Two alternating surface buffers to eliminate expensive copying. //
surface<void, 2> gpuSurfBuf1 ; // Surface Buffer 
surface<void, 2> gpuSurfBuf2 ; // Surface Buffer


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  simulateRowSegment21(
 *    Arguments:  int pipeLength - Length of individual pipe.
 *                int numPipes - Number of pipes.
 *  Description:  Performs nearest neighbour approximations of pipe with periodic
 *                boundaries. Reads from buffer 2 and writes to buffer 1. Shared memory
 *                reduces the cost of accessing neighbours.
 * =====================================================================================
 */

template <typename T>
__global__ void simulateRowSegment21(int pipeLength, int numPipes) {

	int threadID = threadIdx.x + blockDim.x*blockIdx.x ;	
	int rowID = blockIdx.y*blockDim.y + threadIdx.y ; 
	extern __shared__ T oldRowData[] ;
	int elementID = threadIdx.x + 2 ;
	int globalID = threadID + 2 ;
	int rowOffset = 0 ;
	if (blockIdx.x == (pipeLength-2)/blockDim.x) {
		rowOffset = threadIdx.y*(pipeLength-blockDim.x*blockIdx.x+2) ;
	} else {
		rowOffset = threadIdx.y*(blockDim.x+4) ;
	}
	
	if (globalID < pipeLength && rowID < numPipes) {
		if (threadIdx.x == 0) {
			surf2Dread(&oldRowData[rowOffset+0], gpuSurfBuf1, (threadID)*sizeof(T), rowID) ;
			surf2Dread(&oldRowData[rowOffset+1], gpuSurfBuf1, (threadID+1)*sizeof(T), rowID) ;
		}
		if (threadIdx.x == blockDim.x-1 || globalID == pipeLength-1) {
			surf2Dread(&oldRowData[rowOffset+elementID+1], gpuSurfBuf1, (globalID+1)*sizeof(T), rowID) ; 
			surf2Dread(&oldRowData[rowOffset+elementID+2], gpuSurfBuf1, (globalID+2)*sizeof(T), rowID) ; 
		}
		surf2Dread(&oldRowData[rowOffset+elementID], gpuSurfBuf1, (globalID)*sizeof(T), rowID) ; 
		__syncthreads() ;

		T newVal =  (0.37*oldRowData[rowOffset+elementID-2]) + (0.28*oldRowData[rowOffset+elementID-1]) + 0.2*oldRowData[rowOffset+elementID] + 
			(0.12*oldRowData[rowOffset+elementID+1]) + (0.03*oldRowData[rowOffset+elementID+2]) ;

		surf2Dwrite(newVal, gpuSurfBuf2, globalID*sizeof(T), rowID); 
	} else {
		__syncthreads() ;
	}
}
template __global__ void simulateRowSegment21<float>(int, int) ;

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  simulateRowSegment12(
 *    Arguments:  int pipeLength - Length of individual pipe.
 *                int numPipes - Number of pipes.
 *  Description:  Performs nearest neighbour approximations of pipe with periodic
 *                boundaries. Reads from buffer 1 and writes to buffer 2. Shared memory
 *                reduces the cost of accessing neighbours.
 * =====================================================================================
 */

template <typename T>
__global__ void simulateRowSegment12(int pipeLength, int numPipes) {

	int threadID = threadIdx.x + blockDim.x*blockIdx.x ;	
	int rowID = blockIdx.y*blockDim.y + threadIdx.y ; 
	extern __shared__ T oldRowData[] ;
	int elementID = threadIdx.x + 2 ;
	int globalID = threadID + 2 ;
	int rowOffset = 0 ;
	if (blockIdx.x == (pipeLength-2)/blockDim.x) {
		rowOffset = threadIdx.y*(pipeLength-blockDim.x*blockIdx.x+2) ;
	} else {
		rowOffset = threadIdx.y*(blockDim.x+4) ;
	}

	if (globalID < pipeLength && rowID < numPipes) {
		if (threadIdx.x == 0) {
			surf2Dread(&oldRowData[rowOffset+0], gpuSurfBuf2, (threadID)*sizeof(T), rowID) ;
			surf2Dread(&oldRowData[rowOffset+1], gpuSurfBuf2, (threadID+1)*sizeof(T), rowID) ;
		}
		if (threadIdx.x == blockDim.x-1 || globalID == pipeLength-1) {
			surf2Dread(&oldRowData[rowOffset+elementID+1], gpuSurfBuf2, (globalID+1)*sizeof(T), rowID) ; 
			surf2Dread(&oldRowData[rowOffset+elementID+2], gpuSurfBuf2, (globalID+2)*sizeof(T), rowID) ; 
		}
		surf2Dread(&oldRowData[rowOffset+elementID], gpuSurfBuf2, (globalID)*sizeof(T), rowID) ; 
		__syncthreads() ;

		T newVal = (0.37*oldRowData[rowOffset+elementID-2]) + (0.28*oldRowData[rowOffset+elementID-1]) + 0.2*oldRowData[rowOffset+elementID] + 
			(0.12*oldRowData[rowOffset+elementID+1]) + (0.03*oldRowData[rowOffset+elementID+2]) ;

		surf2Dwrite(newVal, gpuSurfBuf1, globalID*sizeof(T), rowID); 
	} else {
		__syncthreads() ;
	}
}
template __global__ void simulateRowSegment12<float>(int, int) ;

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  copy_kernel
 *    Arguments:  int width - Width of array to be copied.
 *                int height - Height of array to be copied.
 *  Description:  Copies surface data from one surface to another.
 * =====================================================================================
 */

template <typename T>
__global__ void copyKernel(int width, int height) { 
	// Calculate surface coordinates 
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 
	if (x < width && y < height) { 
		T data; 
		// Read from input surface 
		surf2Dread(&data, gpuSurfBuf2, x * sizeof(T), y); 
		// Write to output surface 
		surf2Dwrite(data, gpuSurfBuf1, x * sizeof(T), y); 
	} 
}
template __global__ void copyKernel<float>(int, int) ;


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  transformTextureToGlobal
 *    Arguments:  T * gpu_odata - Pointer to global array on GPU.
 *                int width - Width of array.
 *                int height - Height of array.
 *  Description:  Copies surface to global memory.
 * =====================================================================================
 */

template <typename T>
__global__ void transformTextureToGlobal (T * gpu_odata, int width, int height) {
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
				
	if ( (x < width) && (y < height) ) {
		surf2Dread(&(gpu_odata[y*width+x]), gpuSurfBuf1, x*sizeof(T) , y); 
	}
}
template __global__ void transformTextureToGlobal<float>(float *, int, int) ;


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  cudaSimulateHeatFlow
 *    Arguments:  int numIters - Number of iterations of the simulation.
 *                int numPipes - Number of independent pipes in simulation.
 *                int pipeLength - Number of simulation sites for each pipe.
 *                std::vector<float> & times - Vector storing timing data.
 *                int numThreadPerBlockX - Number of threads in x dimension of 2D block.
 *                int numThreadPerBlockY - Number of threads in y dimension of 2D block.
 *  Description:  Copies simulation initial conditions to GPU, simulates each pipe over
 *                a 2D grid using shared and surface memory and then copies the resulting
 *                data back to the CPU.
 * =====================================================================================
 */

template <typename T>
void cudaSimulateHeatFlow(int numIters, int numPipes, int pipeLength, T * data, std::vector<float> & times, 
		int numThreadPerBlockX, int numThreadPerBlockY) {

	int totSize = numPipes*(pipeLength+2) ;
	// Allocation Timing. //
	float elapsedTime ;
	cudaEvent_t start, finish ;
	cudaEventCreate(&start) ;
	cudaEventCreate(&finish) ;
	cudaEventRecord(start, 0) ;

	// Allocate memory for Array. //
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>() ;
	cudaArray * bufferArray1 ; 
	cudaArray * bufferArray2 ; 
	T * gpuDataGlobal ;
	cudaMalloc( (void**) &gpuDataGlobal, totSize*sizeof(T)) ;
	cudaMallocArray(&bufferArray1, &channelDesc, pipeLength+2, numPipes, cudaArraySurfaceLoadStore) ; 
	cudaMallocArray(&bufferArray2, &channelDesc, pipeLength+2 , numPipes, cudaArraySurfaceLoadStore) ; 
	
	cudaEventRecord(finish, 0) ;
	cudaEventSynchronize(finish) ;
	// Timing for transfer CPU->GPU. //
	cudaEventElapsedTime(&elapsedTime, start, finish);
	times.push_back(elapsedTime) ;

	cudaEventRecord(start, 0) ;
    cudaMemcpyToArray(bufferArray2, 0, 0, data, totSize*sizeof(T), cudaMemcpyHostToDevice) ; 
    cudaMemcpyToArray(bufferArray1, 0, 0, data, totSize*sizeof(T), cudaMemcpyHostToDevice) ; 
	cudaBindSurfaceToArray(gpuSurfBuf2, bufferArray1) ; 
	cudaBindSurfaceToArray(gpuSurfBuf1, bufferArray2) ; 

	dim3 cpDimBlock(32, 32, 1);
	dim3 cpDimGrid((pipeLength+2)/cpDimBlock.x + ((!(pipeLength+2)%cpDimBlock.x)?0:1), (numPipes)/ cpDimBlock.y + (!(numPipes%cpDimBlock.y)?0:1), 1);
	dim3 dimBlock(numThreadPerBlockX, numThreadPerBlockY, 1) ;
	dim3 dimGrid(pipeLength/dimBlock.x + (!(pipeLength%dimBlock.x)?0:1), (numPipes)/dimBlock.y + (!(numPipes%dimBlock.y)?0:1)) ;

	cudaEventRecord(finish, 0) ;
	cudaEventSynchronize(finish) ;
	// Timing for Compute on GPU . //
	cudaEventElapsedTime(&elapsedTime, start, finish);
	times.push_back(elapsedTime) ;
	cudaEventRecord(start, 0) ;

	
	for (int j = 0 ; j < numIters/2 ; ++j) {
		simulateRowSegment21<T><<<dimGrid,dimBlock,(numThreadPerBlockX+4)*(numThreadPerBlockY)*sizeof(T)>>>(pipeLength, numPipes) ;
		simulateRowSegment12<T><<<dimGrid,dimBlock,(numThreadPerBlockX+4)*(numThreadPerBlockY)*sizeof(T)>>>(pipeLength, numPipes) ;	
	} if (numIters % 2 == 1) {
		simulateRowSegment21<T><<<dimGrid,dimBlock,(numThreadPerBlockX+4)*(numThreadPerBlockY)*sizeof(T)>>>(pipeLength, numPipes) ;
		copyKernel<T><<<cpDimGrid, cpDimBlock>>>(pipeLength+2, numPipes) ;
	}

	cudaEventRecord(finish, 0) ;
	cudaEventSynchronize(finish) ;
	// Timing for transfer GPU -> CPU //
	cudaEventElapsedTime(&elapsedTime, start, finish);
	times.push_back(elapsedTime) ;
	cudaEventRecord(start, 0) ;

	transformTextureToGlobal<T><<<cpDimGrid, cpDimBlock>>>(gpuDataGlobal,pipeLength+2,numPipes) ;
	cudaError err = cudaGetLastError() ;

	cudaEventRecord(finish, 0) ;
	cudaEventSynchronize(finish) ;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	times.push_back(elapsedTime) ;

	float totTime = 0.f ;
	for (int i = 0 ; i < times.size() ; ++i) {
		totTime += times[i] ;
	}
	times.push_back(totTime) ;

	cudaMemcpy(data, gpuDataGlobal, totSize*sizeof(T), cudaMemcpyDeviceToHost) ;

    cudaFreeArray(bufferArray2); 
	cudaFreeArray(bufferArray1); 
	cudaFree(gpuDataGlobal) ;
	cudaDeviceReset() ;
}

// Explicit instantiation of float function. //
template void cudaSimulateHeatFlow<float>(int numIters, int numPipes, int pipeLength, float * data, std::vector<float> & times, 
		int numThreadPerBlockX, int numThreadPerBlock) ;
