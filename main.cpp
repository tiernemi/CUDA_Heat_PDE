/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  Source for main function in radiator simulator.
 *
 *          Usage:  ./rad [OPTIONS]
 *
 *        Options:  -p : Number of iterations.
 *                  -n : Number of pipes in y-direction.
 *                  -m : Number of simulation sites for each pipe x-direction.
 *                  -t : If enabled show timing data.
 *                  -x : Number of threads in x-direction for GPU.
 *                  -y : Number of threads in y-direction for GPU.
 *
 *        Version:  1.0
 *        Created:  19/03/16 18:33:58
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <cstdio>
#include "stdlib.h"
#include "getopt.h"
#include "radiator.hpp"
#include <cmath>

int main(int argc, char *argv[]) {
	
	int pipeLength = 32 ;
	int numPipes = 32 ;
	int numIters = 10 ;
	int numThreadPerBlockX = 32 ;
	int numThreadPerBlockY = 32 ;
	bool timeFlag = false ;

	// ..............................COMMAND LINE ARGS............................. //
	
	while (1) {
		int choice = getopt(argc, argv, "p:n:m:tx:y:") ;
		if (choice == -1)
			break ;
		switch( choice ) {
			case 'p' :
				numIters = atoi(optarg) ;
				break ;
			case 'n' :
				numPipes = atoi(optarg) ;
				break ;
			case 'm' :
				pipeLength = atoi(optarg) ;
				break ;
			case 'x' :
				numThreadPerBlockX = atoi(optarg) ;
				break ;
			case 'y' :
				numThreadPerBlockY = atoi(optarg) ;
				break ;
			case 't' :
				timeFlag = true ;
				break ;
			default  :
				fprintf(stderr, "Unknown Command Line Argument\n") ;
				return EXIT_FAILURE ;
		}
	}
	
	int remainingArgs = argc - optind ;
	if (remainingArgs > 0) {
		fprintf(stderr, "USAGE ./rad [OPTIONS] \n") ;
		return EXIT_FAILURE ;
	}

	// ........................................................................... //
	
	std::vector<float> cpuTimes ;
	std::vector<float> gpuTimes ;

	Radiator<float> cpuRadiator(numPipes, pipeLength) ;
	Radiator<float> gpuRadiator(numPipes, pipeLength) ;
	cpuRadiator.simulateCPU(numIters, cpuTimes) ;
	gpuRadiator.simulateGPU(numIters, gpuTimes, numThreadPerBlockX, numThreadPerBlockY) ;

	if (numPipes < 40 && pipeLength < 40) {
		cpuRadiator.printRadiator(std::cout) ;
		std::cout << "" << std::endl ;
		gpuRadiator.printRadiator(std::cout) ;
	}

	// Check for deviations in values > 1E-5 //
	for (int i = 0 ; i < numPipes ; ++i) {
		for (int j = 0 ; j < pipeLength ; ++j) {
			if (std::abs(cpuRadiator.getData(i,j) - gpuRadiator.getData(i,j)) > 1E-5) {
				std::cout << "Deviation at site (" << i << "," << j << ") " << 
					cpuRadiator.getData(i,j) << " " << gpuRadiator.getData(i,j) << std::endl;
			}
		}
	}

	if (timeFlag) {
		std::cout << "CPU Compute Time : " << cpuTimes[0] << std::endl ;
		std::cout << "............................." << std::endl;
		std::cout << "GPU Allocation Time : " << gpuTimes[0]/1000.f << std::endl ;
		std::cout << "CPU->GPU Transfer Time : " << gpuTimes[1]/1000.f << std::endl ;
		std::cout << "GPU Compute Time : " << gpuTimes[2]/1000.f << std::endl ;
		std::cout << "GPU->CPU Transfer Time : " << gpuTimes[3]/1000.f << std::endl ;
		std::cout << "GPU Total Time : " << gpuTimes[4]/1000.f << std::endl ;
		std::cout << "Compute SpeedUp : " << (cpuTimes[0]*1000.f)/gpuTimes[2] << std::endl ;
		std::cout << "Total (Compute + Transfer) SpeedUp : " << (cpuTimes[0]*1000.f)/gpuTimes[4] << std::endl ;
	}

	return EXIT_SUCCESS ;
}
