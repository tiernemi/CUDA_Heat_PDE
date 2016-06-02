#ifndef CUDA_SIM_HPP_B80LXF1R
#define CUDA_SIM_HPP_B80LXF1R

/*
 * =====================================================================================
 *
 *       Filename:  cuda_sim.hpp
 *
 *    Description:  Definitions for CUDA code.
 *
 *        Version:  1.0
 *        Created:  20/03/16 11:03:30
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <vector>

template <typename T>
void cudaSimulateHeatFlow(int numIters, int numRows, int pipeLength, T * data, std::vector<float> & times, 
		int numThreadPerBlockX, int numThreadPerBlockY) ;

#endif /* end of include guard:  */
