/*
 * =====================================================================================
 *
 *       Filename:  radiator.cpp
 *
 *    Description:  Source file for Radiator class.
 *
 *        Version:  1.0
 *        Created:  19/03/16 17:48:31
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "radiator.hpp"
#include "cuda_sim.hpp"
#include <algorithm>
#include <chrono>

/* 
 * ===  MEMBER FUNCTION CLASS : Radiator  ==============================================
 *         Name:  Radiator
 *    Arguments:  int numPipes - Number of pipes in simulation.
 *                int pipeLength - Length of each pipe.
 *  Description:  Constructor for Radiator class.
 * =====================================================================================
 */

template <typename T>
Radiator<T>::Radiator(int numPipes, int pipeLength) : numPipes{numPipes}, pipeLength{pipeLength}, trueLength{pipeLength+2} {
	this->data = new T[trueLength*numPipes]() ;
	initialiseBoundaries() ;
}		/* -----  end of member function Radiator  ----- */


/* 
 * ===  MEMBER FUNCTION CLASS : Radiator  ==============================================
 *         Name:  getNumPipes
 *      Returns:  Constant reference for pipelength.
 *  Description:  Constant getter for pipelength.
 * =====================================================================================
 */

template <typename T>
const int & Radiator<T>::getNumPipes() const {
	return numPipes ;
}

/* 
 * ===  MEMBER FUNCTION CLASS : Radiator  ==============================================
 *         Name:  getPipeLength
 *      Returns:  Constant reference for pipelength.
 *  Description:  Constant getter for pipelength.
 * =====================================================================================
 */

template <typename T>
const int & Radiator<T>::getPipeLength() const {
	return pipeLength ;
}

/* 
 * ===  MEMBER FUNCTION CLASS : Radiator  ==============================================
 *         Name:  getData
 *    Arguments:  int i - Row number.
 *                int j - Column number.
 *      Returns:  Reference to data stored in Radiator class.
 *  Description:  Getter for data.
 * =====================================================================================
 */

template <typename T>
T & Radiator<T>::getData(int i, int j) {
	return data[i*trueLength+j] ;
}		/* -----  end of member function getData  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Radiator  ==============================================
 *         Name:  getData
 *    Arguments:  int i - Row number.
 *                int j - Column number.
 *      Returns:  Constant reference to data stored in Radiator class.
 *  Description:  Constant getter for data.
 * =====================================================================================
 */

template <typename T>
const T & Radiator<T>::getData(int i, int j) const {
	return data[i*trueLength+j] ;
}		/* -----  end of member function getData  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Radiator  ==============================================
 *         Name:  simulateCPU
 *    Arguments:  int numIters - Number of iterations of the simulation to carry out. 
 *                std::vector<float> & times - Vector storing timing data.
 *  Description:  Runs a heat propogation simulation across each pipe with a flow thats
 *                biased towards the right. Periodic conditions exist at both ends. The
 *                first two columns remain constant.
 * =====================================================================================
 */

template <typename T>
void Radiator<T>::simulateCPU(int numIters, std::vector<float> & times) {
	T * newData = new T[trueLength*numPipes] ;
	std::copy_n(data, trueLength*numPipes, newData) ;

	std::chrono::time_point<std::chrono::system_clock> start, end ;
	start = std::chrono::system_clock::now() ;
					 
	// Run for numIters. //
	for (int i = 0 ; i < numIters ; ++i) {
		// Use nearest neighbour approximations across each row. //
		for (int j = 0 ; j < numPipes ; ++j) {
			// Rightwise Flow. //
			for (int k = 2 ; k < pipeLength ; ++k) {
				newData[j*trueLength+k] = (0.37*getData(j,k-2)) + (0.28*getData(j,k-1)) + 0.2*getData(j,k) + 
					(0.12*getData(j,k+1)) + (0.03*getData(j,k+2)) ;
			}
		}
		// Swap arrays. //
		T * temp = data ;
		data = newData ;
		newData = temp ;
	}

	end = std::chrono::system_clock::now() ;
	std::chrono::duration<float> elapsed_seconds = end-start ;
	times.push_back(elapsed_seconds.count()) ;

	delete [] newData ;
}		/* -----  end of member function simulate  ----- */


/* 
 * ===  MEMBER FUNCTION CLASS : Radiator  ==============================================
 *         Name:  simulateGPU
 *    Arguments:  int numIters - Number of iterations of the simulation to carry out. 
 *                std::vector<float> & times - Vector storing timing data.
 *                int numThreadPerBlockX - Number of threads in x dimension of 2D block.
 *                int numThreadPerBlockY - Number of threads in y dimension of 2D block.
 *  Description:  Runs a heat propogation simulation across each pipe with a flow thats
 *                biased towards the right. Periodic conditions exist at both ends. The
 *                first two columns remain constant.
 * =====================================================================================
 */

template <typename T>
void Radiator<T>::simulateGPU(int numIters, std::vector<float> & times, int numThreadPerBlockX, int numThreadPerBlockY) {
	cudaSimulateHeatFlow<T>(numIters, numPipes, pipeLength, data, times, numThreadPerBlockX, numThreadPerBlockY) ;
}		/* -----  end of member function simulate  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Radiator  =============================================
 *         Name:  initialiseBoundaries
 *  Description:  Initialises the central column of the radiator.
 * =====================================================================================
 */

template <typename T>
void Radiator<T>::initialiseBoundaries() {
	for (int i = 0 ; i < numPipes ; ++i) {
		getData(i,0) = 1.00*(T)(i+1)/(T)(numPipes) ;
		getData(i,1) = 0.75*(T)(i+1)/(T)(numPipes) ;
		getData(i,trueLength-2) = 1.00*(T)(i+1)/(T)(numPipes) ;
		getData(i,trueLength-1) = 0.75*(T)(i+1)/(T)(numPipes) ;
	}
}		/* -----  end of member function initialiseBoundaries  ----- */


/* 
 * ===  MEMBER FUNCTION CLASS : Radiator  ======================================
 *         Name:  printRadiator
 *    Arguments:  std::ostrea & out - The output stream.
 *  Description:  Outputs the radiator data to the output stream.
 * =====================================================================================
 */

template <typename T>
void Radiator<T>::printRadiator(std::ostream & out) const {
	for (int i = 0 ; i < numPipes ; ++i) {
		for (int j = 0 ; j < pipeLength-1 ; ++j) {
			out << getData(i,j) << "," ;
		}
		out << getData(i,pipeLength-1) << std::endl ;
	}
}		/* -----  end of member function printRadiator  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Radiator  =============================================
 *         Name:  ~Radiator
 *  Description:  Destructor for Radiator class.
 * =====================================================================================
 */

template <typename T>
Radiator<T>::~Radiator() {
	delete [] this->data ;
}		/* -----  end of member function ~Radiator  ----- */

template class Radiator<float> ;
