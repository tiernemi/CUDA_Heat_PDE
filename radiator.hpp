#ifndef RADIATOR_HPP_PGB5JMOC
#define RADIATOR_HPP_PGB5JMOC

/*
 * =====================================================================================
 *
 *       Filename:  radiator.hpp
 *
 *    Description:  Header file for the cylindrical radiator class
 *
 *        Version:  1.0
 *        Created:  19/03/16 17:34:45
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

/* 
 * ===  CLASS  =========================================================================
 *         Name:  Radiator
 *       Fields:  T * data - Array storing temperature values at locations in the 
 *                radiator.
 *                int numPipes - Number of pipes in radiator.
 *                int pipeLength - Length of each pipe (simulation sites).
 *                int trueLength - The true length of pipe array (includes end boundary).
 *  Description:  Radiator object for which heatflow can be simulated. The geometry
 *                is cylindrical such that each individual pipe experiences periodic 
 *                boundaries. The flow is right biased.
 * =====================================================================================
 */

#include <iostream>
#include <vector>

template <typename T>

class Radiator {
 public:
	Radiator(int numPipes, int pipeLength) ;
	const int & getNumPipes() const ;
	const int & getPipeLength() const ;
	T & getData(int i, int j) ;
	const T & getData(int i, int j) const ;
	void simulateCPU(int numIters, std::vector<float> & times) ;
	void simulateGPU(int numIters, std::vector<float> & times, 
			int numThreadPerBlockX, int numThreadPerBlockY) ;
	void printRadiator(std::ostream & out) const ;
	virtual ~Radiator() ;
 private:
	T * data ;
	const int numPipes ;
	const int pipeLength ;
	const int trueLength ;
	void initialiseBoundaries() ;
} ;		/* -----  end of class Radiator  ----- */


#endif /* end of include guard: RADIATOR_HPP_PGB5JMOC */
