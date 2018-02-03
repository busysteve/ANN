/*
 ============================================================================
 Name        : ThrustPrime.cu
 Author      : Stephen Mathews
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>

template <typename T> __host__ __device__  T is_primo(const T &x)
{
	if( x == 2 ) return 1;

	for( int i=2; i <= ((x/2)+1); i++)
		if( (x % i) == 0 )
			return 0;

	return 1;
}

template <typename T> class ReciprocalFunctor {
	public:
	__host__ __device__ T operator()(const T &x) {
		return is_primo(x);
	}
};

template <typename T, class OpClass> T transformAndSumCPU(std::vector<T> data, OpClass op)
{
	std::vector<T> temp(data.size());
	std::transform(data.begin(), data.end(), temp.begin(), op);
	return std::accumulate(temp.begin(), temp.end(), (T)0);
}

template <typename T, class OpClass> T transformAndSumGPU(std::vector<T> data, OpClass op)
{
	thrust::device_vector<T> temp(data.begin(), data.end());
	thrust::transform(temp.begin(), temp.end(), temp.begin(), op);
	return thrust::reduce(temp.begin(), temp.end());
}

template<typename T> void initialize(std::vector<T> &data, T workStart, T workEnd)
{
	/* Initialize the vector */
	for (unsigned i = workStart; i <= workEnd; i++)
		data.push_back( i );
}

template<typename T> void doCompute(T workStart, T workEnd )
{
	std::vector<T> hostData;
	initialize(hostData, workStart, workEnd );

	T gpuResults = transformAndSumGPU(hostData, ReciprocalFunctor<T>());
	std::cout<<"transformAndSumGPU = "<<gpuResults<<std::endl << std::flush;

	T cpuResults = transformAndSumCPU(hostData, ReciprocalFunctor<T>());
	std::cout<<"transformAndSumCPU = "<<cpuResults<<std::endl << std::flush;
}

int main(int argc, char** argv)
{
	int max = 100000;
	if( argc > 1 )
		max = atoi( argv[1] );

	doCompute<int> ( 2, max );
	return 0;
}
