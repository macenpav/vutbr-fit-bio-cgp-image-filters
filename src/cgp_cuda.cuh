#ifndef H_CGP_CUDA
#define H_CGP_CUDA

#include "cuda_runtime.h"

namespace imcgp
{
    namespace cuda
    {
        __global__ void calc_fitness_add(float* fitness, FitnessMethod method, uint8* input, uint8* reference, const uint32 width, const uint32 height);

        __device__ void get_3x3_kernel(uint32 const& x, uint32 const& y, uint8* kernel, const uint8* input, uint32 const& pitch);

        __global__ void filter_image(const uint8* input, uint8* output, const Chromosome* chromosome, const uint32 width, const uint32 height);

        __device__ uint8 eval_chromosome(const Chromosome* chromosome, uint8* kernel, const uint32 numRows, const uint32 numCols);
    }
}

#endif // H_CGP
