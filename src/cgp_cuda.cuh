#ifndef H_CGP_CUDA
#define H_CGP_CUDA

#include "cuda_runtime.h"

namespace imcgp
{
    namespace cuda
    {        
        /**
         * @brief Gets a kernel (3x3 pixel area) around a given pixel
         * 
         * @param x         X-coordinate.
         * @param y         Y-coordinate.
         * @param kernel    Output kernel.
         * @param input     Input 8-bit image.
         * @param pitch     Image pitch.
         */
        __device__ void get_3x3_kernel(uint32 const& x, uint32 const& y, uint8* kernel, const uint8* input, uint32 const& pitch);

        /**
         * @brief Filters image based on a given chromosome. 
         * 
         * @param input         Input 8-bit image.
         * @param output        Filtered output image.
         * @param chromosome    Input chromosome which stores filtering parameters.
         * @param width         Image width.
         * @param height        Image height.
         */
        __global__ void filter_image(const uint8* input, uint8* output, const Chromosome* chromosome, const uint32 width, const uint32 height);

        /**
         * @brief Evaluates an input kernel (3x3 pixel area) with a given chromosome. 
         *
         * @param chromosome    Input chromosome which stores filtering parameters.
         * @param kernel        Input image kernel (3x3 pixel area).
         * @param numRows       Number of CGP rows.
         * @param numCols       Number of CGP columns.
         */
        __device__ uint8 eval_chromosome(const Chromosome* chromosome, uint8* kernel, const uint32 numRows, const uint32 numCols);
    }
}

#endif // H_CGP
