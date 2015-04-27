/**
 * @file cgp_cuda.cuh
 * @brief CUDA accelerated CGP library.
 *
 * Methods used by the CGP with CUDA acceleration.
 *
 * @author Pavel Macenauer <macenauer.p@gmail.com>
 */

#ifndef H_CGP_CUDA
#define H_CGP_CUDA

#include "cuda_runtime.h"


/// Image CGP wrapper.
namespace imcgp
{
    /// Functions accelerated on GPU.
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
         * @brief Gets a kernel (3x3 pixel area) around a given pixel
         *
         * @param x         X-coordinate.
         * @param y         Y-coordinate.
         * @param kernel    Output kernel.
         * @param input     Input 8-bit image.
         * @param pitch     Image pitch.
         */
        __device__ void get_5x5_kernel(uint32 const& x, uint32 const& y, uint8* kernel, const uint8* input, uint32 const& pitch);

        /**
         * @brief Filters image based on a given chromosome. Uses 3x3 filtering window.
         * 
         * @param input         Input 8-bit image.
         * @param output        Filtered output image.
         * @param chromosome    Input chromosome which stores filtering parameters.
         * @param width         Image width.
         * @param height        Image height.
         */
        __global__ void filter_image_3x3(const uint8* input, uint8* output, const Chromosome* chromosome, const uint32 width, const uint32 height);

        /**
         * @brief Filters image based on a given chromosome. Uses 5x5 filtering window.
         *
         * @param input         Input 8-bit image.
         * @param output        Filtered output image.
         * @param chromosome    Input chromosome which stores filtering parameters.
         * @param width         Image width.
         * @param height        Image height.
         */
        __global__ void filter_image_5x5(const uint8* input, uint8* output, const Chromosome* chromosome, const uint32 width, const uint32 height);

        /**
         * @brief Evaluates an input kernel (3x3 pixel area) with a given chromosome. 
         *
         * @param chromosome    Input chromosome which stores filtering parameters.
         * @param kernel        Input image kernel (3x3 pixel area).
         * @param numRows       Number of CGP rows.
         * @param numCols       Number of CGP columns.
         */
        __device__ uint8 eval_chromosome_3x3(const Chromosome* chromosome, uint8* kernel, const uint32 numRows, const uint32 numCols);

        /**
         * @brief Evaluates an input kernel (3x3 pixel area) with a given chromosome.
         *
         * @param chromosome    Input chromosome which stores filtering parameters.
         * @param kernel        Input image kernel (3x3 pixel area).
         * @param numRows       Number of CGP rows.
         * @param numCols       Number of CGP columns.
         */
        __device__ uint8 eval_chromosome_5x5(const Chromosome* chromosome, uint8* kernel, const uint32 numRows, const uint32 numCols);

        /**
         * @brief Evaluates given inputs and a function.
         *
         * @param func       Given function.
         * @param in1        First input.
         * @param in2        Second input.
         * @param outputs    Output intermediate array.
         * @param outputIdx  Index of the output
         */
        __device__ uint8 eval_func(uint32 const& func, uint32 const& in1, uint32 const& in2, uint8* outputs, uint32 const& outputIdx);
    }
}

#endif // H_CGP
