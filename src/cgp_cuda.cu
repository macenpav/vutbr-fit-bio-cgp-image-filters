#include "cgp_enums.h"
#include "cgp_cuda.cuh"

namespace imcgp
{
    namespace cuda
    {        
        __device__ void get_3x3_kernel(uint32 const& x, uint32 const& y, uint8* kernel, const uint8* input, uint32 const& pitch)
        {
            const uint32 idx = y * pitch + x;

            kernel[0] = input[idx - pitch - 1];
            kernel[1] = input[idx - pitch];
            kernel[2] = input[idx - pitch + 1];

            kernel[3] = input[idx - 1];
            kernel[4] = input[idx];
            kernel[5] = input[idx + 1];

            kernel[6] = input[idx + pitch - 1];
            kernel[7] = input[idx + pitch];
            kernel[8] = input[idx + pitch + 1];
        }       

        __device__ void get_5x5_kernel(uint32 const& x, uint32 const& y, uint8* kernel, const uint8* input, uint32 const& pitch)
        {
            const uint32 idx = y * pitch + x;

            kernel[0] = input[idx - 2 * pitch - 2];
            kernel[1] = input[idx - 2 * pitch - 1];
            kernel[2] = input[idx - 2 * pitch];
            kernel[3] = input[idx - 2 * pitch + 1];
            kernel[4] = input[idx - 2 * pitch + 2];

            kernel[5] = input[idx - pitch - 2];
            kernel[6] = input[idx - pitch - 1];
            kernel[7] = input[idx - pitch];
            kernel[8] = input[idx - pitch + 1];
            kernel[9] = input[idx - pitch + 2];

            kernel[10] = input[idx - 2];
            kernel[11] = input[idx - 1];
            kernel[12] = input[idx];
            kernel[13] = input[idx + 1];
            kernel[14] = input[idx + 2];

            kernel[15] = input[idx + pitch - 2];
            kernel[16] = input[idx + pitch - 1];
            kernel[17] = input[idx + pitch];
            kernel[18] = input[idx + pitch + 1];
            kernel[19] = input[idx + pitch + 2];

            kernel[20] = input[idx + 2 * pitch - 2];
            kernel[21] = input[idx + 2 * pitch - 1];
            kernel[22] = input[idx + 2 * pitch];
            kernel[23] = input[idx + 2 * pitch + 1];
            kernel[24] = input[idx + 2 * pitch + 2];
        }

        __global__ void filter_image_3x3(const uint8* input, uint8* output, const Chromosome* chromosome, const uint32 width, const uint32 height)
        {
            const uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
            const uint32 y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x > 0 && x < (width - 1) && y > 0 && y < (height - 1))
            {
                const uint32 idx = y * width + x;

                uint8 kernel[9];
                get_3x3_kernel(x, y, kernel, input, width);

                output[idx] = eval_chromosome_3x3(chromosome, kernel, CGP_PARAM_ROWS, CGP_PARAM_COLS);
            }
        }

        __global__ void filter_image_5x5(const uint8* input, uint8* output, const Chromosome* chromosome, const uint32 width, const uint32 height)
        {
            const uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
            const uint32 y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x > 1 && x < (width - 2) && y > 1 && y < (height - 2))
            {
                const uint32 idx = y * width + x;

                uint8 kernel[25];
                get_5x5_kernel(x, y, kernel, input, width);

                output[idx] = eval_chromosome_5x5(chromosome, kernel, CGP_PARAM_ROWS, CGP_PARAM_COLS);
            }
        }

        __device__ uint8 eval_func(uint32 const& func, uint32 const& in1, uint32 const& in2, uint8* outputs, uint32 const& outputIdx)
        {
            uint8 out;
            switch (func)
            {
                case FUNC_CONST: out = 255; break;
                case FUNC_IDENTITY: out = in1; break;
                case FUNC_INVERT: out = 255 - in1; break;
                case FUNC_OR: out = in1 | in2; break;
                case FUNC_AND: out = in1 & in2; break;
                case FUNC_NAND: out = ~(in1 & in2); break;
                case FUNC_XOR: out = in1 ^ in2; break;
                case FUNC_SHR1: out = in1 >> 1; break;
                case FUNC_SHR2: out = in1 >> 2; break;
                case FUNC_SWAP: out = ((in1 & 0x0F) << 4) | (in2 & 0x0F); break;
                case FUNC_ADD:
                {
                    if (static_cast<uint32>(in1)+static_cast<uint32>(in2) > 255)
                        out = 255;
                    else
                        out = in1 + in2;
                    break;
                }
                case FUNC_AVERAGE:
                {
                    out = static_cast<uint8>((static_cast<uint32>(in1)+static_cast<uint32>(in2)) >> 1);
                    break;
                }
                case FUNC_MAX:
                {
                    if (in1 > in2)
                        out = in1;
                    else
                        out = in2;

                    break;
                }
                case FUNC_MIN:
                {
                    if (in1 < in2)
                        out = in1;
                    else
                        out = in2;

                    break;
                }
                case FUNC_SHL1: out = in1 << 1; break;
                case FUNC_SHL2: out = in1 << 2; break;

                default: out = 255;
            }
            outputs[outputIdx] = out;

            return out;
        }        

        __device__ uint8 eval_chromosome_3x3(const Chromosome* chromosome, uint8* kernel, const uint32 numRows, const uint32 numCols)
        {
            uint8 outputs[CGP_PARAM_TOTAL_3X3];
            memcpy(outputs, kernel, CGP_PARAM_INPUTS_3X3 * sizeof(uint8));

            uint32 in1, in2, func;

            uint32 v = 0;

            uint8 out;
            for (uint32 i = 0; i < numCols; i++)
            {
                for (uint32 j = 0; j < numRows; j++)
                {
                    in1 = outputs[chromosome->val[v++]];
                    in2 = outputs[chromosome->val[v++]];
                    func = chromosome->val[v++];

                    out = eval_func(func, in1, in2, outputs, numRows * i + j + CGP_PARAM_INPUTS_3X3);
                }
            }

            return out;
        }

        __device__ uint8 eval_chromosome_5x5(const Chromosome* chromosome, uint8* kernel, const uint32 numRows, const uint32 numCols)
        {
            uint8 outputs[CGP_PARAM_TOTAL_5X5];
            memcpy(outputs, kernel, CGP_PARAM_INPUTS_5X5 * sizeof(uint8));

            uint32 in1, in2, func;

            uint32 v = 0;

            uint8 out;
            for (uint32 i = 0; i < numCols; i++)
            {
                for (uint32 j = 0; j < numRows; j++)
                {
                    in1 = outputs[chromosome->val[v++]];
                    in2 = outputs[chromosome->val[v++]];
                    func = chromosome->val[v++];

                    out = eval_func(func, in1, in2, outputs, numRows * i + j + CGP_PARAM_INPUTS_5X5);
                }
            }

            return out;
        }
    }    

} // namespace CGP
