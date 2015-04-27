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

        __global__ void filter_image(const uint8* input, uint8* output, const Chromosome* chromosome, const uint32 width, const uint32 height)
        {
            const uint32 x = blockIdx.x * blockDim.x + threadIdx.x;
            const uint32 y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x > 0 && x < (width - 1) && y > 0 && y < (height - 1))
            {
                const uint32 idx = y * width + x;

                uint8 kernel[9];
                get_3x3_kernel(x, y, kernel, input, width);

                output[idx] = eval_chromosome(chromosome, kernel, CGP_PARAM_ROWS, CGP_PARAM_COLS);
            }
        }

        __device__ uint8 eval_chromosome(const Chromosome* chromosome, uint8* kernel, const uint32 numRows, const uint32 numCols)
        {
            uint8 outputs[CGP_PARAM_TOTAL];
            memcpy(outputs, kernel, 9 * sizeof(uint8));

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
                    outputs[numRows * i + j + CGP_PARAM_INPUTS] = out;
                }
            }

            return out;
        }
    }    

} // namespace CGP
