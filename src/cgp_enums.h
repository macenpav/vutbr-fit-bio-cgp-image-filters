/**
 * @file cgp_enums.h
 * @brief CGP structures.
 *
 * Used structures are held here.
 *
 * @author Pavel Macenauer <macenauer.p@gmail.com>
 */

#ifndef H_CGP_ENUMS
#define H_CGP_ENUMS

#include <vector>
#include <chrono>

///////////////////////////////////////////////////////////////
// Typedefs
///////////////////////////////////////////////////////////////

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

typedef char int8;
typedef short int16;
typedef int int32;

typedef std::chrono::duration<double, std::ratio<1, 1000000000>> Ticks;

///////////////////////////////////////////////////////////////
// Defines
///////////////////////////////////////////////////////////////

/** @brief Number of CGP rows. */
#define CGP_PARAM_ROWS 6
/** @brief Number of CGP columns. */
#define CGP_PARAM_COLS 6
/** @brief Number of inputs. Should be 9 because of a 3x3 image kernel. */
#define CGP_PARAM_INPUTS_3X3 9
/** @brief Number of inputs. Should be 25 because of a 5x5 image kernel. */
#define CGP_PARAM_INPUTS_5X5 25
/** @brief Number of outputs. Should be 1 because of a single pixel output. */
#define CGP_PARAM_OUTPUTS 1
/** @brief Distance between columns, we can choose candidates from. */
#define CGP_PARAM_LBACK 1
/** @brief inputs + rows*cols + ouputs */
#define CGP_PARAM_TOTAL_3X3 46
/** @brief inputs + rows*cols + ouputs */
#define CGP_PARAM_TOTAL_5X5 62
/** @brief rows * cols * (func_inputs + 1) + outputs */
#define CGP_CHROMOSOME_SIZE 109

///////////////////////////////////////////////////////////////
// Functions & Methods
///////////////////////////////////////////////////////////////

/// Image CGP wrapper.
namespace imcgp 
{
    ///////////////////////////////////////////////////////////////
    // Enums
    ///////////////////////////////////////////////////////////////

    /** @brief Run options. */
    enum Options
    {
        OPT_VERBOSE             = 0x00000001,
        OPT_MEASURE             = 0x00000002,
        OPT_CUDA_ACCELERATION   = 0x00000004,
        OPT_OUTPUT_CSV          = 0x00000008
    };

    /** @brief Functions used to design a filter. */
    enum Function
    {
        FUNC_CONST,		///< 255
        FUNC_IDENTITY,	///< in1
        FUNC_INVERT,	///< 255 - in1
        FUNC_OR,		///< in1 | in2
        FUNC_AND,		///< in1 & in2
        FUNC_NAND,		///< ~(in1 & in2)
        FUNC_XOR,		///< in1 ^ in2
        FUNC_SHR1,		///< in1 >> 1
        FUNC_SHR2,		///< in1 >> 2
        FUNC_SWAP,		///< in2
        FUNC_ADD,		///< max(in1 + in2, 255)        
        FUNC_AVERAGE,	///< (in1 + in2) >> 1
        FUNC_MAX,		///< max(in1, in2)
        FUNC_MIN,		///< min(in1, in2)
        FUNC_SHL1,		///< in1 << 1
        FUNC_SHL2,		///< in1 << 2

        NUM_FUNCTIONS	///< total number of functions
    };

    /** @brief Types of images used. */
    enum ImageType
    {
        ORIGINAL_IMAGE,
        REFERENCE_IMAGE,
        FILTERED_IMAGE,

        MAX_IMAGE_TYPES
    };

    /** @brief Represents a chromosome.
     *
     * mod 0, 1 values represent inputs
     * mod 2 values represent function
     * last value represents output
     */
    struct Chromosome
    {
        uint32 val[CGP_CHROMOSOME_SIZE];
    };

    /** @brief An array of chromosomes, representing a population. */
    typedef std::vector<Chromosome> Population;

    const float ERROR_FITNESS = -1.f;	///< Fitness error.
    const int32 ERROR_FILTER = -1;		///< Filter error.

    /** @brief Different methods how to calculate fitness. */
    enum FitnessMethod
    {
        MDPP,	///< Mean difference per pixel
        PSNR,	///< Peak signal-to-noise ratio
        MSE,    ///< Mean square error

        NUM_FITNESS_METHODS
    };

    /** @brief A structure to simplify saving run stats. */
    struct Statistics
    {
        Chromosome best_filter;
        float fitness;
        double total_time, average_gen_time, init_time;
        Population initial_population;
        uint32 num_generations, num_genes_mutated, population_size, num_inputs;
        FitnessMethod method;
    };
}

#endif // H_CGP_ENUMS
