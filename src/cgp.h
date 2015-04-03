#ifndef H_CGP
#define H_CGP

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

///////////////////////////////////////////////////////////////
// Typedefs
///////////////////////////////////////////////////////////////

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

typedef char int8;
typedef short int16;
typedef int int32;

///////////////////////////////////////////////////////////////
// Defines
///////////////////////////////////////////////////////////////

/** @brief Number of CGP rows. */
#define CGP_PARAM_ROWS 6
/** @brief Number of CGP columns. */
#define CGP_PARAM_COLS 6
/** @brief Number of inputs. Should be 9 because of a 3x3 image kernel. */
#define CGP_PARAM_INPUTS 9
/** @brief Number of outputs. Should be 1 because of a single pixel output. */
#define CGP_PARAM_OUTPUTS 1
/** @brief Distance between columns, we can choose candidates from. */
#define CGP_PARAM_LBACK 1
/** @brief inputs + rows*cols + ouputs */
#define CGP_PARAM_TOTAL 46
/** @brief rows * cols * (func_inputs + 1) + outputs */
#define CGP_CHROMOSOME_SIZE 109

///////////////////////////////////////////////////////////////
// Enums
///////////////////////////////////////////////////////////////

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
	FUNC_ADD_SATUR, ///< ???
	FUNC_AVERAGE,	///< (in1 + in2) >> 1
	FUNC_MAX,		///< max(in1, in2)
	FUNC_MIN,		///< min(in1, in2)

	NUM_FUNCTIONS	///< total number of functions
};

///////////////////////////////////////////////////////////////
// Functions & Methods
///////////////////////////////////////////////////////////////

namespace imcgp
{
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
		SCORE,

		NUM_FITNESS_METHODS
	};

	/** @brief Calculates fitness.
	 *
	 * @param method Fitness method used.
	 * @param input Input image.
	 * @param reference Reference image to compare input to.
	 * @return Fitness value.
	 */
	float calc_fitness(FitnessMethod method, cv::Mat const& input, cv::Mat const& reference);

	/** @brief Fills an array with an image kernel. 
	 *
	 * @param kernel Input array.
	 * @param input Input image.
	 * @param x X-coordinate.
	 * @param y Y-coordinate.
	 * @return Void.
	 */
	void get_3x3_kernel(uint8* kernel, cv::Mat const& input, uint32 const& x, uint32 const& y);	

	/** @brief Evaluates a chromosome. 
	 *
	 * @brief chromosome Passed chromosome.
	 * @brief inputs An array of inputs. In our case an image kernel.
	 * @brief numRows Number of CGP rows.
	 * @brief numCols Number of CGP columns.
	 * @return Pixel value.
	 */
	uint8 eval_chromosome(Chromosome const& chromosome, uint8* inputs, uint32 const& numRows, uint32 const& numCols);

	/** @brief Evolves a population. 
	 *
	 * @param population Passed population.
	 * @param possibleValues Possible values to assign for every row.
	 * @param bestFilter Best chromosome in the given generation.
	 * @param numPopulation Number of chromosomes in population.
	 * @param numMutate Number of genes to mutate.
	 * @param numRows Number of CGP rows.
	 * @param numCols Number of CGP columns.
	 * @return Void.
	 */
	void evolve_population(Population& population, std::vector<uint32>* possibleValues, uint32 const& bestFilter, uint32 const& numPopulation, uint32 const& numMutate, uint32 const& numRows, uint32 const& numCols);

	/** @brief Fills a table of possible values for every row. 
	 *
	 * @param table Pointer to the table we're filling.
	 * @param numRows Number of CGP rows.
	 * @param numcols Number of CGP columns.
	 * @param lback L-back parameter - how far the connected rows can be from each other.
	 * @return Void.
	 */
	void find_possible_col_values(std::vector<uint32>* table, uint32 const& numRows, uint32 const& numCols, uint32 const& lback);

	/** @brief Creates an initial population. 
	 *
	 * @param population Passed population.
	 * @param possibleValues Possible values to assign for every row.
	 * @param numPopulation Number of chromosomes in population.	 
	 * @param numRows Number of CGP rows.
	 * @param numCols Number of CGP columns.
	 * @return Void.
	 */
	void create_init_population(Population& population, std::vector<uint32>* possibleValues, uint32 const& maxPopulation, uint32 const& numRows, uint32 const& numCols);

	/** @brief Mutates a chromosome. 
	 * 
	 * @param parent Parent chromosome to mutate from.
	 * @param possibleValues Possible values to assign for every row.
	 * @param numBits Number of bits to mutate.
	 * @param chromosomeLength Length of the chromosome.
	 * @param numRows Number of CGP rows.
	 * @param numCols Number of CGP columns.
	 * @return Mutated chromosome.
	 */
	Chromosome mutate(Chromosome const& parent, const std::vector<uint32>* possibleValues, const uint32 numBits, const uint32 chromosomeLength, const uint32 numRows, const uint32 numCols);

	class CGPWrapper
	{
		public:
			/** @brief Loads reference image. 
			 *
			 * @param filename Filename.
			 * @return Image loaded succesfully.
			 */
			bool load_ref_image(std::string const& filename);

			/** @brief Loads input image.
			 *
			 * @param filename Filename.
			 * @return Image loaded succesfully.
			 */
			bool load_image(std::string const& filename);

			/** @brief Displays an image after filtering. 
			 *
			 * @return Void.
			 */
			void display_filtered_image();

			/** @brief Displays the original image.
			 *
			 * @return Void.
			 */
			void display_original_image();

			/** @brief Saves the filtered image.
			 *
			 * @param filename Filename.
			 * @return Void.
			 */
			void save_filtered_image(std::string const& filename);

			/** @brief Saves the original image.
			 *
			 * @param filename Filename.
			 * @return Void.
			 */
			void save_original_image(std::string const& filename);

			/** @brief Runs the whole thing. 
			 *
			 * @param method Fitness method used to compare input and reference image.
			 * @param numRuns Number of runs.
			 * @param numGenerations Number of generations per single run.
			 * @param numPopulation Number of chromosomes in a generation.
			 * @param numMutate Number of mutated genes in a chromosome.
			 * @return Succesful run.
			 */
			bool run(FitnessMethod method, uint32 const& numRuns, uint32 const& numGenerations, uint32 const& numPopulation, uint32 const& numMutate);

		private:		
			/** @brief Input image. */
			cv::Mat _inputImage;
			
			/** @brief Reference image. */
			cv::Mat _referenceImage;

			/** @brief Output image. */
			cv::Mat _filteredImage;

			/** @brief Options. */
			uint32 _opt;			
	};

}

#endif // H_CGP
