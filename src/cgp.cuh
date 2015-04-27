/**
 * @file cgp.cuh
 * @brief CGP implementation.
 *
 * The CGP wrapper and the CPU implementation are held here.
 *
 * @author Pavel Macenauer <macenauer.p@gmail.com>
 */
#ifndef H_CGP
#define H_CGP

#include <string>
#include <vector>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "cuda_runtime.h"
#include "cgp_enums.h"

// Uncomment this to show some additional debugging outputs
// #define DEBUG ///< debug mode

/** @brief GPU assert call wrapper. */
#define GPU_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/** @brief Assert function for GPU calls. */
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPU assert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;        
        if (abort) exit(code);
    }
}

/// Image CGP wrapper.
namespace imcgp
{

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

    /** @brief Fills an array with an image kernel.
     *
     * @param kernel Input array.
     * @param input Input image.
     * @param x X-coordinate.
     * @param y Y-coordinate.
     * @return Void.
     */   
    void get_5x5_kernel(uint8* kernel, cv::Mat const& input, uint32 const& x, uint32 const& y);

	/** @brief Evaluates a chromosome. 
	 *
	 * @brief chromosome Passed chromosome.
	 * @brief inputs An array of inputs. In our case an image kernel.
	 * @brief numRows Number of CGP rows.
	 * @brief numCols Number of CGP columns.
	 * @return Pixel value.
	 */
    uint8 eval_chromosome(Chromosome const& chromosome, uint8* inputs, uint32 const& numRows, uint32 const& numCols, uint32 const& numInputs);

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
    void evolve_population(Population& population, std::vector<uint32>* possibleValues, uint32 const& bestFilter, uint32 const& numPopulation, uint32 const& numMutate, uint32 const& numRows, uint32 const& numCols, uint32 const& numInputs);

	/** @brief Fills a table of possible values for every row. 
	 *
	 * @param table Pointer to the table we're filling.
	 * @param numRows Number of CGP rows.
	 * @param numcols Number of CGP columns.
	 * @param lback L-back parameter - how far the connected rows can be from each other.
	 * @return Void.
	 */
    void find_possible_col_values(std::vector<uint32>* table, uint32 const& numRows, uint32 const& numCols, uint32 const& lback, uint32 const& numInputs);

	/** @brief Creates an initial population. 
	 *
	 * @param population Passed population.
	 * @param possibleValues Possible values to assign for every row.
	 * @param numPopulation Number of chromosomes in population.	 
	 * @param numRows Number of CGP rows.
	 * @param numCols Number of CGP columns.
	 * @return Void.
	 */
    void create_init_population(Population& population, std::vector<uint32>* possibleValues, uint32 const& maxPopulation, uint32 const& numRows, uint32 const& numCols, uint32 const& numInputs);

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
    Chromosome mutate(Chromosome parent, const std::vector<uint32>* possibleValues, uint32 const& numBits, uint32 const& chromosomeLength, uint32 const& numRows, uint32 const& numCols, uint32 const& numInputs);

    /** @brief A class to simplify CGP handling. */
	class CGPWrapper
	{
		public:
            /** @brief Loads an image. 
             *
             * @param filename      Name of the file.
             * @param type          Type of the file to load (ORIGINAL_IMAGE, REFERENCE_IMAGE)
             * @return Successfully loaded the file.
             */
			bool load_image(std::string const& filename, ImageType type);

            /** @brief Displays an image. 
             * 
             * @param type Type of the file to load (ORIGINAL_IMAGE, REFERENCE_IMAGE, FILTERED_IMAGE) 
             */
			void display_image(ImageType type);

            /** @brief Saves an image to a file.
             *
             * @param filename  Name of the file.
             * @param type      Type of the file to load (ORIGINAL_IMAGE, REFERENCE_IMAGE, FILTERED_IMAGE)            
             */
			void save_image(std::string const& filename, ImageType type);

            /** @brief Passes run options.
             *
             * @param opts Passed options (a bit array).
             */
			void set_options(uint32 const& opts);

            /** @brief Writes run stats to a file.
             *
             * @param filename Name of the file.
             */
			void write_stats(std::string const& filename);

			/** @brief Runs the whole thing. 
			 *
			 * @param method Fitness method used to compare input and reference image.
			 * @param numRuns Number of runs.
			 * @param numGenerations Number of generations per single run.
			 * @param numPopulation Number of chromosomes in a generation.
			 * @param numMutate Number of mutated genes in a chromosome.
			 * @return Succesful run.
			 */
			bool run(FitnessMethod method, uint32 const& numRuns, uint32 const& numGenerations, uint32 const& numPopulation, uint32 const& numMutate, uint32 const& numInputs);

		private:		
			/** @brief Input image. */
			cv::Mat _inputImage;
			
			/** @brief Reference image. */
			cv::Mat _referenceImage;

			/** @brief Output image. */
			cv::Mat _filteredImage;
            
            /** @brief Input image in GPU memory. */
            uint8* _cudaInputImage;

            /** @brief Output image in GPU memory. */
            uint8* _cudaFilteredImage;            

			/** @brief Options. */
			uint32 _options;		

            /** @brief Run statistics */
			Statistics _stats;
	};

}

#endif // H_CGP
