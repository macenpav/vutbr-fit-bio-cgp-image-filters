#include <iostream>
#include <cmath>
#include <vector>
#include <ctime>

#include "cgp.h"

namespace imcgp
{ 
	float calc_fitness(FitnessMethod method, cv::Mat const& input, cv::Mat const& reference)
	{
		if (input.rows != reference.rows || input.cols != reference.cols)
		{
			std::cerr << "Input image and reference image dimensions are different." << std::endl;
			return ERROR_FITNESS;
		}

		float fitness = ERROR_FITNESS;
		switch (method)
		{
			case MDPP:
			{
				fitness = 0.f;
				for (uint32 y = 0; y < input.rows; ++y)
				{
					for (uint32 x = 0; x < input.cols; ++x)
						fitness += std::abs(static_cast<float>(input.at<uint8>(y, x)) - static_cast<float>(reference.at<uint8>(y, x)));
				}
				fitness /= static_cast<float>(input.cols * input.rows);
				break;
			}
			case PSNR:
			{
				float tmp = 0.f;
				for (uint32 y = 0; y < input.rows; ++y)
				{
					for (uint32 x = 0; x < input.cols; ++x)
					{
						float a = static_cast<float>(input.at<uint8>(y, x)) - static_cast<float>(reference.at<uint8>(y, x));
						tmp += (a * a);
					}
				}
				tmp /= (input.cols * input.rows);
				fitness = 10.f * std::log10(255.f * 255.f / tmp);
				break;
			}
			case SCORE:
			// TODO: implement score and other methods
			default:			
				break;			
		}

		return fitness;
	}

	void get_3x3_kernel(uint8* kernel, cv::Mat const& input, uint32 const& x, uint32 const& y)
	{
		kernel[0] = input.at<uint8>(y - 1, x - 1);
		kernel[1] = input.at<uint8>(y - 1, x);
		kernel[2] = input.at<uint8>(y - 1, x + 1);

		kernel[3] = input.at<uint8>(y, x - 1);
		kernel[4] = input.at<uint8>(y, x);
		kernel[5] = input.at<uint8>(y, x + 1);

		kernel[6] = input.at<uint8>(y + 1, x - 1);
		kernel[7] = input.at<uint8>(y + 1, x);
		kernel[8] = input.at<uint8>(y + 1, x + 1);
	}

	uint8 eval_chromosome(Chromosome const& chromosome, uint8* inputs, uint32 const& numRows, uint32 const& numCols)
	{
		uint8 outputs[CGP_PARAM_TOTAL];
		memcpy(outputs, inputs, 9 * sizeof(uint8));
		
		uint32 in1, in2, func;

		uint32 v = 0;

		uint8 out;
		for (uint32 i = 0; i < numCols; i++)
		{
			for (uint32 j = 0; j < numRows; j++)
			{
				in1 = outputs[chromosome.val[v++]];
				in2 = outputs[chromosome.val[v++]];
				func = chromosome.val[v++];

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
					case FUNC_ADD_SATUR:
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
					case FUNC_MAX: out = std::max(in1, in2); break;
					case FUNC_MIN: out = std::min(in1, in2); break;
					default: out = 255;
				}
				outputs[numRows * i + j + CGP_PARAM_INPUTS] = out;
			}
		}

		return out;
	}

	void evolve_population(Population& population, std::vector<uint32>* possibleValues, uint32 const& bestFilter, uint32 const& numPopulation, uint32 const& numMutate, uint32 const& numRows, uint32 const& numCols)
	{
		Chromosome parent = population[bestFilter];

		population[0] = parent;
		for (uint32 ch = 1; ch < numPopulation; ++ch)
		{
			population[ch] = mutate(parent, possibleValues, numMutate, CGP_PARAM_TOTAL, numRows, numCols);
		}
	}

	void find_possible_col_values(std::vector<uint32>* table, uint32 const& numRows, uint32 const& numCols, uint32 const& lback)
	{
		for (uint32 i = 0; i < numCols; ++i)
		{
			uint32 minidx = numRows * (i - lback) + CGP_PARAM_INPUTS;
			if (minidx < CGP_PARAM_INPUTS)
				minidx = CGP_PARAM_INPUTS;

			uint32 maxidx = i * numRows + CGP_PARAM_INPUTS;

			for (uint32 j = 0; j < CGP_PARAM_INPUTS; ++j)
				table[i].push_back(j);

			for (uint32 j = minidx; j < maxidx; ++j)
				table[i].push_back(j);
		}

		#ifdef VERBOSE
		std::cout << "Possible Values:" << std::endl;
		std::cout << "-------------------------------" << std::endl;
		for (uint32 i = 0; i < numCols; ++i)
		{
			std::cout << "Col " << i << " vals: ";
			std::vector<uint32> colVals = table[i];
			for (std::vector<uint32>::const_iterator it = colVals.begin(); it != colVals.end(); ++it)
			{
				std::cout << *it << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "-------------------------------" << std::endl;
		#endif
	}

	void create_init_population(Population& population, std::vector<uint32>* possibleValues, uint32 const& maxPopulation, uint32 const& numRows, uint32 const& numCols)
	{
		population.reserve(maxPopulation);
		for (uint32 i = 0; i < maxPopulation; ++i)
		{
			Chromosome ch;
			uint32 j = 0;
			for (uint32 col = 0; col < numCols; ++col)
			{
				for (uint32 row = 0; row < numRows; ++row)
				{
					ch.val[j++] = possibleValues[col][rand() % possibleValues[col].size()];
					ch.val[j++] = possibleValues[col][rand() % possibleValues[col].size()];
					ch.val[j++] = rand() % NUM_FUNCTIONS;
				}
			}

			for (uint32 output = 0; output < CGP_PARAM_OUTPUTS; ++output)
				ch.val[j++] = rand() % (numRows * numCols + CGP_PARAM_INPUTS);

			population[i] = ch;
		}

		#ifdef VERBOSE
		std::cout << std::endl << "Initial population:" << std::endl;
		std::cout << "-------------------------------" << std::endl;
		for (uint32 i = 0; i < maxPopulation; ++i)
		{
			Chromosome ch = population[i];
			for (uint32 j = 0; j < CGP_CHROMOSOME_SIZE; ++j)
			{
				std::cout << ch.val[j];
				if (j % 3 < 2)
					std::cout << ",";
				else
					std::cout << ";";
			}			
			std::cout << std::endl;
		}
		std::cout << "-------------------------------" << std::endl;
		#endif
	}

	Chromosome mutate(Chromosome parent, const std::vector<uint32>* possibleValues, const uint32 numBits, const uint32 chromosomeLength, const uint32 numRows, const uint32 numCols)
	{
		const uint32 numGenes = rand() % numBits + 1;
		for (uint32 i = 0; i < numGenes; ++i)
		{		
			const uint32 idx = rand() % ((3 * numRows * numCols) + CGP_PARAM_OUTPUTS);
			const uint32 col = idx / (3 * numRows);
			const uint32 rnd = rand();
			
			// ouptut
			if (idx < (3 * numRows * numCols))
			{		
				// input
				if ((idx % 3) < 2)
					parent.val[idx] = possibleValues[col][rnd % possibleValues[col].size()];
				// func
				else
					parent.val[idx] = rand() % NUM_FUNCTIONS;				
			}
			else			
				parent.val[idx] = rand() % (numCols * numRows + CGP_PARAM_INPUTS);			
			
		}
		return parent;
	}

	///////////////////////////////////////////////////////////////
	// CGPWrapper
	///////////////////////////////////////////////////////////////

	bool CGPWrapper::run(FitnessMethod method, uint32 const& numRuns, uint32 const& numGenerations, uint32 const& numPopulation, uint32 const& numMutate)
	{
		srand(time(NULL));

		if (_inputImage.rows != _referenceImage.rows || _inputImage.cols != _referenceImage.cols)
		{
			std::cerr << "Input image and reference image dimensions are different." << std::endl;
			return false;
		}

		_filteredImage = cv::Mat(_inputImage.rows, _inputImage.cols, CV_8UC1);

		std::vector<uint32> possibleValues[CGP_PARAM_COLS];
		find_possible_col_values(possibleValues, CGP_PARAM_ROWS, CGP_PARAM_COLS, CGP_PARAM_LBACK);		

		for (uint32 r = 0; r < numRuns; ++r)
		{
			// generate initial population		
			Population pop;
			create_init_population(pop, possibleValues, numPopulation, CGP_PARAM_ROWS, CGP_PARAM_COLS);
			float fitness;
			switch (method)
			{
				// max value
				case MDPP:
					fitness = std::numeric_limits<float>::max();
					break;
				// min value
				case PSNR:
					fitness = std::numeric_limits<float>::min();
					break;
			}

			for (uint32 gen = 0; gen < numGenerations; ++gen)
			{
				int32 bestFilter = 0;
				std::vector<uint32> candidates;				

				for (uint32 ch = 0; ch < numPopulation; ++ch)
				{					
					// generate filtered image using chromosome evaluation
				
					for (uint32 y = 1; y < _inputImage.rows - 1; ++y)
					{
						for (uint32 x = 1; x < _inputImage.cols - 1; ++x)
						{
							// get image kernel and copy it to outputs
							uint8 kernel[9];
							get_3x3_kernel(kernel, _inputImage, x, y);

							_filteredImage.at<uint8>(y, x) = eval_chromosome(pop[ch], kernel, CGP_PARAM_ROWS, CGP_PARAM_COLS);
						}
					}

					// calculate fitness for the given filtered image
					float newFitness = calc_fitness(method, _filteredImage, _referenceImage);					
					if (newFitness == ERROR_FITNESS)
					{
						std::cerr << "An error occured while calculating fitness." << std::endl;
						return false;
					}

					switch (method)
					{
						// min value
						case MDPP:
						{
							if (newFitness < fitness)
							{
								candidates.clear();
								candidates.push_back(ch);
								fitness = newFitness;
							}
							else if (newFitness == fitness)						
								candidates.push_back(ch);
														
							break;
						}
						// max value
						case PSNR:
						{
							if (newFitness > fitness)
							{
								candidates.clear();
								candidates.push_back(ch);
								fitness = newFitness;
							}
							else if (newFitness == fitness)
								candidates.push_back(ch);
							
							break;
						}
					}										
				}

				bestFilter = candidates[rand() % candidates.size()];

				#ifdef VERBOSE
				std::cout << std::endl << "Generation[" << gen << "]" << std::endl;				
				std::cout << "f: " << fitness << " c: ";
				for (std::vector<uint32>::const_iterator it = candidates.begin(); it != candidates.end(); ++it)
					std::cout << *it << " ";
				std::cout << "b: " << bestFilter << std::endl;				
				std::cout << "-----------------------------" << std::endl;
				#endif

				evolve_population(pop, possibleValues, bestFilter, numPopulation, numMutate, CGP_PARAM_ROWS, CGP_PARAM_COLS);
			}
		}
		return true;
	}

	bool CGPWrapper::load_image(std::string const& filename, ImageType type)
	{
		switch (type)
		{
			case REFERENCE_IMAGE:
			{
				_referenceImage = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
				if (!_inputImage.data)
					return false;
				break;
			}
			case ORIGINAL_IMAGE:
			{
				_inputImage = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
				if (!_inputImage.data)
					return false;
				break;
			}
			default:
				break;
		}
		

		return true;
	}

	void CGPWrapper::display_image(ImageType type)
	{
		switch (type)
		{
			case REFERENCE_IMAGE:
				cv::imshow("Reference image", _referenceImage);			
				break;
			case FILTERED_IMAGE:
				cv::imshow("Filtered image", _filteredImage);
				break;
			case ORIGINAL_IMAGE:
				cv::imshow("Original image", _inputImage);
				break;
			default:
				break;
		}		
		cv::waitKey(0);
	}
	
	void CGPWrapper::save_image(std::string const& filename, ImageType type)
	{
		switch (type)
		{
			case REFERENCE_IMAGE:
				cv::imwrite(filename, _referenceImage);
				break;
			case FILTERED_IMAGE:
				cv::imwrite(filename, _filteredImage);
				break;
			case ORIGINAL_IMAGE:
				cv::imwrite(filename, _inputImage);
				break;
			default:
				break;
		}		
	}

} // namespace CGP
