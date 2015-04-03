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
		if (bestFilter == ERROR_FILTER)
			return;

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
			const uint32 minidx = std::max(static_cast<uint32>(CGP_PARAM_OUTPUTS), static_cast<uint32>(numRows * (i - lback) + CGP_PARAM_INPUTS));
			const uint32 maxidx = i * numRows + CGP_PARAM_INPUTS;

			for (uint32 j = 0; j < CGP_PARAM_INPUTS; ++j)
				table[i].push_back(j);

			for (uint32 j = minidx; j < maxidx; ++j)
				table[i].push_back(j);
		}
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
					ch.val[j++] = possibleValues[row][rand() % possibleValues[row].size()];
					ch.val[j++] = possibleValues[row][rand() % possibleValues[row].size()];
					ch.val[j++] = rand() % NUM_FUNCTIONS;
				}
			}

			for (uint32 output = 0; output < CGP_PARAM_OUTPUTS; ++output)
				ch.val[j++] = rand() % (numRows * numCols + CGP_PARAM_INPUTS);

			population[i] = ch;
		}
	}

	Chromosome mutate(Chromosome const& parent, const std::vector<uint32>* possibleValues, const uint32 numBits, const uint32 chromosomeLength, const uint32 numRows, const uint32 numCols)
	{
		Chromosome child = parent;

		for (uint32 i = 0; i < numBits; ++i)
		{
			const uint32 maxValue = chromosomeLength - CGP_PARAM_INPUTS;
			const uint32 rnd = rand() % maxValue;						
			const uint32 row = rnd / numCols;
			
			// ouptut
			if (rnd == maxValue - CGP_PARAM_OUTPUTS)
			{		
				child.val[rnd] = rand() % maxValue;
			}
			else
			{
				// input
				if ((rnd % 3) < 2)
					child.val[rnd] = possibleValues[row][rand() % numRows];
				// func
				else
					child.val[rnd] = rand() % NUM_FUNCTIONS;
			}
			
		}

		return child;
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

		std::vector<uint32> possibleValues[CGP_PARAM_COLS];
		find_possible_col_values(possibleValues, CGP_PARAM_ROWS, CGP_PARAM_COLS, CGP_PARAM_LBACK);

		for (uint32 r = 0; r < numRuns; ++r)
		{
			std::cout << "Run: " << r << " ..." << std::endl;

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
				std::cout << "Generation " << gen << " ..." << std::endl;

				int32 bestFilter = ERROR_FILTER;
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
							if (newFitness <= fitness)
							{
								fitness = newFitness;
								bestFilter = static_cast<int32>(ch);
							}
							break;
						}
						// max value
						case PSNR:
						{
							if (newFitness >= fitness)
							{
								fitness = newFitness;
								bestFilter = static_cast<int32>(ch);
							}
							break;
						}
					}
				}

				evolve_population(pop, possibleValues, bestFilter, numPopulation, numMutate, CGP_PARAM_ROWS, CGP_PARAM_COLS);
			}
		}
		return true;
	}

	bool CGPWrapper::load_image(std::string const& filename)
	{
		_inputImage = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

		if (!_inputImage.data)
			return false;

		_filteredImage = cv::Mat(_inputImage.rows, _inputImage.cols, CV_8UC1);

		return true;
	}

	bool CGPWrapper::load_ref_image(std::string const& filename)
	{
		_referenceImage = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

		if (!_referenceImage.data)
			return false;

		return true;
	}

	void CGPWrapper::display_original_image()
	{
		cv::imshow("Original image", _inputImage);
		cv::waitKey(0);
	}

	void CGPWrapper::display_filtered_image()
	{
		cv::imshow("Filtered image", _filteredImage);
		cv::waitKey(0);
	}

	void CGPWrapper::save_filtered_image(std::string const& filename)
	{
		cv::imwrite(filename, _filteredImage);
	}

	void CGPWrapper::save_original_image(std::string const& filename)
	{
		cv::imwrite(filename, _inputImage);
	}

} // namespace CGP
