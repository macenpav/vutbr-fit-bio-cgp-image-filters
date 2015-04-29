#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "cgp.cuh"
#include "cgp_cuda.cuh"

namespace imcgp
{ 
	float calc_fitness(FitnessMethod method, cv::Mat const& input, cv::Mat const& reference, const uint32 offset = 1)
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
                for (uint32 y = offset; y < input.rows - offset; ++y)
				{
                    for (uint32 x = offset; x < input.cols - offset; ++x)
						fitness += std::abs(static_cast<float>(input.at<uint8>(y, x)) - static_cast<float>(reference.at<uint8>(y, x)));
				}
                fitness /= static_cast<float>((input.cols - 2 * offset) * (input.rows - 2 * offset));
				break;
			}
			case PSNR:
			{
				float tmp = 0.f;
                for (uint32 y = offset; y < input.rows - offset; ++y)
				{
                    for (uint32 x = offset; x < input.cols - offset; ++x)
					{
						float a = static_cast<float>(input.at<uint8>(y, x)) - static_cast<float>(reference.at<uint8>(y, x));
						tmp += (a * a);
					}
				}
                tmp /= ((input.cols - 2 * offset) * (input.rows - 2 * offset));
				fitness = 10.f * std::log10(255.f * 255.f / tmp);
				break;
			}
            case MSE:
            {
                fitness = 0.f;
                for (uint32 y = offset; y < input.rows - offset; ++y)
                {
                    for (uint32 x = offset; x < input.cols - offset; ++x)
                    {
                        float tmp = static_cast<float>(input.at<uint8>(y, x)) - static_cast<float>(reference.at<uint8>(y, x));
                        fitness += tmp * tmp;
                    }
                }
                fitness /= static_cast<float>((input.cols - 2 * offset) * (input.rows - 2 * offset));
                break;
            }
			
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

    void get_5x5_kernel(uint8* kernel, cv::Mat const& input, uint32 const& x, uint32 const& y)
    {
        kernel[0] = input.at<uint8>(y - 2, x - 2);
        kernel[1] = input.at<uint8>(y - 2, x - 1);
        kernel[2] = input.at<uint8>(y - 2, x);
        kernel[3] = input.at<uint8>(y - 2, x + 1);
        kernel[4] = input.at<uint8>(y - 2, x + 2);

        kernel[5] = input.at<uint8>(y - 1, x - 2);
        kernel[6] = input.at<uint8>(y - 1, x - 1);
        kernel[7] = input.at<uint8>(y - 1, x);
        kernel[8] = input.at<uint8>(y - 1, x + 1);
        kernel[9] = input.at<uint8>(y - 1, x + 2);

        kernel[10] = input.at<uint8>(y, x - 2);
        kernel[11] = input.at<uint8>(y, x - 1);
        kernel[12] = input.at<uint8>(y, x);
        kernel[13] = input.at<uint8>(y, x + 1);
        kernel[14] = input.at<uint8>(y, x + 2);

        kernel[15] = input.at<uint8>(y + 1, x - 2);
        kernel[16] = input.at<uint8>(y + 1, x - 1);
        kernel[17] = input.at<uint8>(y + 1, x);
        kernel[18] = input.at<uint8>(y + 1, x + 1);
        kernel[19] = input.at<uint8>(y + 1, x + 2);

        kernel[20] = input.at<uint8>(y + 2, x - 2);
        kernel[21] = input.at<uint8>(y + 2, x - 1);
        kernel[22] = input.at<uint8>(y + 2, x);
        kernel[23] = input.at<uint8>(y + 2, x + 1);
        kernel[24] = input.at<uint8>(y + 2, x + 2);
    }

    uint8 eval_chromosome(Chromosome const& chromosome, uint8* inputs, uint32 const& numRows, uint32 const& numCols, uint32 const& numInputs)
	{		
        uint8* outputs;
        if (numInputs == 9)
        { 
            outputs = (uint8*)malloc(CGP_PARAM_TOTAL_3X3);
            memcpy(outputs, inputs, CGP_PARAM_INPUTS_3X3 * sizeof(uint8));
        }        
        else if (numInputs == 25)
        {
            outputs = (uint8*)malloc(CGP_PARAM_TOTAL_5X5);
            memcpy(outputs, inputs, CGP_PARAM_INPUTS_5X5 * sizeof(uint8));
        }
		
		
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
                    case FUNC_SHL1: out = in1 << 1; break;
                    case FUNC_SHL2: out = in1 << 2; break;
					default: out = 255;
				}

                if (numInputs == 9)                
                    outputs[numRows * i + j + CGP_PARAM_INPUTS_3X3] = out;                
                else if (numInputs == 25)
                    outputs[numRows * i + j + CGP_PARAM_INPUTS_5X5] = out;				
			}
		}

		return out;
	}

	void evolve_population(Population& population, std::vector<uint32>* possibleValues, uint32 const& bestFilter, uint32 const& numPopulation, uint32 const& numMutate, uint32 const& numRows, uint32 const& numCols, uint32 const& numInputs)
	{
		Chromosome parent = population[bestFilter];

		population[0] = parent;
		for (uint32 ch = 1; ch < numPopulation; ++ch)
		{
            population[ch] = mutate(parent, possibleValues, numMutate, CGP_CHROMOSOME_SIZE, numRows, numCols, numInputs);
		}
	}

	void find_possible_col_values(std::vector<uint32>* table, uint32 const& numRows, uint32 const& numCols, uint32 const& lback, uint32 const& numInputs)
	{
		for (uint32 i = 0; i < numCols; ++i)
		{
            uint32 minidx = numRows * (i - lback) + numInputs;
            if (minidx < numInputs)
                minidx = numInputs;

            uint32 maxidx = i * numRows + numInputs;

            for (uint32 j = 0; j < numInputs; ++j)
				table[i].push_back(j);

			for (uint32 j = minidx; j < maxidx; ++j)
				table[i].push_back(j);
		}

		#ifdef DEBUG
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

	void create_init_population(Population& population, std::vector<uint32>* possibleValues, uint32 const& maxPopulation, uint32 const& numRows, uint32 const& numCols, uint32 const& numInputs)
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
                ch.val[j++] = rand() % (numRows * numCols + numInputs);

			population[i] = ch;
		}

		#ifdef DEBUG
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

	Chromosome mutate(Chromosome parent, const std::vector<uint32>* possibleValues, uint32 const& numBits, uint32 const& chromosomeLength, uint32 const& numRows, uint32 const& numCols, uint32 const& numInputs)
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
                parent.val[idx] = rand() % (numCols * numRows + numInputs);
			
		}
		return parent;
	}

	///////////////////////////////////////////////////////////////
	// CGPWrapper
	///////////////////////////////////////////////////////////////

	bool CGPWrapper::run(FitnessMethod method, uint32 const& numRuns, uint32 const& numGenerations, uint32 const& numPopulation, uint32 const& numMutate, uint32 const& numInputs)
	{
		srand(time(NULL));

		if (_inputImage.rows != _referenceImage.rows || _inputImage.cols != _referenceImage.cols)
		{
			std::cerr << "Input image and reference image dimensions are different." << std::endl;
			return false;
		}      

        if (numInputs != 9 && numInputs != 25)
        {            
            std::cerr << "The number of inputs " << std::endl;
            return false;
        }

		_filteredImage = cv::Mat(_inputImage.rows, _inputImage.cols, CV_8UC1);
        
        if (_options & OPT_CUDA_ACCELERATION)
        {
            GPU_CHECK_ERROR(cudaMalloc((void**)&_cudaInputImage, _inputImage.rows * _inputImage.cols * sizeof(uint8)));
            GPU_CHECK_ERROR(cudaMalloc((void**)&_cudaFilteredImage, _inputImage.rows * _inputImage.cols * sizeof(uint8)));            
            GPU_CHECK_ERROR(cudaMemcpy(_cudaInputImage, _inputImage.data, _inputImage.cols * _inputImage.rows * sizeof(uint8), cudaMemcpyHostToDevice));
        }

		std::vector<uint32> possibleValues[CGP_PARAM_COLS];
        find_possible_col_values(possibleValues, CGP_PARAM_ROWS, CGP_PARAM_COLS, CGP_PARAM_LBACK, numInputs);

        // in case we are using gpu acceleration
        // decide which kernel to use
        void(*filter_image_func)(const uint8*, uint8*, const Chromosome*, const uint32, const uint32);
        if (_options & OPT_CUDA_ACCELERATION)
        {
            if (numInputs == 9)
                filter_image_func = cuda::filter_image_3x3;
            else if (numInputs == 25)
                filter_image_func = cuda::filter_image_5x5;
            else
            {
                std::cerr << "Wrong input parameter value." << std::endl;
                return false;
            }
        }        

		for (uint32 r = 0; r < numRuns; ++r)
		{
            std::time_t t = std::time(NULL);
            auto now = std::time(nullptr);
            auto tm = *std::localtime(&now);
            std::stringstream ss;
            ss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
            std::string date = ss.str();

            std::string outStatsFilename = "stats (" + date + ").txt";
            std::string outCsvFilename = "fitness (" + date + ").csv";

            if (_options & OPT_OUTPUT_CSV)
            {
                std::ofstream file;
                file.open(outCsvFilename, std::ios::out);
                file << "generation;fitness" << std::endl;
                file.close();
            }

			_stats.total_time = 0.0;
	
			// generate initial population					
			auto timer_init_start = std::chrono::high_resolution_clock::now();
			Population pop;
			create_init_population(pop, possibleValues, numPopulation, CGP_PARAM_ROWS, CGP_PARAM_COLS, numInputs);
            auto timer_init_end = std::chrono::high_resolution_clock::now();

            _stats.init_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(timer_init_end - timer_init_start).count()) / 1000.0;

			if (_options & OPT_VERBOSE)							
                std::cout << "Initial population created in " << _stats.init_time << " ms" << std::endl;             

			float fitness;
			switch (method)
			{
				// max value
                case MSE:
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
				auto timer_gen_start = std::chrono::high_resolution_clock::now();

				int32 bestFilter = 0;
				std::vector<uint32> candidates;		
                Chromosome* cuda_pop;                

                if (_options & OPT_CUDA_ACCELERATION)
                {
                    GPU_CHECK_ERROR(cudaMalloc((void**)&cuda_pop, sizeof(Chromosome) * numPopulation));
                    GPU_CHECK_ERROR(cudaMemcpy(cuda_pop, &pop[0], sizeof(Chromosome) * numPopulation, cudaMemcpyHostToDevice));
                }

				for (uint32 ch = 0; ch < numPopulation; ++ch)
				{					
					// generate filtered image using chromosome evaluation
				
                    if (_options & OPT_CUDA_ACCELERATION)
                    {
                        dim3 block(16, 16);
                        dim3 grid(64, 64);                        
                        
                        filter_image_func<<<grid, block>>>(_cudaInputImage, _cudaFilteredImage, &cuda_pop[ch], _inputImage.cols, _inputImage.rows);                      

                        GPU_CHECK_ERROR(cudaMemcpy(_filteredImage.data, _cudaFilteredImage, _filteredImage.cols * _filteredImage.rows * sizeof(uint8), cudaMemcpyDeviceToHost));
                    }
                    else
                    {
                        for (uint32 y = 1; y < _inputImage.rows - 1; ++y)
                        {
                            for (uint32 x = 1; x < _inputImage.cols - 1; ++x)
                            {
                                // get image kernel and copy it to outputs
                                if (numInputs == CGP_PARAM_INPUTS_3X3)
                                {
                                    uint8 kernel[CGP_PARAM_INPUTS_3X3];
                                    get_3x3_kernel(kernel, _inputImage, x, y);
                                    _filteredImage.at<uint8>(y, x) = eval_chromosome(pop[ch], kernel, CGP_PARAM_ROWS, CGP_PARAM_COLS, numInputs);
                                }
                                else if (numInputs == CGP_PARAM_INPUTS_5X5)
                                {
                                    uint8 kernel[CGP_PARAM_INPUTS_5X5];
                                    get_3x3_kernel(kernel, _inputImage, x, y);
                                    _filteredImage.at<uint8>(y, x) = eval_chromosome(pop[ch], kernel, CGP_PARAM_ROWS, CGP_PARAM_COLS, numInputs);
                                }   
                                else
                                {
                                    std::cerr << "Wrong input parameter value." << std::endl;
                                    return false;
                                }
                            }
                        }
                    }

                    float newFitness = calc_fitness(method, _filteredImage, _referenceImage, numInputs == 9 ? 1 : 2);                    
					
					if (newFitness == ERROR_FITNESS)
					{
						std::cerr << "An error occured while calculating fitness." << std::endl;
						return false;
					}
                

					switch (method)
					{
						// min value
                        case MSE:
						case MDPP: // mean difference per pixel
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
						case PSNR: // peak signal-to-noise ration
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

                if (_options & OPT_CUDA_ACCELERATION)
                {
                    cudaFree(cuda_pop);
                }                

				bestFilter = candidates[rand() % candidates.size()];
				evolve_population(pop, possibleValues, bestFilter, numPopulation, numMutate, CGP_PARAM_ROWS, CGP_PARAM_COLS, numInputs);			
				                                
                auto timer_gen_end = std::chrono::high_resolution_clock::now();
                double gen_time = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(timer_gen_end - timer_gen_start).count()) / 1000.0;

                _stats.total_time += gen_time;
                _stats.average_gen_time += gen_time;
                
				if (_options & OPT_VERBOSE)
				{					       
                    std::cout << "Generation (" << gen << ") evaulated in " << gen_time << " ms" << std::endl;
                    std::cout << "Current fitness: " << fitness << std::endl;
				}		

                if (_options & OPT_OUTPUT_CSV)
                {
                    if (gen % 10 == 0)
                    {
                        std::ofstream file;
                        file.open(outCsvFilename, std::ios::out | std::ios::app);
                        file << gen << ";" << fitness << std::endl;
                        file.close();
                    }
                }
			}
														                
			_stats.average_gen_time /= static_cast<double>(numGenerations);
			_stats.fitness = fitness;
			_stats.best_filter = pop[0];
			_stats.num_generations = numGenerations;
			_stats.num_genes_mutated = numMutate;
            _stats.num_inputs = numInputs;
			_stats.population_size = numPopulation;
			_stats.method = method;			
            _stats.input_file = _inputImageFilename;
            _stats.reference_file = _refImageFilename;

			write_stats(outStatsFilename);
			save_image("filtered (" + date + ").jpg", FILTERED_IMAGE);
			save_image("original (" + date + ").jpg", ORIGINAL_IMAGE);
			save_image("reference (" + date + ").jpg", REFERENCE_IMAGE);
						
		}

        if (_options & OPT_CUDA_ACCELERATION)
        {
            GPU_CHECK_ERROR(cudaFree(_cudaInputImage));
            GPU_CHECK_ERROR(cudaFree(_cudaFilteredImage));            
        }

		return true;
	}

	void CGPWrapper::write_stats(std::string const& filename)
	{
		std::ofstream myfile;
		myfile.open(filename);

        myfile << "Input image: " << _stats.input_file << std::endl;
        myfile << "Reference image: " << _stats.reference_file << std::endl;
        myfile << "Init time: " << _stats.init_time << std::endl;
		myfile << "Total time: " << _stats.total_time << std::endl;
		myfile << "Average gen. time: " << _stats.average_gen_time << std::endl;
		myfile << "Fitness method: ";
		switch (_stats.method)
		{
            case MSE:
                myfile << "MSE";
                break;        
			case PSNR:
				myfile << "PSNR";
				break;
			case MDPP:
				myfile << "MDPP";
				break;
		}
		myfile << std::endl;
		myfile << "Fitness: " << _stats.fitness << std::endl;
		myfile << "Best filter: ";

		for (uint32 i = 0; i < CGP_CHROMOSOME_SIZE; ++i)
		{
			myfile << _stats.best_filter.val[i];
			if (i != CGP_CHROMOSOME_SIZE - 1)
			{
				if (i % 3 < 2)
					myfile << ",";
				else
					myfile << ";";
			}			
		}
		myfile << std::endl;
        myfile << "Number of inputs: " << _stats.num_inputs << std::endl;
		myfile << "Number of generations: " << _stats.num_generations << std::endl;
		myfile << "Max. genes mutated: " << _stats.num_genes_mutated << std::endl;
		myfile << "Population size: " << _stats.population_size << std::endl;		

		myfile.close();
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
                _refImageFilename = filename;
				break;
			}
			case ORIGINAL_IMAGE:
			{
				_inputImage = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
				if (!_inputImage.data)
					return false;
                _inputImageFilename = filename;
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

	void CGPWrapper::set_options(uint32 const& opts)
	{
		_options = opts;
	}

} // namespace CGP
