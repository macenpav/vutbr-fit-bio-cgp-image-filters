#include <iostream>

#include "cgp.h"

int main(int32 argc, char** argv)
{	
	std::string filename, refFilename, methodname;
	imcgp::FitnessMethod method = imcgp::MDPP;
	uint32 numRuns = 1, numMutations = 5, numGenerations = 30000, numPopulation = 5;
	for (int32 i = 1; i < argc; ++i)
	{
		// input image
		if (std::string(argv[i]) == "-in" && i + 1 < argc) {
			filename = argv[++i];
		}

		if (std::string(argv[i]) == "-ref" && i + 1 < argc) {
			refFilename = argv[++i];
		}

		if (std::string(argv[i]) == "-method" && i + 1 < argc) {
			methodname = argv[++i];
			if (methodname == "mdpp")
				method = imcgp::MDPP;
			else if (methodname == "psnr")
				method = imcgp::PSNR;
		}

		if (std::string(argv[i]) == "-run" && i + 1 < argc) {
			numRuns = atoi(argv[++i]);
		}

		if (std::string(argv[i]) == "-mut" && i + 1 < argc) {
			numMutations = atoi(argv[++i]);
		}

		if (std::string(argv[i]) == "-gen" && i + 1 < argc) {
			numGenerations = atoi(argv[++i]);
		}

		if (std::string(argv[i]) == "-pop" && i + 1 < argc) {
			numPopulation = atoi(argv[++i]);
		}
	}

	if (filename.empty())
	{
		std::cerr << "Unspecified input." << std::endl;
		return EXIT_FAILURE;
	}

	imcgp::CGPWrapper cgp;

	if (cgp.load_image(filename) && cgp.load_ref_image(refFilename))
	{
		if (cgp.run(method, numRuns, numGenerations, numPopulation, numMutations))
		{ 
			cgp.save_filtered_image("filered.jpg");
			cgp.save_original_image("original.jpg");
		}
		else
		{
			std::cerr << "Error occured while running evolution." << std::endl;
			return EXIT_FAILURE;
		}
	}
	else
	{
		std::cerr << "Cannot load file." << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
