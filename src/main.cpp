#include <iostream>

#include "cgp.cuh"

int main(int32 argc, char** argv)
{	
	std::string filename, refFilename, methodname;
	imcgp::FitnessMethod method = imcgp::MDPP;
	uint32 numRuns = 1, numMutations = 5, numGenerations = 30000, numPopulation = 5;
	uint32 opts = 0;
	for (int32 i = 1; i < argc; ++i)
	{
		// input image
        if ((std::string(argv[i]) == "-i" || std::string(argv[i]) == "--input") && i + 1 < argc) {
			filename = argv[++i];
		}

        if ((std::string(argv[i]) == "-r" || std::string(argv[i]) == "--reference") && i + 1 < argc) {
			refFilename = argv[++i];
		}

        if ((std::string(argv[i]) == "-m" || std::string(argv[i]) == "--method") && i + 1 < argc) {
			methodname = argv[++i];
			if (methodname == "mdpp")
				method = imcgp::MDPP;
			else if (methodname == "psnr")
				method = imcgp::PSNR;
		}

        if ((std::string(argv[i]) == "-R" || std::string(argv[i]) == "--runs") && i + 1 < argc) {
			numRuns = atoi(argv[++i]);
		}

        if ((std::string(argv[i]) == "-M" || std::string(argv[i]) == "--mutations") && i + 1 < argc) {
			numMutations = atoi(argv[++i]);
		}

        if ((std::string(argv[i]) == "-g" || std::string(argv[i]) == "--generations") && i + 1 < argc) {
			numGenerations = atoi(argv[++i]);
		}

        if ((std::string(argv[i]) == "-p" || std::string(argv[i]) == "--population") && i + 1 < argc) {
			numPopulation = atoi(argv[++i]);
		}

        if (std::string(argv[i]) == "-v" || std::string(argv[i]) == "--verbose") {
			opts |= imcgp::OPT_VERBOSE;
		}

        if (std::string(argv[i]) == "-C" || std::string(argv[i]) == "--cuda") {
            opts |= imcgp::OPT_CUDA_ACCELERATION;
        }

        if (std::string(argv[i]) == "-c" || std::string(argv[i]) == "--csv") {
            opts |= imcgp::OPT_OUTPUT_CSV;
        }
	}

	if (filename.empty())
	{
		std::cerr << "Unspecified input image." << std::endl;
		return EXIT_FAILURE;
	}

    if (refFilename.empty())
    {
        std::cerr << "Unspecified reference image." << std::endl;
        return EXIT_FAILURE;
    }

	imcgp::CGPWrapper cgp;

	cgp.set_options(opts);

	if (cgp.load_image(filename, imcgp::ORIGINAL_IMAGE) && cgp.load_image(refFilename, imcgp::REFERENCE_IMAGE))
	{				
		bool result = cgp.run(method, numRuns, numGenerations, numPopulation, numMutations);
		if (!result)
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
