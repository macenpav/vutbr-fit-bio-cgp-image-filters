#ifndef H_CGP
#define H_CGP

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#define POPULACE_MAX 5       //maximalni pocet jedincu populace
#define MUTACE_MAX 3         //max pocet genu, ktery se muze zmutovat behem jedne mutace (o 1 mensi!)

#define PARAM_M 6            //pocet sloupcu
#define PARAM_N 6            //pocet radku
#define L_BACK 1             //1 (pouze predchozi sloupec)  .. param_m (maximalni mozny rozsah);
#define DATASIZE 6

#define PARAM_GENERATIONS 50000   //max. pocet generaci evoluce
#define PARAM_RUNS 150            //max. pocet behu evoluce
#define FUNCTIONS 4              //max. pocet pouzitych funkci bloku (viz fitness() )
#define PERIODICLOGG  (PARAM_GENERATIONS/2) //po kolika krocich se ma vypsat populace
#define xPERIODIC_LOG           //zda se ma vypisovat populace

#define PARAM_IN 9
#define PARAM_OUT 1

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

typedef char int8;
typedef short int16;
typedef int int32;

class CGP
{
	public:
		/** @brief Initial image loading. */
		bool loadImage(std::string const& filename);

		/** @brief Displays image after filtering. */
		void displayFilteredImage();

		/** @brief A wrapper around the whole evolution stuff. */
		void runEvolution();		

	private:
		uint8 _filter(uint8* kernel);

		/** @brief Fills an array representing a 3x3 kernel with values. */
		void _get3x3Kernel(uint32 const& x, uint32 const y, uint8* kernel);

		/** @brief Input image. */
		cv::Mat _image;

		/** @brief Output image. */
		cv::Mat _filteredImage;

		/** @brief Options. */
		uint32 opt;
};

#endif
