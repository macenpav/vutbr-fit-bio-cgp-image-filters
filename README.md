# Imcgp
Imcgp (Image CGP) is a library and a tool for evolutionary design of image filters improving the quality of damaged images using Cartesian Genetic Programming. It is originally a semester project at <a href="http://www.fit.vutbr.cz/">BUT FIT</a> for <a href="http://www.fit.vutbr.cz/study/course-l.php?id=9884">BIN</a>.

## What's inside?
* C++ CGP library 
* CUDA accelerated CGP library
* MSVC project to build your own executable
* Results and measurements

## How to build?

*Requirements:* CUDA 6.5, OpenCV 2.4.*, Microsoft Visual Studio 2013

* Open the solution: bio-image-filters-cuda.sln in msvc folder
* C/C++ > Additional Include Directories : set to your OpenCV include directory
* Linker > Additional Library Directories : set to you OpenCV lib directory
* Linker > Input > Additional Dependencies : add all needed OpenCV libs
* CUDA should somehow manage on it's own if you have CUDA Toolkit installed (CUDA 7.0 propably won't work due to OpenCV dependency on CUDA 6.5)
* When everything is set up, the solution can be normally built.

## How to use the executable?

The executable supports several configuration options:

* -in [filename] Input image. (required)
* -ref [filename] Reference image. (required)
* -method [name] Fitness method used. Options: mdpp (default), psnr, mde
* -run [number] Number of runs.
* -mut [number] Number of mutations inside the chromosome.
* -gen [number] Number of generations.
* -pop [number] Population count inside a generation.
* -v Verbose output.
* -cuda Use CUDA acceleration.
* -csv Output fitness values per generation inside a CSV file.

## Other information

The whole project is under construction and is only a school project, therefore there are still a lot of TODOs and a lot less spare time. If you are interested in this topic, would like to contribute or have anything else on mind, you can contact me at <a href="mailto:macenauer.p@gmail.com">macenauer.p@gmail.com</a>.

In case you would like to use it, feel free to do so, only add some form of an acknowledgement. I would also be happy to help if it would lead to something useful.
