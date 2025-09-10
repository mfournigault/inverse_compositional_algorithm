# Inverse Compositional Algorithm

## Description
This is a Python implementation of the inverse compositional algorithm as defined in the publicaiton "The Inverse Compositional Algorithm for Parametric Registration", Image Processing On Line, 6 (2016), pp. 212–232. https://doi.org/10.5201/ipol.2016.153.


The contributions of the paper are:

- The paper presents an implementation of the inverse compositional algorithm for parametric motion estimation, offering a method for computing global motion between images using a non-linear least square technique
- It introduces improvements over the Lucas-Kanade method for computing general parametric motion models, focusing on the study of the inverse compositional algorithm
- The authors proposed several theoretical justifications and enhancements over the basic model, including the study of different algorithms for image alignment, analysis of error functions, and strategies for dealing with linear appearance variations
- The paper discusses the introduction of robust error functions, different from the L2 norm, to improve the stability of the method under noise and occlusions, enabling the detection of predominant motion even in the presence of various displacements
- It explores the use of a coarse-to-fine strategy for handling large displacements in multi-channel images, enhancing the accuracy of the solutions provided by the iterative algorithm
- The authors delve into the application of robust error functions, as explained in previous works, to further enhance the efficiency and performance of the inverse compositional algorithm
- The paper extends the inverse compositional algorithm to 3D volumetric data and surfaces, broadening its applicability to more complex scenarios beyond planar transformations
- It discusses the implementation of incremental refinement and coarse-to-fine strategies for estimating large displacements, along with the inclusion of robust error functions to improve the overall effectiveness of the algorithm
- The authors also address the computational complexity of the variant of the inverse compositional algorithm for robust functions, highlighting its efficiency compared to other methods

The original publication comes with a C++ implementation.
This python implementation is based on the original C++ code. It is intented to be lightweight, easy to deploy on any platform and easy to use.
The python implementation makes use of standard libraries for classic image processing tasks like filter convolutions or bi-cubic interpolation, such as Scipy.

**Update 2024-08-26:**
The python implementation is based on the code of the modified inverse compositional algorithm. The original code was still buggy for homographies and was not working well if only one scale was processed with the quadratic metric.
The modified algorithm brings several improvements:
- reduced time computation,
- better convergence.
References of the modified inverse compositional algorithm are: "Improvements of the Inverse Compositional Algorithm for Parametric Motion Estimation",  Image Processing On Line, Vol 6, pp 435--464, 2018 https://doi.org/10.5201/ipol.2018.222.

**Update 2025-09-01:**
A faster implementation (hardware accelerated), based on Keras and Tensorflow has been developped providing in average a 10x acceleration compared to the numpy version.
It can be much faster if images processed in the batch are from the same source and can be processed under the same convergence criteria.
The implementation is available in the branch hardware_acceleration, and will be merged with the main branch soon.

## Installation
Instructions to install the project.

- Clone the repository
git clone https://github.com/mfournigault/inverse_compositional_algorithm.git

- Navigate to the project directory
cd inverse_compositional_algorithm

- Install dependencies
pip install -r requirements.txt

## Usage
Examples of how to use the project.
See examples in notebook `inverse_compositional_algorithm.ipynb` or others of subdirectory test.

## Configuration
Use the script configuration_handler.py to create a configuration file or to read a configuration file.
Modify the configuration file to change the parameters of the algorithm.

## Fork the project
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

## License
See LICENSE for more information.


Project Link: https://github.com/mfournigault/inverse_compositional_algorithm