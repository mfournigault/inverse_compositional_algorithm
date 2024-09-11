# Inverse Compositional Algorithm

## Description
This is a Python implementation of the inverse compositional algorithm as defined in the publicaiton "The Inverse Compositional Algorithm for Parametric Registration", Image Processing On Line, 6 (2016), pp. 212–232. https://doi.org/10.5201/ipol.2016.153.
An improved version of the algorithm has been proposed by the author.
References of the modified inverse compositional algorithm are: "Improvements of the Inverse Compositional Algorithm for Parametric Motion Estimation",  Image Processing On Line, Vol 6, pp 435--464, 2018 https://doi.org/10.5201/ipol.2018.222.
The modified algorithm brings several improvements:
- reduced time computation,
- better accuracy and convergence of algorithms.

Our impplementation is based on the code of the modified inverse compositional algorithm.
For the moment, not all the improvements have been implemented in the code. Only the improvement of the definition of the image domain is implemented.

## About the implementation
The python implementation is based on numpy and skimage libraries.
In consequence, results on transformation and error calculations may differ from those obtained by the original C++ code.

## Methodology of validation
For each modification made in the code, we propose the following methodology to validate the code:
- start by validating the simple and robust version of the algorithm, with only one scale,
- compare the results with those obtained with the original C++ code executed on the same images,
- for the simple and robust version, validate at least for each type of transformation,
- for the robust version, in addition validate the results for each type of metric possible,
- for the pyramidal version, validate the results for at least two and three scales.

Please note that with only one scale, the simple and robust version may not converge to a good solution with the given meta-parameters (for example a tolerance limit on error of 0.001 and a maximum number of iterations of 30). What matters is that the results are close to those obtained with the original C++ code.
With more than one scale, the pyramidal version should converge to a good solution with the given meta-parameters.

Plotting graph of the error and dp values of iterations and comparing them to reference values can be a good way to validate the code.

## Simple version results
Reference results are
./inverse_compositional_algorithm ./data2/rubber_whale_tr.png ./data2/rubber_whale.png -t 2 -r 0 -c 0 -p 1 -d 10 -g 0 -v -n 
arameters: scales=1, zoom=0.500000, TOL=0.001000, transform type=2, robust function=0, lambda=0.000000, output file=transform.mat, delta=10, nanifoutside=1, graymethod=0, first scale=0, gradient type=0, type output=0
Scale: 0 (L2 norm)
b = [3.91783e+07 1.32992e+07 ]
Iteration 0: |Dp|=1.130254: p=(-1.024215 -0.477970)
b = [4.05265e+07 1.35715e+07 ]
Iteration 1: |Dp|=1.166537: p=(-2.083089 -0.967455)
b = [4.18154e+07 1.37805e+07 ]
Iteration 2: |Dp|=1.200521: p=(-3.174933 -1.466581)
b = [4.22291e+07 1.38452e+07 ]
Iteration 3: |Dp|=1.211402: p=(-4.277353 -1.968740)
b = [4.21989e+07 1.39496e+07 ]
Iteration 4: |Dp|=1.212127: p=(-5.379345 -2.473582)
b = [4.16225e+07 1.43099e+07 ]
Iteration 5: |Dp|=1.203329: p=(-6.468031 -2.986189)
b = [4.09084e+07 1.55039e+07 ]
Iteration 6: |Dp|=1.203570: p=(-7.542594 -3.528309)
b = [3.94396e+07 1.7664e+07 ]
Iteration 7: |Dp|=1.202136: p=(-8.587178 -4.123268)
b = [3.43554e+07 1.86609e+07 ]
Iteration 8: |Dp|=1.101560: p=(-9.507468 -4.728660)
b = [1.6899e+07 7.63685e+06 ]
Iteration 9: |Dp|=0.516177: p=(-9.955266 -4.985403)
b = [1.50968e+06 334026 ]
Iteration 10: |Dp|=0.041233: p=(-9.994168 -4.999072)
b = [198898 7628.62 ]
Iteration 11: |Dp|=0.005079: p=(-9.999178 -4.999904)
b = [28153.2 -218.224 ]
Iteration 12: |Dp|=0.000710: p=(-9.999883 -4.999988)

./inverse_compositional_algorithm ./data2/rubber_whale_rt.png ./data2/rubber_whale.png -t 3 -r 0 -c 0 -p 1 -d 10 -g 0 -v -n 1
Scale: 0 (L2 norm)
b = [-2.90975e+07 3.42413e+07 1.41154e+10 ]
Iteration 0: |Dp|=0.869457: p=(0.621655 -0.607866 -0.000510)
b = [-2.9194e+07 3.42495e+07 1.42313e+10 ]
Iteration 1: |Dp|=0.849563: p=(1.232598 -1.198209 -0.001094)
b = [-2.92223e+07 3.42066e+07 1.43493e+10 ]
Iteration 2: |Dp|=0.821754: p=(1.827446 -1.765161 -0.001772)
b = [-2.91848e+07 3.43444e+07 1.45302e+10 ]
Iteration 3: |Dp|=0.788395: p=(2.399570 -2.307599 -0.002567)
b = [-2.91712e+07 3.40742e+07 1.46856e+10 ]
Iteration 4: |Dp|=0.734385: p=(2.941646 -2.803053 -0.003532)
b = [-2.87533e+07 3.3489e+07 1.47322e+10 ]
Iteration 5: |Dp|=0.660002: p=(3.438052 -3.238004 -0.004696)
b = [-2.82451e+07 3.33599e+07 1.4837e+10 ]
Iteration 6: |Dp|=0.598166: p=(3.889650 -3.630254 -0.006032)
b = [-2.78146e+07 3.29022e+07 1.49319e+10 ]
Iteration 7: |Dp|=0.521325: p=(4.292303 -3.961385 -0.007583)
b = [-2.74868e+07 3.296e+07 1.51426e+10 ]
Iteration 8: |Dp|=0.459332: p=(4.650937 -4.248376 -0.009327)
b = [-2.70763e+07 3.30684e+07 1.53163e+10 ]
Iteration 9: |Dp|=0.403926: p=(4.967744 -4.498948 -0.011239)
b = [-2.66782e+07 3.30888e+07 1.54148e+10 ]
Iteration 10: |Dp|=0.360050: p=(5.250950 -4.721268 -0.013276)
b = [-2.63381e+07 3.30246e+07 1.5462e+10 ]
Iteration 11: |Dp|=0.325364: p=(5.508026 -4.920690 -0.015407)
b = [-2.60514e+07 3.30467e+07 1.55449e+10 ]
Iteration 12: |Dp|=0.291642: p=(5.739538 -5.098038 -0.017636)
b = [-2.55726e+07 3.30413e+07 1.56055e+10 ]
Iteration 13: |Dp|=0.249621: p=(5.937840 -5.249635 -0.019976)
b = [-2.48384e+07 3.25976e+07 1.5537e+10 ]
Iteration 14: |Dp|=0.192424: p=(6.094506 -5.361334 -0.022445)
b = [-2.40391e+07 3.21936e+07 1.54623e+10 ]
Iteration 15: |Dp|=0.135497: p=(6.208298 -5.434848 -0.025039)
b = [-2.33588e+07 3.20574e+07 1.54971e+10 ]
Iteration 16: |Dp|=0.079478: p=(6.278684 -5.471657 -0.027777)
b = [-2.26322e+07 3.15833e+07 1.5486e+10 ]
Iteration 17: |Dp|=0.027158: p=(6.300166 -5.455299 -0.030691)
b = [-2.18283e+07 3.10226e+07 1.53952e+10 ]
Iteration 18: |Dp|=0.067397: p=(6.277028 -5.392072 -0.033749)
b = [-2.07801e+07 3.04352e+07 1.52124e+10 ]
Iteration 19: |Dp|=0.123575: p=(6.208176 -5.289504 -0.036916)
b = [-1.95288e+07 3.02461e+07 1.50681e+10 ]
Iteration 20: |Dp|=0.174897: p=(6.089275 -5.161283 -0.040178)
b = [-1.80685e+07 2.99204e+07 1.47347e+10 ]
Iteration 21: |Dp|=0.208312: p=(5.930263 -5.026752 -0.043444)
b = [-1.63927e+07 2.91741e+07 1.41923e+10 ]
Iteration 22: |Dp|=0.238416: p=(5.735592 -4.889148 -0.046660)
b = [-1.50014e+07 2.81868e+07 1.36634e+10 ]
Iteration 23: |Dp|=0.270148: p=(5.511702 -4.738008 -0.049844)
b = [-1.36986e+07 2.71247e+07 1.31506e+10 ]
Iteration 24: |Dp|=0.304602: p=(5.259184 -4.567694 -0.053009)
b = [-1.30323e+07 2.60559e+07 1.28321e+10 ]
Iteration 25: |Dp|=0.342644: p=(4.985879 -4.361057 -0.056219)
b = [-1.28334e+07 2.55681e+07 1.28419e+10 ]
Iteration 26: |Dp|=0.387684: p=(4.689642 -4.110993 -0.059552)
b = [-1.28607e+07 2.54073e+07 1.29999e+10 ]
Iteration 27: |Dp|=0.431231: p=(4.372129 -3.819217 -0.063031)
b = [-1.27964e+07 2.43148e+07 1.27702e+10 ]
Iteration 28: |Dp|=0.457052: p=(4.053357 -3.491697 -0.066545)
b = [-1.22253e+07 2.24351e+07 1.204e+10 ]
Iteration 29: |Dp|=0.454971: p=(3.749801 -3.152817 -0.069927)


