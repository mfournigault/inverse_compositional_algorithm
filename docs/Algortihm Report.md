# Inverse Compositional Algorithm

## 1. Description
This is a Python implementation of the inverse compositional algorithm as defined in the publicaiton "The Inverse Compositional Algorithm for Parametric Registration", Image Processing On Line, 6 (2016), pp. 212–232. https://doi.org/10.5201/ipol.2016.153.
An improved version of the algorithm has been proposed by the author.
References of the modified inverse compositional algorithm are: "Improvements of the Inverse Compositional Algorithm for Parametric Motion Estimation",  Image Processing On Line, Vol 6, pp 435--464, 2018 https://doi.org/10.5201/ipol.2018.222.
The modified algorithm brings several improvements:
- reduced time computation,
- better accuracy and convergence of algorithms.

Our impplementation is based on the code of the modified inverse compositional algorithm.
For the moment, not all the improvements have been implemented in the code. Only the improvement of the definition of the image domain is implemented.

## 2. About the implementation
The python implementation is based on numpy and skimage libraries.
In consequence, results on transformation and error calculations may differ from those obtained by the original C++ code.

## 3. Environment of execution
The environment of execution is defined in the file requirements.txt of the project.
The version of python used is 3.11.


## 4. Methodology of validation
For each modification made in the code, we propose the following methodology to validate the code:
- start by validating the simple and robust version of the algorithm, with only one scale,
- compare the results with those obtained with the original C++ code executed on the same images,
- for the simple and robust version, validate at least for each type of transformation,
- for the robust version, in addition validate the results for each type of metric possible,
- for the pyramidal version, validate the results for at least two and three scales.

Please note that with only one scale, the simple and robust version may not converge to a good solution with the given meta-parameters (for example a tolerance limit on error of 0.001 and a maximum number of iterations of 30). What matters is that the results are close to those obtained with the original C++ code.
With more than one scale, the pyramidal version should converge to a good solution with the given meta-parameters.

Plotting graph of the error and dp values of iterations and comparing them to reference values can be a good way to validate the code.

## 5. Simple version results
### 1. Reference results
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

For three scales, the results are:

./inverse_compositional_algorithm ./data2/rubber_whale_tr.png ./data2/rubber_whale.png -t 2 -r 0 -c 0 -p 1 -d 10 -g 0 -v -n 3
Parameters: scales=3, zoom=0.500000, TOL=0.001000, transform type=2, robust function=0, lambda=0.000000, output file=transform.mat, delta=10, nanifoutside=1, graymethod=0, first scale=0, gradient type=0, type output=0
Scale: 2 (L2 norm)
b = [6.84122e+06 2.59181e+06 ]
Iteration 0: |Dp|=1.623572: p=(-1.464218 -0.701465)
b = [4.38466e+06 1.86933e+06 ]
Iteration 1: |Dp|=1.064025: p=(-2.408678 -1.191510)
b = [446081 323983 ]
Iteration 2: |Dp|=0.125482: p=(-2.508631 -1.267372)
b = [-42363.2 -10145.7 ]
Iteration 3: |Dp|=0.009453: p=(-2.499735 -1.264176)
b = [3715.07 280.293 ]
Iteration 4: |Dp|=0.000780: p=(-2.500497 -1.264338)
Scale: 1 (L2 norm)
b = [51884.3 -467579 ]
Iteration 0: |Dp|=0.033136: p=(-4.999919 -2.495558)
b = [-7946.63 71743 ]
Iteration 1: |Dp|=0.005084: p=(-5.000085 -2.500639)
b = [1257.66 -11269.9 ]
Iteration 2: |Dp|=0.000799: p=(-5.000060 -2.499841)
Scale: 0 (L2 norm)
b = [-5592.77 12227 ]
Iteration 0: |Dp|=0.000324: p=(-10.000018 -4.999990)
./inverse_compositional_algorithm ./data2/rubber_whale_rt.png ./data2/rubber_whale.png -t 3 -r 0 -c 0 -p 1 -d 10 -g 0 -v -n 3

Parameters: scales=3, zoom=0.500000, TOL=0.001000, transform type=3, robust function=0, lambda=0.000000, output file=transform.mat, delta=10, nanifoutside=1, graymethod=0, first scale=0, gradient type=0, type output=0
Scale: 2 (L2 norm)
b = [-3.87072e+06 5.20886e+06 5.08983e+08 ]
Iteration 0: |Dp|=0.643638: p=(0.383185 -0.517117 -0.005373)
b = [-4.11314e+06 4.73311e+06 4.72246e+08 ]
Iteration 1: |Dp|=0.766672: p=(0.916341 -1.068045 -0.008585)
b = [-4.19022e+06 4.48941e+06 4.66503e+08 ]
Iteration 2: |Dp|=0.717615: p=(1.448718 -1.549223 -0.012280)
b = [-3.96355e+06 4.22193e+06 4.69161e+08 ]
Iteration 3: |Dp|=0.465319: p=(1.820852 -1.828497 -0.018855)
b = [-3.54738e+06 3.89235e+06 4.66163e+08 ]
Iteration 4: |Dp|=0.157271: p=(1.974282 -1.861538 -0.028939)
b = [-2.90303e+06 3.46507e+06 4.43282e+08 ]
Iteration 5: |Dp|=0.218843: p=(1.893193 -1.658696 -0.042046)
b = [-2.18457e+06 3.08618e+06 4.03951e+08 ]
Iteration 6: |Dp|=0.425308: p=(1.635127 -1.320938 -0.056496)
b = [-1.34515e+06 2.56637e+06 3.37116e+08 ]
Iteration 7: |Dp|=0.567344: p=(1.244724 -0.909525 -0.070772)
b = [-500179 2.0105e+06 2.44759e+08 ]
Iteration 8: |Dp|=0.552042: p=(0.816742 -0.561033 -0.082425)
b = [-21926.2 1.38748e+06 1.56424e+08 ]
Iteration 9: |Dp|=0.439609: p=(0.454704 -0.311801 -0.090616)
b = [75619.6 752651 8.50336e+07 ]
Iteration 10: |Dp|=0.279072: p=(0.226357 -0.151440 -0.095473)
b = [67775.6 370241 4.05826e+07 ]
Iteration 11: |Dp|=0.142546: p=(0.107986 -0.072055 -0.097870)
b = [38395.5 171338 1.86655e+07 ]
Iteration 12: |Dp|=0.068349: p=(0.051143 -0.034118 -0.098999)
b = [17889 79256.7 8.65986e+06 ]
Iteration 13: |Dp|=0.031848: p=(0.024711 -0.016360 -0.099524)
b = [8409.81 36729.7 4.01541e+06 ]
Iteration 14: |Dp|=0.014828: p=(0.012412 -0.008082 -0.099768)
b = [3897.83 17047 1.86439e+06 ]
Iteration 15: |Dp|=0.006885: p=(0.006703 -0.004236 -0.099882)
b = [1806.78 7914 865663 ]
Iteration 16: |Dp|=0.003196: p=(0.004053 -0.002450 -0.099935)
b = [838.04 3674.2 401933 ]
Iteration 17: |Dp|=0.001484: p=(0.002823 -0.001621 -0.099959)
b = [388.899 1705.87 186618 ]
Iteration 18: |Dp|=0.000689: p=(0.002252 -0.001236 -0.099970)
Scale: 1 (L2 norm)
b = [23248 8231.65 1.56155e+06 ]
Iteration 0: |Dp|=0.004136: p=(0.001034 -0.000221 -0.099993)
b = [755.529 7461.75 1.4763e+06 ]
Iteration 1: |Dp|=0.000661: p=(0.000429 0.000046 -0.099999)
Scale: 0 (L2 norm)
b = [16733.9 11527.7 1.39799e+06 ]
Iteration 0: |Dp|=0.000722: p=(0.000138 0.000128 -0.100000)
./inverse_compositional_algorithm ./data2/rubber_whale_eu.png ./data2/rubber_whale.png -t 3 -r 0 -c 0 -p 1 -d 10 -g 0 -v -n 3
Parameters: scales=3, zoom=0.500000, TOL=0.001000, transform type=3, robust function=0, lambda=0.000000, output file=transform.mat, delta=10, nanifoutside=1, graymethod=0, first scale=0, gradient type=0, type output=0
Scale: 2 (L2 norm)
b = [-2.62179e+06 6.04119e+06 5.31338e+08 ]
Iteration 0: |Dp|=0.523805: p=(0.016483 -0.523493 -0.007410)
b = [-2.64062e+06 5.85858e+06 5.04164e+08 ]
Iteration 1: |Dp|=0.604142: p=(0.114837 -1.119549 -0.013028)
b = [-2.56719e+06 5.69586e+06 4.88735e+08 ]
Iteration 2: |Dp|=0.596664: p=(0.213490 -1.707976 -0.018344)
b = [-2.29416e+06 5.21956e+06 4.55293e+08 ]
Iteration 3: |Dp|=0.483003: p=(0.247121 -2.189771 -0.024232)
b = [-1.88173e+06 4.8235e+06 4.45075e+08 ]
Iteration 4: |Dp|=0.293330: p=(0.082768 -2.432567 -0.033212)
b = [-1.4393e+06 4.17842e+06 4.16856e+08 ]
Iteration 5: |Dp|=0.346471: p=(-0.262461 -2.405746 -0.045032)
b = [-975752 3.58359e+06 3.72853e+08 ]
Iteration 6: |Dp|=0.489260: p=(-0.715433 -2.221284 -0.057879)
b = [-492922 2.86581e+06 3.11843e+08 ]
Iteration 7: |Dp|=0.604324: p=(-1.234676 -1.912371 -0.070819)
b = [-113522 2.12713e+06 2.34411e+08 ]
Iteration 8: |Dp|=0.579604: p=(-1.720019 -1.595730 -0.081840)
b = [87034.9 1.38553e+06 1.54116e+08 ]
Iteration 9: |Dp|=0.458318: p=(-2.097690 -1.336202 -0.089875)
b = [135541 830419 9.02782e+07 ]
Iteration 10: |Dp|=0.297595: p=(-2.344995 -1.170738 -0.094845)
b = [100374 430652 4.60908e+07 ]
Iteration 11: |Dp|=0.162800: p=(-2.480954 -1.081228 -0.097481)
b = [54384.6 211335 2.24952e+07 ]
Iteration 12: |Dp|=0.081343: p=(-2.548972 -1.036635 -0.098785)
b = [28438 101252 1.07784e+07 ]
Iteration 13: |Dp|=0.040030: p=(-2.582396 -1.014617 -0.099420)
b = [13982.7 48655.3 5.1813e+06 ]
Iteration 14: |Dp|=0.019389: p=(-2.598575 -1.003936 -0.099727)
b = [6739.63 23405.7 2.49319e+06 ]
Iteration 15: |Dp|=0.009337: p=(-2.606364 -0.998789 -0.099875)
b = [3238.68 11256.2 1.1991e+06 ]
Iteration 16: |Dp|=0.004490: p=(-2.610109 -0.996313 -0.099946)
b = [1555.62 5411.69 576527 ]
Iteration 17: |Dp|=0.002158: p=(-2.611909 -0.995123 -0.099980)
b = [747.321 2601.38 277141 ]
Iteration 18: |Dp|=0.001037: p=(-2.612774 -0.994551 -0.099997)
b = [359.079 1250.35 133210 ]
Iteration 19: |Dp|=0.000499: p=(-2.613190 -0.994277 -0.100005)
Scale: 1 (L2 norm)
b = [-15264.2 -6820.2 36811.8 ]
Iteration 0: |Dp|=0.001541: p=(-5.224842 -1.988645 -0.100000)
b = [-141.456 -2519.29 -512912 ]
Iteration 1: |Dp|=0.000218: p=(-5.224648 -1.988743 -0.099998)
Scale: 0 (L2 norm)
b = [-7939.18 -16314 -609597 ]
Iteration 0: |Dp|=0.000772: p=(-10.449191 -3.976721 -0.100000)
./inverse_compositional_algorithm ./data2/rubber_whale_zo.png ./data2/rubber_whale.png -t 4 -r 0 -c 0 -p 1 -d 10 -g 0 -v -n 3
Parameters: scales=3, zoom=0.500000, TOL=0.001000, transform type=4, robust function=0, lambda=0.000000, output file=transform.mat, delta=10, nanifoutside=1, graymethod=0, first scale=0, gradient type=0, type output=0
Scale: 2 (L2 norm)
b = [-6.29274e+06 -2.14678e+06 -6.63469e+08 2.0422e+08 ]
Iteration 0: |Dp|=0.677635: p=(0.611005 0.301886 0.005762 0.000105)
b = [-6.20207e+06 -2.69377e+06 -6.8989e+08 1.73529e+08 ]
Iteration 1: |Dp|=0.686308: p=(1.186462 0.690821 0.012073 0.000256)
b = [-5.88808e+06 -3.1338e+06 -6.8112e+08 1.3581e+08 ]
Iteration 2: |Dp|=0.744934: p=(1.765533 1.180425 0.017989 0.000574)
b = [-3.78338e+06 -3.31401e+06 -4.45304e+08 1.10821e+07 ]
Iteration 3: |Dp|=1.033229: p=(2.548160 1.882860 0.017805 0.002402)
b = [-3.36801e+06 -2.69783e+06 -4.3253e+08 2.0185e+07 ]
Iteration 4: |Dp|=0.571340: p=(2.991754 2.261859 0.021212 0.004539)
b = [-2.97228e+06 -1.86829e+06 -4.02203e+08 4.29327e+07 ]
Iteration 5: |Dp|=0.142626: p=(3.126114 2.320025 0.027590 0.006618)
b = [-2.64582e+06 -1.25242e+06 -3.71521e+08 6.02298e+07 ]
Iteration 6: |Dp|=0.165533: p=(3.041664 2.171036 0.035793 0.008358)
b = [-2.48164e+06 -824580 -3.48076e+08 7.86667e+07 ]
Iteration 7: |Dp|=0.309791: p=(2.828965 1.927228 0.044776 0.009346)
b = [-2.32495e+06 -545163 -3.27033e+08 8.77941e+07 ]
Iteration 8: |Dp|=0.400887: p=(2.536365 1.622428 0.054161 0.009900)
b = [-2.24927e+06 -368619 -3.08287e+08 9.8378e+07 ]
Iteration 9: |Dp|=0.409212: p=(2.211255 1.333382 0.063260 0.009596)
b = [-2.09562e+06 -238026 -2.84619e+08 1.0081e+08 ]
Iteration 10: |Dp|=0.414809: p=(1.862079 1.058171 0.071982 0.008725)
b = [-1.89978e+06 -163534 -2.56003e+08 9.70863e+07 ]
Iteration 11: |Dp|=0.389793: p=(1.517435 0.816560 0.079972 0.007456)
b = [-1.66531e+06 -131487 -2.21756e+08 8.79825e+07 ]
Iteration 12: |Dp|=0.333290: p=(1.209177 0.626441 0.086844 0.005976)
b = [-1.35262e+06 -116457 -1.84042e+08 7.26787e+07 ]
Iteration 13: |Dp|=0.295190: p=(0.926153 0.471819 0.092720 0.004531)
b = [-1.03486e+06 -94863.8 -1.4282e+08 5.64157e+07 ]
Iteration 14: |Dp|=0.238924: p=(0.691258 0.355442 0.097366 0.003265)
b = [-799542 -83980.1 -1.10788e+08 4.28481e+07 ]
Iteration 15: |Dp|=0.182374: p=(0.511315 0.266437 0.100962 0.002313)
b = [-597363 -67075.9 -8.28575e+07 3.17844e+07 ]
Iteration 16: |Dp|=0.134934: p=(0.377473 0.201202 0.103644 0.001601)
b = [-423421 -55289.6 -6.07108e+07 2.21471e+07 ]
Iteration 17: |Dp|=0.105026: p=(0.273276 0.149980 0.105678 0.001105)
b = [-297096 -40049.9 -4.30157e+07 1.55455e+07 ]
Iteration 18: |Dp|=0.076007: p=(0.197547 0.113319 0.107135 0.000749)
b = [-217538 -30111.4 -3.14856e+07 1.11195e+07 ]
Iteration 19: |Dp|=0.054690: p=(0.143886 0.085166 0.108200 0.000520)
b = [-157004 -22478.7 -2.28535e+07 7.94059e+06 ]
Iteration 20: |Dp|=0.039871: p=(0.104932 0.064259 0.108977 0.000363)
b = [-113369 -16479.1 -1.65436e+07 5.70393e+06 ]
Iteration 21: |Dp|=0.028912: p=(0.076743 0.048959 0.109541 0.000253)
b = [-81857.2 -11965.2 -1.19578e+07 4.11056e+06 ]
Iteration 22: |Dp|=0.020918: p=(0.056360 0.037850 0.109948 0.000173)
b = [-59108.7 -8648.01 -8.637e+06 2.96739e+06 ]
Iteration 23: |Dp|=0.015115: p=(0.041629 0.029817 0.110243 0.000116)
b = [-42682.1 -6237.01 -6.23624e+06 2.14378e+06 ]
Iteration 24: |Dp|=0.010916: p=(0.030986 0.024020 0.110456 0.000075)
b = [-30820.8 -4494.57 -4.50218e+06 1.54918e+06 ]
Iteration 25: |Dp|=0.007882: p=(0.023297 0.019839 0.110610 0.000045)
b = [-22255.6 -3238.6 -3.25017e+06 1.11952e+06 ]
Iteration 26: |Dp|=0.005690: p=(0.017744 0.016824 0.110721 0.000023)
b = [-16070.5 -2334.08 -2.34634e+06 808938 ]
Iteration 27: |Dp|=0.004108: p=(0.013733 0.014650 0.110801 0.000007)
b = [-11604.1 -1682.7 -1.69389e+06 584444 ]
Iteration 28: |Dp|=0.002965: p=(0.010837 0.013081 0.110859 -0.000004)
b = [-8378.92 -1213.48 -1.2229e+06 422195 ]
Iteration 29: |Dp|=0.002141: p=(0.008746 0.011950 0.110901 -0.000012)
Scale: 1 (L2 norm)
b = [-77806.8 38368.7 -1.94681e+07 8.29106e+06 ]
Iteration 0: |Dp|=0.012116: p=(0.012416 0.011433 0.110977 0.000002)
b = [-40656 -33567.7 -1.56484e+07 -71500.8 ]
Iteration 1: |Dp|=0.002809: p=(0.009588 0.010113 0.111013 -0.000004)
b = [-22842.2 -27906.5 -1.09667e+07 -1.21873e+06 ]
Iteration 2: |Dp|=0.002508: p=(0.007028 0.009012 0.111040 -0.000008)
b = [-15788.5 -22000 -8.00702e+06 -1.34845e+06 ]
Iteration 3: |Dp|=0.001715: p=(0.005322 0.008164 0.111059 -0.000009)
b = [-10944.7 -16769.4 -5.83974e+06 -1.22977e+06 ]
Iteration 4: |Dp|=0.001234: p=(0.004132 0.007483 0.111073 -0.000008)
b = [-7721.76 -12670.4 -4.27421e+06 -1.03313e+06 ]
Iteration 5: |Dp|=0.000890: p=(0.003297 0.006954 0.111083 -0.000008)
Scale: 0 (L2 norm)
b = [32518.4 278630 3.15098e+07 4.67141e+07 ]
Iteration 0: |Dp|=0.015890: p=(-0.002017 -0.001505 0.111111 0.000002)
b = [-39401.7 -5265.89 -6.49521e+06 7.80486e+06 ]
Iteration 1: |Dp|=0.002501: p=(0.000207 0.000161 0.111107 0.000000)
b = [3291.29 9532.8 2.29533e+06 3.03007e+06 ]
Iteration 2: |Dp|=0.000458: p=(-0.000282 0.000022 0.111108 -0.000001)

### 2. Results of the python implementation
The results of the python implementation are available in the notebook `inverse_compositional_algorithm.ipynb`.
The results obtained with the python implementation are very similar to the reference results, including for the mullti-scale algorithm.
This validates the implementation of the algorithm in python.

## 6. Robust version results
### 1. Reference results
./inverse_compositional_algorithm ./data2/rubber_whale_tr.png ./data2/rubber_whale.png -t 2 -r 4 -c 0 -p 1 -d 10 -g 0 -v -n 3
Parameters: scales=3, zoom=0.500000, TOL=0.001000, transform type=2, robust function=4, lambda=0.000000, output file=transform.mat, delta=10, nanifoutside=1, graymethod=0, first scale=0, gradient type=0, type output=0
Scale: 2 (Robust error function 4)
|Dp|=1.380177: p=(-1.250102 -0.584923), lambda=72.000000
|Dp|=1.160920: p=(-2.284002 -1.112925), lambda=64.800000
|Dp|=0.297598: p=(-2.533152 -1.275679), lambda=58.320000
|Dp|=0.040185: p=(-2.495009 -1.263030), lambda=52.488000
|Dp|=0.006442: p=(-2.501303 -1.264404), lambda=47.239200
|Dp|=0.001044: p=(-2.500269 -1.264259), lambda=42.515280
|Dp|=0.000159: p=(-2.500429 -1.264255), lambda=38.263752
Scale: 1 (Robust error function 4)
|Dp|=0.034227: p=(-4.999723 -2.494302), lambda=72.000000
|Dp|=0.006760: p=(-5.000125 -2.501050), lambda=64.800000
|Dp|=0.001332: p=(-5.000051 -2.499721), lambda=58.320000
|Dp|=0.000257: p=(-5.000066 -2.499977), lambda=52.488000
Scale: 0 (Robust error function 4)
|Dp|=0.000139: p=(-10.000000 -5.000000), lambda=72.000000

./inverse_compositional_algorithm ./data2/rubber_whale_rt.png ./data2/rubber_whale.png -t 3 -r 4 -c 0 -p 1 -d 10 -g 0 -v -n 3
Parameters: scales=3, zoom=0.500000, TOL=0.001000, transform type=3, robust function=4, lambda=0.000000, output file=transform.mat, delta=10, nanifoutside=1, graymethod=0, first scale=0, gradient type=0, type output=0
Scale: 2 (Robust error function 4)
|Dp|=0.490404: p=(0.304926 -0.384058 -0.004001), lambda=72.000000
|Dp|=0.491950: p=(0.628362 -0.754713 -0.008296), lambda=64.800000
|Dp|=0.469007: p=(0.951079 -1.095004 -0.013161), lambda=58.320000
|Dp|=0.374179: p=(1.216846 -1.358319 -0.019726), lambda=52.488000
|Dp|=0.143893: p=(1.326261 -1.451217 -0.029900), lambda=47.239200
|Dp|=0.141008: p=(1.244363 -1.337295 -0.043960), lambda=42.515280
|Dp|=0.403871: p=(0.982176 -1.030583 -0.061207), lambda=38.263752
|Dp|=0.611352: p=(0.575339 -0.574651 -0.080199), lambda=34.437377
|Dp|=0.638208: p=(0.133432 -0.114466 -0.096256), lambda=30.993639
|Dp|=0.194853: p=(-0.014372 0.012437 -0.100392), lambda=27.894275
|Dp|=0.023612: p=(0.003889 -0.002524 -0.099931), lambda=25.104848
|Dp|=0.003102: p=(0.001440 -0.000620 -0.099987), lambda=22.594363
|Dp|=0.000350: p=(0.001725 -0.000824 -0.099980), lambda=20.334927
Scale: 1 (Robust error function 4)
|Dp|=0.004406: p=(-0.000386 0.000520 -0.100006), lambda=72.000000
|Dp|=0.000694: p=(0.000212 0.000167 -0.100001), lambda=64.800000
Scale: 0 (Robust error function 4)
|Dp|=0.000519: p=(-0.000019 0.000063 -0.100000), lambda=72.000000

./inverse_compositional_algorithm ./data2/rubber_whale_eu.png ./data2/rubber_whale.png -t 3 -r 4 -c 0 -p 1 -d 10 -g 0 -v -n 3
Parameters: scales=3, zoom=0.500000, TOL=0.001000, transform type=3, robust function=4, lambda=0.000000, output file=transform.mat, delta=10, nanifoutside=1, graymethod=0, first scale=0, gradient type=0, type output=0
Scale: 2 (Robust error function 4)
|Dp|=0.383129: p=(0.031471 -0.381789 -0.005898), lambda=72.000000
|Dp|=0.416312: p=(0.067546 -0.796492 -0.011791), lambda=64.800000
|Dp|=0.388467: p=(0.060923 -1.184843 -0.018555), lambda=58.320000
|Dp|=0.362796: p=(-0.015728 -1.539358 -0.026622), lambda=52.488000
|Dp|=0.348366: p=(-0.320987 -1.706753 -0.039061), lambda=47.239200
|Dp|=0.467276: p=(-0.787622 -1.725999 -0.054152), lambda=42.515280
|Dp|=0.643031: p=(-1.411702 -1.572013 -0.071511), lambda=38.263752
|Dp|=0.781855: p=(-2.129335 -1.262200 -0.089257), lambda=34.437377
|Dp|=0.549933: p=(-2.610027 -0.995280 -0.099998), lambda=30.993639
|Dp|=0.003975: p=(-2.613780 -0.993972 -0.100011), lambda=27.894275
|Dp|=0.000366: p=(-2.613433 -0.994087 -0.100010), lambda=25.104848
Scale: 1 (Robust error function 4)
|Dp|=0.002797: p=(-5.224174 -1.988932 -0.099995), lambda=72.000000
|Dp|=0.000468: p=(-5.224611 -1.988764 -0.099998), lambda=64.800000
Scale: 0 (Robust error function 4)
|Dp|=0.001036: p=(-10.449167 -3.976494 -0.100000), lambda=72.000000
|Dp|=0.000194: p=(-10.449181 -3.976687 -0.100000), lambda=64.800000

./inverse_compositional_algorithm ./data2/rubber_whale_zo.png ./data2/rubber_whale.png -t 4 -r 4 -c 0 -p 1 -d 10 -g 0 -v -n 3

Parameters: scales=3, zoom=0.500000, TOL=0.001000, transform type=4, robust function=4, lambda=0.000000, output file=transform.mat, delta=10, nanifoutside=1, graymethod=0, first scale=0, gradient type=0, type output=0
Scale: 2 (Robust error function 4)
|Dp|=0.576213: p=(0.521129 0.250926 0.003806 0.000427), lambda=72.000000
|Dp|=0.532385: p=(0.988143 0.516606 0.009282 0.002196), lambda=64.800000
|Dp|=0.554406: p=(1.435871 0.857161 0.014693 0.003652), lambda=58.320000
|Dp|=0.596619: p=(1.871099 1.281772 0.019169 0.004823), lambda=52.488000
|Dp|=0.224121: p=(1.966060 1.491833 0.029634 0.005047), lambda=47.239200
|Dp|=0.280911: p=(1.678743 1.431442 0.046977 0.003823), lambda=42.515280
|Dp|=0.529568: p=(1.213643 1.110431 0.067871 0.002709), lambda=38.263752
|Dp|=0.723817: p=(0.627559 0.582204 0.090497 0.001787), lambda=34.437377
|Dp|=0.679056: p=(0.071671 0.074571 0.108927 0.000209), lambda=30.993639
|Dp|=0.100077: p=(-0.010162 -0.000714 0.111377 -0.000128), lambda=27.894275
|Dp|=0.017576: p=(0.005777 0.010556 0.110945 -0.000012), lambda=25.104848
|Dp|=0.003090: p=(0.002774 0.008894 0.111020 -0.000041), lambda=22.594363
|Dp|=0.000542: p=(0.003324 0.009139 0.111007 -0.000034), lambda=20.334927
Scale: 1 (Robust error function 4)
|Dp|=0.015037: p=(-0.000031 0.002963 0.111132 0.000002), lambda=72.000000
|Dp|=0.002851: p=(0.001385 0.005796 0.111107 -0.000004), lambda=64.800000
|Dp|=0.000543: p=(0.001090 0.005269 0.111112 -0.000003), lambda=58.320000
Scale: 0 (Robust error function 4)
|Dp|=0.011423: p=(-0.000816 -0.001795 0.111105 -0.000002), lambda=72.000000
|Dp|=0.002267: p=(-0.000041 0.000601 0.111108 -0.000001), lambda=64.800000
|Dp|=0.000438: p=(-0.000247 0.000160 0.111107 -0.000002), lambda=58.320000

### 2. Results of the python implementation
The results of the python implementation are available in the notebook `inverse_compositional_algorithm_robust.ipynb`.
The results obtained with the python implementation are correct when we use several scales, we can converge to a good solution.
Nevertheless when we compare these results to the reference results, we can see that the gradient dp converges much slower than in the reference results.
It should not be the case, so there is a problem of calculation in the python implementation to solve.