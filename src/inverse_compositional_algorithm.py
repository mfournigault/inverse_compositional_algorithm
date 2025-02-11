import numpy as np
from skimage.transform import rescale
from skimage.filters import prewitt_h, prewitt_v
from skimage.exposure import rescale_intensity
from skimage import img_as_ubyte
import imageio

import derivatives as de
import image_optimisation as io
import transformation as tr
# import conv_filtering as cf
import bicubic_interpolation as bi
import zoom as zm
import constants as cts

 
def inverse_compositional_algorithm(
        I1: np.ndarray, 
        I2: np.ndarray, 
        p: np.ndarray, 
        transform_type: tr.TransformType, 
        TOL: float, 
        nanifoutside: bool, 
        delta: int, 
        verbose: bool
        ) -> (np.ndarray, float, np.ndarray, np.ndarray):
    """
    Inverse compositional algorithm
    Quadratic version - L2 norm

    Parameters:
    - I1: First image.
    - I2: Second image.
    - p: Initial Parameters of the transform (may be not null if we iterate on the function call).
    - transform_type (TransformType): The type of transformation.
    - TOL: Tolerance used for the convergence in the iterations.
    - nanifoutside: If True, the pixels outside the image are considered as NaN.
    - delta: The maximal distance to boundary to consider the pixel as NaN.
    - verbose: Enable verbose mode.

    Returns: 
    - p:updated parameters of the transform,
    - error: error value,
    - DI: error image,
    - Iw: warped image.
    """
    # We suppose that I1 and I2 are RGB images with channels in the last dimension, if not we raise an error
    if len(I1.shape) != 3 or len(I2.shape) != 3 or I1.shape[2] != 3 or I2.shape[2] != 3:
        raise ValueError("I1 and I2 must be RGB images with channels in the last dimension")

    # Define nx, ny, nz from the shape of I1 and I2
    ny, nx, nz = I1.shape # suppose that I1 and I2 are not flattened

    # Verify the dimensions of I1 and I2
    if I1.shape != I2.shape:
        raise ValueError("I1 and I2 must have the same dimensions")

    # Sanity check on the value of TOL
    if TOL >= 0.01:
        raise ValueError("TOL must be positive and very small (less than 0.01)")
    
    # We force the images to be float64 to avoid problems with the computation accuracy
    if I1.dtype != np.float64 or I2.dtype != np.float64:
        I1 = I1.astype(np.float64)
        I2 = I2.astype(np.float64)

    nparams = transform_type.nparams()

    Ix = np.zeros(I1.shape, dtype=np.float64)  # x derivative of the first image
    Iy = np.zeros(I1.shape, dtype=np.float64)  # y derivative of the first image
    Iw = np.zeros(I1.shape, dtype=np.float64)  # warp of the second image
    DI = np.zeros(I1.shape, dtype=np.float64)  # error image (I2(w)-I1)
    DIJ = np.zeros((ny, nx, nz, nparams), dtype=np.float64)  # steepest descent images
    dp = np.zeros(nparams, dtype=np.float64)  # incremental solution
    b = np.zeros(nparams, dtype=np.float64)  # steepest descent images
    J = np.zeros((ny, nx, 2 * nparams), dtype=np.float64)  # jacobian matrix for all points
    H = np.zeros((nparams, nparams), dtype=np.float64)  # Hessian matrix
    H_1 = np.zeros((nparams, nparams), dtype=np.float64)  # inverse Hessian matrix

    # Evaluate the gradient of I1
    Ix[:, 1:-1, :] = 0.5 * (I1[:, 2:, :] - I1[:, :-2, :])
    Iy[1:-1, :, :] = 0.5 * (I1[2:, :, :] - I1[:-2, :, :])

    # Like in the modified version of the algorithm, we discard boundary pixels
    if (nanifoutside is True and delta > 0):
        Ix[:delta, :, :] = np.nan
        Ix[-delta:, :, :] = np.nan
        Ix[:, :delta, :] = np.nan
        Ix[:, -delta:, :] = np.nan
        Iy[:delta, :, :] = np.nan
        Iy[-delta:, :, :] = np.nan
        Iy[:, :delta, :] = np.nan
        Iy[:, -delta:, :] = np.nan
    
    # Evaluate the Jacobian: values of the jacobian are not in range [0., 1.]
    J = de.jacobian(transform_type, nx, ny)
    
    # Compute the steepest descent images
    DIJ = io.steepest_descent_images(Ix, Iy, J, nparams)
    
    # Compute the Hessian matrix
    H = de.hessian(DIJ) 
    H_1 = de.inverse_hessian(H, nparams) 
    
    # Iterate
    error = 1E10
    niter = 0
    
    while error > TOL and niter < cts.MAX_ITER:
        # Warp image I2 to compute I2w
        Iw = bi.bicubic_interpolation_skimage(I2, p, transform_type, nanifoutside, delta) 
        
        # Compute the error image (I1-I2w)
        DI = Iw - I1
        
        # Compute the independent vector
        b = io.independent_vector(DIJ, DI, nparams)
        
        # Solve equation and compute increment of the motion 
        error, dp = io.parametric_solve(H_1, b, nparams) 
        
        # Update the warp x'(x;p) := x'(x;p) * x'(x;dp)^-1
        p = tr.update_transform(p, dp, transform_type)
        
        if verbose:
            print(f"Iteration {niter}: |Dp|={error}: p=(", end="")
            for i in range(nparams - 1):
                print(f"{p[i]} ", end="")
            print(f"{p[nparams - 1]})")
        
        niter += 1
    
    return p, error, DI, Iw

def robust_inverse_compositional_algorithm(
    I1: np.ndarray,    # first image
    I2: np.ndarray,    # second image
    p: np.ndarray,     # parameters of the transform (output, all in input if we iterate on the function call)
    transform_type: tr.TransformType,   # transform type
    TOL: float,    # Tolerance used for the convergence in the iterations
    robust_type: io.RobustErrorFunctionType, # type (RobustErrorFunctionType) of robust error function
    lambda_: float, # parameter of robust error function
    nanifoutside: bool, # if True, the pixels outside the image are considered as NaN
    delta: int, # maximal distance to boundary to consider the pixel as NaN
    verbose: bool  # enable verbose mode
    ) -> (np.ndarray, float, np.ndarray, np.ndarray):
    """
    Robust Inverse Compositional Algorithm for image alignment.

    Parameters:
    - I1: First image.
    - I2: Second image.
    - p: Initial Parameters of the transform (may be not null if we iterate on the function call).
    - transform_type (TransformType): The type of transformation.
    - TOL: Tolerance used for the convergence in the iterations.
    - robust: Robust error function.
    - lambda_: Parameter of the robust error function.
    - nanifoutside: If True, the pixels outside the image are considered as NaN.
    - delta: The maximal distance to boundary to consider the pixel as NaN.
    - verbose: Enable verbose mode.

    Returns: 
    - p:updated parameters of the transform,
    - error: error value,
    - DI: error image,
    - Iw: warped image.
    """
    # Define nx, ny, nz from the shape of I1 and I2
    ny, nx, nz = I1.shape

    # Verify the dimensions of I1 and I2
    if I1.shape != I2.shape:
        raise ValueError("I1 and I2 must have the same dimensions")

    # Sanity check on the value of TOL
    if TOL >= 0.01:
        raise ValueError("TOL must be positive and very small (less than 0.01)")

    #TODO: if images are colored and processing requested in grey scale, we must convert them to grey scale

    # We force the images to be float64 to avoid problems with the computation accuracy
    if I1.dtype != np.float64 or I2.dtype != np.float64:
        I1 = I1.astype(np.float64)
        I2 = I2.astype(np.float64)

    nparams = transform_type.nparams()

    Ix = np.zeros(I1.shape, dtype=np.float64)  # x derivative of the first image
    Iy = np.zeros(I1.shape, dtype=np.float64)  # y derivative of the first image
    Iw = np.zeros(I1.shape, dtype=np.float64)  # warp of the second image
    DI = np.zeros(I1.shape, dtype=np.float64)  # error image (I2(w)-I1)
    DIJ = np.zeros((ny, nx, nz, nparams), dtype=np.float64)  # steepest descent images
    dp = np.zeros(nparams, dtype=np.float64)  # incremental solution
    b = np.zeros(nparams, dtype=np.float64)  # steepest descent images
    J = np.zeros((ny, nx, 2 * nparams), dtype=np.float64)  # jacobian matrix for all points
    H = np.zeros((nparams, nparams), dtype=np.float64)  # Hessian matrix
    H_1 = np.zeros((nparams, nparams), dtype=np.float64)  # inverse Hessian matrix

    # Evaluate the gradient of I1
    Ix[:, 1:-1, :] = 0.5 * (I1[:, 2:, :] - I1[:, :-2, :])
    Iy[1:-1, :, :] = 0.5 * (I1[2:, :, :] - I1[:-2, :, :])

    # Like in the modified version of the algorithm, we discard boundary pixels
    if (nanifoutside is True and delta > 0):
        Ix[:delta, :, :] = np.nan
        Ix[-delta:, :, :] = np.nan
        Ix[:, :delta, :] = np.nan
        Ix[:, -delta:, :] = np.nan
        Iy[:delta, :, :] = np.nan
        Iy[-delta:, :, :] = np.nan
        Iy[:, :delta, :] = np.nan
        Iy[:, -delta:, :] = np.nan

    # Evaluate the Jacobian
    J = de.jacobian(transform_type, nx, ny)

    # Compute the steepest descent images
    DIJ = io.steepest_descent_images(Ix, Iy, J, nparams)

    # Iterate
    error = 1E10
    niter = 0
    lambda_it = lambda_ if lambda_ > 0 else cts.LAMBDA_0

    while error > TOL and niter < cts.MAX_ITER:
        # Warp image I2 to compute I2w
        Iw = bi.bicubic_interpolation_skimage(I2, p, transform_type, nanifoutside, delta) 

        # Compute the error image (I1-I2w)
        DI = Iw - I1

        # Compute robustification function
        rho = io.robust_error_function(DI, lambda_it, robust_type)

        if lambda_ <= 0 and lambda_it > cts.LAMBDA_N:
            lambda_it *= cts.LAMBDA_RATIO
            if lambda_it < cts.LAMBDA_N:
                lambda_it = cts.LAMBDA_N

        # Compute the independent vector
        b = io.independent_vector_robust(DIJ, DI, rho, nparams)

        # Compute the Hessian matrix
        H = de.hessian_robust(DIJ, rho, nparams)
        H_1 = de.inverse_hessian(H, nparams)

        # Solve equation and compute increment of the motion
        error, dp = io.parametric_solve(H_1, b, nparams)

        # Update the warp x'(x;p) := x'(x;p) * x'(x;dp)^-1
        p = tr.update_transform(p, dp, transform_type)

        if verbose:
            print(f"|Dp|={error}: p=(", end="")
            for i in range(nparams - 1):
                print(f"{p[i]} ", end="")
            print(f"{p[nparams - 1]}), lambda_={lambda_it}")

        niter += 1

    return p, error, DI, Iw
 

def pyramidal_inverse_compositional_algorithm(
    I1: np.ndarray,     # first image
    I2: np.ndarray,     # second image
    p: np.ndarray,      # parameters of the transform
    transform_type: tr.TransformType, # typeof transformation
    nscales: int, # number of scales
    nu: float,      # downsampling factor
    TOL: float,     # stopping criterion threshold
    robust_type: io.RobustErrorFunctionType,  # type of robust error function
    lambda_: float,  # parameter of robust error function
    nanifoutside: bool, # if True, the pixels outside the image are considered as NaN
    delta: int, # maximal distance to boundary to consider the pixel as NaN
    verbose: bool  # switch on messages
    ) -> (np.ndarray, float, np.ndarray, np.ndarray):
    """
    Performs the pyramidal inverse compositional algorithm for image alignment.

    Args:
    - I1: First image.
    - I2: Second image.
    - p: Parameters of the transform.
    - transform_type: type of transformation to recover.
    - nscales: Number of scales.
    - nu: Downsampling factor.
    - TOL: Stopping criterion threshold.
    - robust_type: type of Robust error function.
    - lambda_: Parameter of robust error function.
    - verbose: Switch on messages.

    Returns:
    - p:updated parameters of the transform,
    - error: error value,
    - DI: error image,
    - Iw: warped image.
    """
    # We suppose that I1 and I2 are RGB images with channels in the last dimension, if not we raise an error
    if len(I1.shape) != 3 or len(I2.shape) != 3 or I1.shape[2] != 3 or I2.shape[2] != 3:
        raise ValueError("I1 and I2 must be RGB images with channels in the last dimension")
    # Define nx, ny, nz from the shape of I1 and I2
    nyy, nxx, nzz = I1.shape

    # Verify the dimensions of I1 and I2
    if I1.shape != I2.shape:
        raise ValueError("I1 and I2 must have the same dimensions")

    # Sanity check on the value of TOL
    if TOL >= 0.01:
        raise ValueError("TOL must be positive and very small (less than 0.01)")

    # We force the images to be float64 to avoid problems with the computation accuracy
    if I1.dtype != np.float64 or I2.dtype != np.float64:
        I1 = I1.astype(np.float64)
        I2 = I2.astype(np.float64)
        
    nparams = transform_type.nparams()
    I1s = [np.zeros((nyy, nxx, nzz), dtype=np.float64)]
    I2s = [np.zeros((nyy, nxx, nzz), dtype=np.float64)]
    ps = np.zeros((nscales, nparams), dtype=np.float64)
    nx = np.zeros(nscales, dtype=np.float64)
    ny = np.zeros(nscales, dtype=np.float64)

    I1s[0] = I1
    I2s[0] = I2
    ps[0] = np.copy(p)
    nx[0] = nxx
    ny[0] = nyy

    for s in range(1, nscales):
        nx[s], ny[s] = zm.zoom_size(nx[s-1], ny[s-1], nu)
        I1s.append(rescale(I1s[s-1], nu, mode='constant', cval=0, order=3, #bicubic interpolation
                            anti_aliasing=True, channel_axis=2, preserve_range=True))
        I2s.append(rescale(I2s[s-1], nu, mode='constant', cval=0, order=3, #bicubic interpolation
                            anti_aliasing=True, channel_axis=2, preserve_range=True))
        ps[s] = np.zeros(nparams, dtype=np.float64)

    # Function implementation...
    for s in range(nscales-1, -1, -1):
        if verbose:
            print(f"Scale: {s}")
        if robust_type == io.RobustErrorFunctionType.QUADRATIC:
            if verbose:
                print("(L2 norm)")
            ps[s], error, DI, Iw = inverse_compositional_algorithm(
                I1=I1s[s], 
                I2=I2s[s], 
                p=ps[s], 
                transform_type=transform_type, 
                TOL=TOL, 
                nanifoutside=nanifoutside, 
                delta=delta, 
                verbose=verbose
            )
        else:
            if verbose:
                print(f"(Robust error function {robust_type})")
            ps[s], error, DI, Iw = robust_inverse_compositional_algorithm(
                I1=I1s[s], 
                I2=I2s[s], 
                p=ps[s], 
                transform_type=transform_type, 
                TOL=TOL, 
                robust_type=robust_type, 
                lambda_=lambda_, 
                nanifoutside=nanifoutside,
                delta=delta,
                verbose=verbose
            )
        if s > 0:
            ps[s-1] = zm.zoom_in_parameters(ps[s], transform_type, nx[s], ny[s], nx[s-1], ny[s-1])

    return ps[0], error, DI, Iw
