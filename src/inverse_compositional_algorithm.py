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

 
def inverse_compositional_algorithm(I1, I2, p, transform_type, nanifoutside, delta, TOL, verbose):
    """
    Inverse compositional algorithm
    Quadratic version - L2 norm

    :param I1: First image, a numpy array of shape (ny, nx, nz).
    :param I2: Second image, a numpy array of shape (ny, nx, nz).
    :param p: Initial transformation parameters (may be not null if we iterate on the function call).
    :param transform_type (TransformType): The type of transformation.
    :param nanifoutside: If True, the pixels outside the image are considered as NaN.
    :param delta: The maximal distance to boundary to consider the pixel as NaN.
    :param TOL: Tolerance used for the convergence in the iterations.
    :param verbose: Enable verbose mode.
    
    :return: The updated transformation parameters.
    """
    #TODO: remove this constraint to alow the processing of grey scale images
    #TODO: if images are colored and processing requested in grey scale, we must convert them to grey scale
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
        # As later in the code we make use of libraries like skimage that supposes all float images to be in
        # the range [0., 1.], we must scale the images to this range
        # I1 = rescale_intensity(I1, in_range=(0, 255), out_range=(0, 1))
        # I2 = rescale_intensity(I2, in_range=(0, 255), out_range=(0, 1))

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
    # Value range of Ix and Iy must not be in [0., 1.] as for J
    DIJ = io.steepest_descent_images(Ix, Iy, J, nparams)
    # DIJ is flattened
    
    # Compute the Hessian matrix
    H = de.hessian(DIJ) # H is not flattened
    H_1 = de.inverse_hessian(H, nparams) # H_1 is not flattened
    
    # Iterate
    error = 1E10
    niter = 0
    
    while error > TOL and niter < cts.MAX_ITER:
        # Warp image I2
        # Iw = bi.bicubic_interpolation_image(I2, p, transform_type.nparams(), nanifoutside, delta) 
        Iw = bi.bicubic_interpolation_skimage(I2, p, transform_type, nanifoutside, delta) 
        
        # Compute the error image (I1-I2w)
        # difference_image(I1, Iw, DI, nx, ny, nz)
        DI = Iw - I1
        
        # Compute the independent vector
        b = io.independent_vector(DIJ, DI, nparams) # b is flattened
        
        # Solve equation and compute increment of the motion 
        error, dp = io.parametric_solve(H_1, b, nparams) # H_1 is not flattened, b is flattened
        
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
    I1,    # first image
    I2,    # second image
    p,     # parameters of the transform (output, all in input if we iterate on the function call)
    transform_type,   # transform type
    nanifoutside, 
    delta, 
    TOL,    # Tolerance used for the convergence in the iterations
    robust_type, # type (RobustErrorFunctionType) of robust error function
    lambda_, # parameter of robust error function
    verbose  # enable verbose mode
):
    """
    Robust Inverse Compositional Algorithm for image alignment.

    Parameters:
    - I1: First image.
    - I2: Second image.
    - p: Initial Parameters of the transform (may be not null if we iterate on the function call).
    - transform_type (TransformType): The type of transformation.
    - nanifoutside: If True, the pixels outside the image are considered as NaN.
    - delta: The maximal distance to boundary to consider the pixel as NaN.
    - TOL: Tolerance used for the convergence in the iterations.
    - robust: Robust error function.
    - lambda_: Parameter of the robust error function.
    - verbose: Enable verbose mode.

    Returns: updated parameters of the transform.
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
        # Warp image I2
        Iw = bi.bicubic_interpolation_image(I2, p, transform_type, nanifoutside, delta) 

        # Compute the error image (I1-I2w)
        # difference_image(I1, Iw, DI, nx, ny, nz)
        DI = Iw - I1

        # Compute robustification function
        #TODO: correct this function to work with non flat images and matrices 
        rho = io.robust_error_function(DI, lambda_it, robust_type)

        if lambda_ <= 0 and lambda_it > cts.LAMBDA_N:
            lambda_it *= cts.LAMBDA_RATIO
            if lambda_it < cts.LAMBDA_N:
                lambda_it = cts.LAMBDA_N

        # Compute the independent vector
        #TODO: correct this function to work with non flat images and matrices
        b = io.independent_vector_robust(DIJ, DI, rho, nparams)

        # Compute the Hessian matrix
        #TODO: correct this function to work with non flat images and matrices
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
            print(f"{p[nparams - 1]}), lambda={lambda_it}")

        niter += 1

    return p, error, DI, Iw
 

def pyramidal_inverse_compositional_algorithm(
    I1,     # first image
    I2,     # second image
    p,      # parameters of the transform
    transform_type, # typeof transformation
    nscales, # number of scales
    nu,      # downsampling factor
    TOL,     # stopping criterion threshold
    robust_type,  # type of robust error function
    lambda_,  # parameter of robust error function
    verbose  # switch on messages
):
    """
    Performs the pyramidal inverse compositional algorithm for image alignment.

    Args:
        I1: First image.
        I2: Second image.
        p: Parameters of the transform.
        transform_type: type of transformation to recover.
        nscales: Number of scales.
        nu: Downsampling factor.
        TOL: Stopping criterion threshold.
        robust_type: type of Robust error function.
        lambda_: Parameter of robust error function.
        verbose: Switch on messages.

    Returns:
        Updated parameters of the transform.
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
        # As later in the code we make use of libraries like skimage that supposes all float images to be in
        # the range [0., 1.], we must scale the images to this range
        # I1 = rescale_intensity(I1, in_range=(0, 255), out_range=(0, 1))
        # I2 = rescale_intensity(I2, in_range=(0, 255), out_range=(0, 1))
        
    #TODO: correction all the function and subfunctions to work with non flat images and matrices
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
        #TODO: replace zoom_out by skimage.transform.rescale -> done
        # I1s[s] = zm.zoom_out(I1s[s-1], nx[s-1], ny[s-1], nzz, nu)
        I1s.append(rescale(I1s[s-1], nu, mode='constant', cval=0, anti_aliasing=True, channel_axis=2, preserve_range=True))
        # I2s[s] = zm.zoom_out(I2s[s-1], nx[s-1], ny[s-1], nzz, nu)
        I2s.append(rescale(I2s[s-1], nu, mode='constant', cval=0, anti_aliasing=True, channel_axis=2, preserve_range=True))
        ps[s] = np.zeros(nparams, dtype=np.float64)

    # Function implementation...
    for s in range(nscales-1, -1, -1):
        if verbose:
            print(f"Scale: {s}")
        if robust_type == io.RobustErrorFunctionType.QUADRATIC:
            if verbose:
                print("(L2 norm)")
            ps[s], error, DI, Iw = inverse_compositional_algorithm(
                I1s[s], I2s[s], ps[s], transform_type, True, 10, TOL, verbose
            )
        else:
            if verbose:
                print(f"(Robust error function {robust_type})")
            robust_inverse_compositional_algorithm(
                I1s[s], I2s[s], ps[s], transform_type, TOL, robust_type, lambda_, verbose
            )
        if s > 0:
            ps[s-1] = zm.zoom_in_parameters(ps[s], transform_type, nx[s], ny[s], nx[s-1], ny[s-1])
            # print("ps[%d] = ".format(s), ps[s])
            # print("ps[%d] = ".format(s-1), ps[s-1])

    return ps[0], error, DI, Iw
