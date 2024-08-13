import numpy as np
from scipy.ndimage import sobel

import derivatives as de
import image_optimisation as io
import transformation as tr
# import conv_filtering as cf
import bicubic_interpolation as bi
import zoom as zm
import constants as cts

 
def inverse_compositional_algorithm(I1, I2, p, transform_type, TOL, verbose):
    """
    Inverse compositional algorithm
    Quadratic version - L2 norm

    :param I1: First image, a numpy array of shape (ny, nx, nz).
    :param I2: Second image, a numpy array of shape (ny, nx, nz).
    :param p: Initial transformation parameters (may be not null if we iterate on the function call).
    :param transform_type (TransformType): The type of transformation.
    :param TOL: Tolerance used for the convergence in the iterations.
    :param verbose: Enable verbose mode.
    
    :return: The updated transformation parameters.
    """
    # Define nx, ny, nz from the shape of I1 and I2
    ny, nx, nz = I1.shape # suppose that I1 and I2 are not flattened

    # Verify the dimensions of I1 and I2
    if I1.shape != I2.shape:
        raise ValueError("I1 and I2 must have the same dimensions")

    # Sanity check on the value of TOL
    if TOL >= 0.01:
        raise ValueError("TOL must be positive and very small (less than 0.01)")
    
    nparams = transform_type.nparams()

    #TODO: correction all the function and subfunctions to work with non flat images and matrices
    size1 = nx * ny * nz
    size2 = size1 * nparams
    size3 = nparams * nparams
    size4 = 2 * nx * ny * nparams

    Ix = np.zeros(I1.shape)  # x derivative of the first image
    Iy = np.zeros(I1.shape)  # y derivative of the first image
    Iw = np.zeros(I1.shape)  # warp of the second image
    DI = np.zeros(I1.shape)  # error image (I2(w)-I1)
    DIJ = np.zeros((ny, nx, nz, nparams))  # steepest descent images
    dp = np.zeros(nparams)  # incremental solution
    b = np.zeros(nparams)  # steepest descent images
    J = np.zeros((ny, nx, 2 * nparams))  # jacobian matrix for all points
    H = np.zeros((nparams, nparams))  # Hessian matrix
    H_1 = np.zeros((nparams, nparams))  # inverse Hessian matrix

    # Evaluate the gradient of I1
    for channel in range(nz):
        Ix[:, :, channel] = sobel(I1[:, :, channel], axis=1)
        Iy[:, :, channel] = sobel(I1[:, :, channel], axis=0)
    
    # Evaluate the Jacobian
    J = de.jacobian(nparams, nx, ny)
    
    # Compute the steepest descent images
    # Ix, Iy are supposed to be flattened
    DIJ = io.steepest_descent_images(Ix, Iy, J, nparams, nx, ny, nz)
    # DIJ is flattened
    
    # Compute the Hessian matrix
    H = de.hessian(DIJ, nparams, nx, ny, nz) # H is not flattened
    H_1 = de.inverse_hessian(H, nparams) # H_1 is not flattened
    
    # Iterate
    error = 1E10
    niter = 0
    
    while error > TOL and niter < cts.MAX_ITER:
        # Warp image I2
        bi.bicubic_interpolation_image(I2, Iw, p, transform_type, nx, ny, nz)
        
        # Compute the error image (I1-I2w)
        # difference_image(I1, Iw, DI, nx, ny, nz)
        DI = Iw - I1
        
        # Compute the independent vector
        b = io.independent_vector(DIJ, DI, nparams, nx, ny, nz) # b is flattened
        
        # Solve equation and compute increment of the motion 
        error, dp = io.parametric_solve(H_1, b, nparams) # H_1 is not flattened, b is flattened
        
        # Update the warp x'(x;p) := x'(x;p) * x'(x;dp)^-1
        p = tr.update_transform(p, dp, transform_type)
        
        if verbose:
            print(f"|Dp|={error}: p=(", end="")
            for i in range(nparams - 1):
                print(f"{p[i]} ", end="")
            print(f"{p[nparams - 1]})")
        
        niter += 1
    
    return p

def robust_inverse_compositional_algorithm(
    I1,    # first image
    I2,    # second image
    p,     # parameters of the transform (output, all in input if we iterate on the function call)
    transform_type,   # transform type
    nx,        # number of columns of the image
    ny,        # number of rows of the image
    nz,        # number of channels of the images
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
    - nx: Number of columns of the image.
    - ny: Number of rows of the image.
    - nz: Number of channels of the images.
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

    nparams = transform_type.nparams()
    size0 = nx * ny
    size1 = nx * ny * nz
    size2 = size1 * nparams
    size3 = nparams * nparams
    size4 = 2 * nx * ny * nparams

    #TODO: correction all the function and subfunctions to work with non flat images and matrices
    Ix = np.zeros(size1)   # x derivative of the first image
    Iy = np.zeros(size1)   # y derivative of the first image
    Iw = np.zeros(size1)   # warp of the second image
    DI = np.zeros(size1)   # error image (I2(w)-I1)
    DIJ = np.zeros(size2)  # steepest descent images
    dp = np.zeros(nparams) # incremental solution
    b = np.zeros(nparams)  # steepest descent images
    J = np.zeros(size4)    # jacobian matrix for all points
    H = np.zeros(size3)    # Hessian matrix
    H_1 = np.zeros(size3)  # inverse Hessian matrix
    rho = np.zeros(size0)  # robust function

    # Evaluate the gradient of I1
    for channel in range(nz):
        Ix[:, :, channel] = sobel(I1[:, :, channel], axis=1)
        Iy[:, :, channel] = sobel(I1[:, :, channel], axis=0)

    # Evaluate the Jacobian
    J = de.jacobian(transform_type, nx, ny)

    # Compute the steepest descent images
    DIJ = io.steepest_descent_images(Ix, Iy, J, nparams, nx, ny, nz)

    # Iterate
    error = 1E10
    niter = 0
    lambda_it = lambda_ if lambda_ > 0 else cts.LAMBDA_0

    while error > TOL and niter < cts.MAX_ITER:
        # Warp image I2
        bi.bicubic_interpolation(I2, Iw, p, transform_type, nx, ny, nz)

        # Compute the error image (I1-I2w)
        # difference_image(I1, Iw, DI, nx, ny, nz)
        DI = Iw - I1

        # Compute robustification function
        rho = io.robust_error_function(DI, lambda_it, robust_type, nx, ny, nz)

        if lambda_ <= 0 and lambda_it > cts.LAMBDA_N:
            lambda_it *= cts.LAMBDA_RATIO
            if lambda_it < cts.LAMBDA_N:
                lambda_it = cts.LAMBDA_N

        # Compute the independent vector
        b = io.independent_vector_robust(DIJ, DI, rho, nparams, nx, ny, nz)

        # Compute the Hessian matrix
        H = de.hessian_robust(DIJ, rho, nparams, nx, ny, nz)
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

    return p
 

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
    # Define nx, ny, nz from the shape of I1 and I2
    nyy, nxx, nzz = I1.shape

    # Verify the dimensions of I1 and I2
    if I1.shape != I2.shape:
        raise ValueError("I1 and I2 must have the same dimensions")

    # Sanity check on the value of TOL
    if TOL >= 0.01:
        raise ValueError("TOL must be positive and very small (less than 0.01)")
    
    #TODO: correction all the function and subfunctions to work with non flat images and matrices
    nparams = transform_type.nparams()
    size = nxx * nyy * nzz
    I1s = np.zeros((nscales, size))
    I2s = np.zeros((nscales, size))
    ps = np.zeros((nscales, nparams))
    nx = np.zeros(nscales)
    ny = np.zeros(nscales)

    # I1s[0] = np.copy(I1).reshape((nyy, nxx, nzz))
    temp = np.copy(I1).reshape((nyy, nxx, nzz))
    I1s[0] = temp
    # I2s[0] = np.copy(I2).reshape((nyy, nxx, nzz))
    temp = np.copy(I2).reshape((nyy, nxx, nzz))
    I2s[0] = temp
    ps[0] = np.copy(p)
    nx[0] = nxx
    ny[0] = nyy

    for i in range(nparams):
        p[i] = 0.0

    for s in range(1, nscales):
        nx[s], ny[s] = zm.zoom_size(nx[s-1], ny[s-1], nu)
        I1s[s] = zm.zoom_out(I1s[s-1], nx[s-1], ny[s-1], nzz, nu)
        I2s[s] = zm.zoom_out(I2s[s-1], nx[s-1], ny[s-1], nzz, nu)
        ps[s] = np.zeros(nparams)

    # Function implementation...
    for s in range(nscales-1, -1, -1):
        ps[s] = np.zeros(nparams)
        if verbose:
            print(f"Scale: {s}")
        if robust_type == 'QUADRATIC':
            if verbose:
                print("(L2 norm)")
            inverse_compositional_algorithm(
                I1s[s], I2s[s], ps[s], transform_type, TOL, verbose
            )
        else:
            if verbose:
                print(f"(Robust error function {robust_type})")
            robust_inverse_compositional_algorithm(
                I1s[s], I2s[s], ps[s], transform_type, TOL, robust_type, lambda_, verbose
            )
        if s > 0:
            ps[s-1] = zm.zoom_in_parameters(ps[s], transform_type, nx[s], ny[s], nx[s-1], ny[s-1])

    return ps[0]