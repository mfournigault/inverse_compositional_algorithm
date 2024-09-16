import numpy as np
from enum import Enum
from numba import jit

from matrix_operators import AtA, sAtA, Atb, sAtb
import utils


# Définition de l'énumération pour les types de fonctions robustes
class RobustErrorFunctionType(Enum):
    QUADRATIC = 0
    TRUNCATED_QUADRATIC = 1
    GERMAN_MCCLURE = 2
    LORENTZIAN = 3
    CHARBONNIER = 4

def rhop(t2, lambda_, type_):
    """
    Derivative of robust error functions

    Parameters:
    t2 (float): squared difference of both images, now a mtrix  
    lambda_ (float): robust threshold
    type_ (int): choice of the robust error function

    Returns:
    float: result of the robust error function derivative
    """
    lambda2 = lambda_ * lambda_
    result = 0.0

    match type_:
        case RobustErrorFunctionType.QUADRATIC:
            result = np.ones_like(t2)
        case RobustErrorFunctionType.TRUNCATED_QUADRATIC:
            if t2 < lambda2:
                result = np.ones_like(t2)
            else:
                result = np.zeros_like(t2)
        case RobustErrorFunctionType.GERMAN_MCCLURE:
            result = lambda2 / ((lambda2 + t2) * (lambda2 + t2))
        case RobustErrorFunctionType.LORENTZIAN:
            result = 1.0 / (lambda2 + t2)
        case RobustErrorFunctionType.CHARBONNIER:
            result = 1.0 / np.sqrt(t2 + lambda2)
        case _:
            raise ValueError("Unknown type for robust error function")

    return result


def robust_error_function(DI, lambda_, type_):
    """
    Function to store the values of p'((I2(x'(x;p))-I1(x))²)

    Parameters:
    DI (numpy.ndarray): input difference array
    lambda_ (float): threshold used in the robust functions
    type_ (int): choice of robust error function

    Returns:
    numpy.ndarray: output robust function array
    """
    ny, nx, nz = DI.shape 
    rho = np.zeros((ny, nx), dtype=np.float64)
    rho = np.where(np.isfinite(DI), DI, 0.0)
    # rho = np.linalg.norm(rho, ord=2, axis=2) # in the original code, it is not strictly a norm of order 2, as they keep the square of the norm
    rho = np.einsum("ijc,ijc->ij", rho, rho)
    rho = np.where(np.isfinite(rho), rhop(rho, lambda_, type_), 0.0)

    # for i in range(ny):
    #     for j in range(nx):
    #         if utils.valid_values(DI[i, j, :]):
    #             norm = 0.0
    #             for c in range(nz):
    #                 norm += DI[i, j, c] * DI[i, j, c]
    #             rho[i, j] = rhop(norm, lambda_, type_)
    #         else:
    #             rho[i, j] = 0.0
    
    return rho


def independent_vector(DIJ, DI, nparams):
    """
    Function to compute b=Sum(DIJ^t * DI)

    Parameters:
    DIJ (numpy.ndarray): the steepest descent image
    DI (numpy.ndarray): I2(x'(x;p))-I1(x)
    nparams (int): number of parameters

    Returns:
    numpy.ndarray: output independent vector
    """
    ny, nx, nz = DI.shape # suppose that DI is not flattened
    b = np.zeros(nparams, dtype=np.float64)

    # Create masked arrays with NaN values masked
    # DIJ_masked = np.ma.masked_where(~np.isfinite(DIJ), DIJ)
    # DI_masked = np.ma.masked_where(~np.isfinite(DI), DI)

    # # Fill masked values with zeros
    # DIJ_filled = DIJ_masked.filled(0)
    # DI_filled = DI_masked.filled(0)

    DIJ_filled = np.where(np.isfinite(DIJ), DIJ, 0)
    DI_filled = np.where(np.isfinite(DI), DI, 0)

    # Vectorized computation using einsum for efficiency
    DIJt = np.einsum("ijlk->ijkl", DIJ_filled)
    prod = np.einsum("ijkl,ijl->ijk", DIJt, DI_filled)
    b = np.einsum("ijl->l", prod)

    # Convert the masked array b to a regular NumPy array
    # b_unmasked = b.filled(0)
    # print("b shape: ", b_unmasked.shape)

    return b


def independent_vector_robust(DIJ, DI, rho, nparams):
    """
    Function to compute b=Sum(rho'*DIJ^t * DI) with robust error functions

    Parameters:
    DIJ (numpy.ndarray): the steepest descent image
    DI (numpy.ndarray): I2(x'(x;p))-I1(x)
    rho (numpy.ndarray): robust function
    nparams (int): number of parameters

    Returns:
    numpy.ndarray: output independent vector
    """
    #TODO: remove params nx, ny, nz and define them from the shape of DIJ   
    #TODO: sanity check on the dimensions of DIJ and DI, they should be compatible 
    ny, nx, nz = DI.shape # suppose that DI is not flattened
    b = np.zeros(nparams, dtype=np.float64)

    DIJ_filled = np.where(np.isfinite(DIJ), DIJ, 0)
    DI_filled = np.where(np.isfinite(DI), DI, 0)

    # Vectorized computation using einsum for efficiency
    DIJt = np.einsum("ijlk->ijkl", DIJ_filled)
    prod = np.einsum("ijkl,ijl->ijk", DIJt, DI_filled)
    prod = np.einsum("ij,ijk->ijk", rho, prod)
    b = np.einsum("ijl->l", prod)
    # for i in range(ny):
    #     for j in range(nx):
    #         try:
    #             if utils.valid_values(DIJ[i, j, :, :]) and utils.valid_values(DI[i, j, :]):
    #                 b += rho[i, j] * DIJ[i, j, :, :].T @ DI[i, j, :]

    #         except IndexError:
    #             print(f"IndexError: i={i}, j={j}, DIJ.shape={DIJ.shape}, DI.shape={DI.shape}")
    #             raise

    return b


def parametric_solve(H_1, b, nparams):
    # Convert inputs to numpy arrays
    # H_1 = np.array(H_1).reshape((nparams, nparams))
    # b = np.array(b, dtype=np.float64)
    
    # Perform matrix-vector multiplication
    # dp = np.dot(H_1, b)
    dp = H_1 @ b
    
    # Calculate the error
    error = np.sum(dp**2)
    
    return np.sqrt(error), dp


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def steepest_descent_images(Ix, Iy, J, nparams):
    """
    Calculate the steepest descent images DI^t*J for optimization.

    Args:
        Ix (ndarray): The x-derivative of the input image.
        Iy (ndarray): The y-derivative of the input image.
        J (ndarray): The Jacobian matrix.
        nparams (int): The number of parameters of the transformation.

    Returns:
        ndarray: The steepest descent images.

    """
    # Define nx, ny, nz from the shape of Ix 
    ny, nx, nz = Ix.shape # suppose that Ix and Iy are not flattened

    # Verify the dimensions of Ix and Iy
    if Ix.shape != Iy.shape:
        raise ValueError("Ix and Iy must have the same dimensions")


    # Initialize the output array
    DIJ = np.zeros((ny, nx, nz, nparams), dtype=np.float64)
    
    # Vectorized computation
    for c in range(nz):
        for n in range(nparams):
            DIJ[:, :, c, n] = Ix[:, :, c] * J[:, :, n] + Iy[:, :, c] * J[:, :, n + nparams]
    
    return DIJ
