import numpy as np
from enum import Enum
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
    t2 (float): squared difference of both images  
    lambda_ (float): robust threshold
    type_ (int): choice of the robust error function

    Returns:
    float: result of the robust error function derivative
    """
    lambda2 = lambda_ * lambda_
    result = 0.0

    if type_ == RobustErrorFunctionType.QUADRATIC:
        result = 1.0
    elif type_ == RobustErrorFunctionType.TRUNCATED_QUADRATIC:
        if t2 < lambda2:
            result = 1.0
        else:
            result = 0.0
    elif type_ == RobustErrorFunctionType.GERMAN_MCCLURE:
        result = lambda2 / ((lambda2 + t2) * (lambda2 + t2))
    elif type_ == RobustErrorFunctionType.LORENTZIAN:
        result = 1.0 / (lambda2 + t2)
    elif type_ == RobustErrorFunctionType.CHARBONNIER:
        result = 1.0 / np.sqrt(t2 + lambda2)
    else:
        raise ValueError("Unknown type for robust error function")

    return result


def robust_error_function(DI, lambda_, type_, nx, ny, nz):
    """
    Function to store the values of p'((I2(x'(x;p))-I1(x))²)

    Parameters:
    DI (numpy.ndarray): input difference array
    lambda_ (float): threshold used in the robust functions
    type_ (int): choice of robust error function
    nx (int): number of columns
    ny (int): number of rows
    nz (int): number of channels

    Returns:
    numpy.ndarray: output robust function array
    """

   #TODO: remove params nx, ny, nz and define them from the shape of DI
    
    DI = np.asarray(DI)
    rho = np.zeros((ny, nx), dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            norm = 0.0
            for c in range(nz):
                norm += DI[i, j, c] * DI[i, j, c]
            rho[i, j] = rhop(norm, lambda_, type_)

    return rho



def independent_vector(DIJ, DI, nparams, nx, ny, nz):
    """
    Function to compute b=Sum(DIJ^t * DI)

    Parameters:
    DIJ (numpy.ndarray): the steepest descent image
    DI (numpy.ndarray): I2(x'(x;p))-I1(x)
    nparams (int): number of parameters
    nx (int): number of columns
    ny (int): number of rows
    nz (int): number of channels

    Returns:
    numpy.ndarray: output independent vector
    """
    b = np.zeros(nparams, dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            # DIJ, DI are supposed to not be flattened
            # b += Atb(
            #     DIJ[(i * nx + j) * nparams * nz : (i * nx + j + 1) * nparams * nz],
            #     DI[(i * nx + j) * nz : (i * nx + j + 1) * nz],
            #     nz, nparams
            # )
            try:
                # b += np.dot(DIJ[i, j, :, :].T, DI[i, j, :])
                if utils.valid_values(DIJ[i, j, :, :]) and utils.valid_values(DI[i, j, :]):
                    b += DIJ[i, j, :, :].T @ DI[i, j, :]

            except IndexError:
                print(f"IndexError: i={i}, j={j}, DIJ.shape={DIJ.shape}, DI.shape={DI.shape}")
                raise
            # b is flattened

    return b


def independent_vector_robust(DIJ, DI, rho, nparams, nx, ny, nz):
    """
    Function to compute b=Sum(rho'*DIJ^t * DI) with robust error functions

    Parameters:
    DIJ (numpy.ndarray): the steepest descent image
    DI (numpy.ndarray): I2(x'(x;p))-I1(x)
    rho (numpy.ndarray): robust function
    nparams (int): number of parameters
    nx (int): number of columns
    ny (int): number of rows
    nz (int): number of channels

    Returns:
    numpy.ndarray: output independent vector
    """
    #TODO: remove params nx, ny, nz and define them from the shape of DIJ   
    #TODO: sanity check on the dimensions of DIJ and DI, they should be compatible 

    b = np.zeros(nparams, dtype=np.float64)

    for i in range(ny):
        for j in range(nx):
            b += sAtb(
                rho[i * nx + j], 
                DIJ[(i * nx + j) * nparams * nz : (i * nx + j + 1) * nparams * nz],
                DI[(i * nx + j) * nz : (i * nx + j + 1) * nz],
                nz, nparams
            )

    return b


def parametric_solve(H_1, b, nparams):
    # Convert inputs to numpy arrays
    # H_1 = np.array(H_1).reshape((nparams, nparams))
    b = np.array(b, dtype=np.float64)
    
    # Perform matrix-vector multiplication
    # dp = np.dot(H_1, b)
    dp = H_1 @ b
    
    # Calculate the error
    error = np.sum(dp**2)
    
    return np.sqrt(error), dp


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
    
    for i in range(ny):
        for j in range(nx):
            for c in range(nz):
                for n in range(nparams):
                    # DIJ[k++]=Ix[p*nz+c]*J[2*p*nparams+n]+Iy[p*nz+c]*J[2*p*nparams+n+nparams];
                    DIJ[i, j, c, n] = Ix[i, j, c] * J[i, j, n] + Iy[i, j, c] * J[i, j, n + nparams]
    
    return DIJ
