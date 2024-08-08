import numpy as np

def AtA(DIJ_slice, nz, nparams):
    """
    Function to compute the multiplication of the transpose of a matrix and itself.
    """
    DIJ_matrix = DIJ_slice.reshape((nz, nparams))
    return DIJ_matrix.T @ DIJ_matrix

def sAtA(rho, DIJ_slice, nz, nparams):
    """
    Function to compute the multiplication of the transpose of a matrix and itself, scaled by a scalar.

    Parameters:
    rho (float): robust function value
    DIJ_slice (numpy.ndarray): slice of the steepest descent image
    nz (int): number of channels
    nparams (int): number of parameters

    Returns:
    numpy.ndarray: result of rho * (DIJ_matrix.T @ DIJ_matrix)
    """
    DIJ_matrix = DIJ_slice.reshape((nz, nparams))
    H = rho * (DIJ_matrix.T @ DIJ_matrix)
    return H

def Atb(DIJ, DI, nz, nparams):
    """
    Function to compute the multiplication of the transpose of a matrix and a vector

    Parameters:
    DIJ (numpy.ndarray): the steepest descent image
    DI (numpy.ndarray): I2(x'(x;p))-I1(x)
    nz (int): number of channels
    nparams (int): number of parameters

    Returns:
    numpy.ndarray: output independent vector
    """
    DIJ = DIJ.reshape((nparams, nz))
    DI = DI.reshape((nz, 1))
    b = np.dot(DIJ.T, DI).flatten()
    return b

def sAtb(rho, DIJ, DI, nz, nparams):
    """
    Function to compute the multiplication of the transpose of a matrix, a vector, and a scalar

    Parameters:
    rho (float): robust function value
    DIJ (numpy.ndarray): the steepest descent image
    DI (numpy.ndarray): I2(x'(x;p))-I1(x)
    nz (int): number of channels
    nparams (int): number of parameters

    Returns:
    numpy.ndarray: result of rho * DIJ.T * DI
    """
    DIJ = DIJ.reshape((nparams, nz))
    DI = DI.reshape((nz, 1))
    return rho * np.dot(DIJ.T, DI).flatten()