import numpy as np
from matrix_operators import AtA, sAtA
from numba import jit
from transformation import TransformType
import utils

def jacobian(transform_type, nx, ny):
    """
    Compute the Jacobian matrix for a given transform type.

    Args:
        transform_type (TransformType): The type of transformation.
        nx (int): The number of columns in the image grid.
        ny (int): The number of rows in the image grid.

    Returns:
        numpy.ndarray: The Jacobian matrix.

    Raises:
        ValueError: If the transform type is unknown.
    """
    nparams = transform_type.nparams()

    J = np.zeros((ny, nx, 2 * nparams), dtype=np.float64)

    y, x = np.mgrid[0:ny, 0:nx]

    match transform_type:
        case TransformType.TRANSLATION:
            J[:, :, 0] = 1.0
            J[:, :, 3] = 1.0

        case TransformType.EUCLIDEAN:
            J[:, :, 0] = 1.0
            J[:, :, 2] = -y
            J[:, :, 4] = 1.0
            J[:, :, 5] = x

        case TransformType.SIMILARITY:
            J[:, :, 0] = 1.0
            J[:, :, 2] = x
            J[:, :, 3] = -y
            J[:, :, 5] = 1.0
            J[:, :, 6] = y
            J[:, :, 7] = x

        case TransformType.AFFINITY:
            J[:, :, 0] = 1.0
            J[:, :, 2] = x
            J[:, :, 3] = y
            J[:, :, 7] = 1.0
            J[:, :, 10] = x
            J[:, :, 11] = y

        case TransformType.HOMOGRAPHY:
            J[:, :, 0] = x
            J[:, :, 1] = y
            J[:, :, 2] = 1.0
            J[:, :, 6] = -x * x
            J[:, :, 7] = -x * y
            J[:, :, 11] = x
            J[:, :, 12] = y
            J[:, :, 13] = 1.0
            J[:, :, 14] = -x * y
            J[:, :, 15] = -y * y

    return J


def hessian(DIJ):
    """
    Function to compute the Hessian matrix.
    The Hessian is equal to DIJ^T * DIJ.
    """
    ny, nx, nz, nparams = DIJ.shape[0], DIJ.shape[1], DIJ.shape[2], DIJ.shape[3]
    
    H = np.zeros((nparams, nparams), dtype=np.float64)

    DIJ_filled = np.where(np.isfinite(DIJ), DIJ, 0)
    DIJt = np.einsum("ijlk->ijkl", DIJ_filled)
    H = np.einsum("ijkl,ijlm->km", DIJt, DIJ_filled) # DIJ[ij,:,:].T @ DIJ[ij,:,:]

    return H


def hessian_robust(DIJ, rho, nparams):
    """
    Function to compute the Hessian matrix with robust error functions.
    The Hessian is equal to rho' * DIJ^T * DIJ.
    """
    ny, nx, nz, nparams = DIJ.shape[0], DIJ.shape[1], DIJ.shape[2], DIJ.shape[3]
    # Initialize the Hessian to zero
    H = np.zeros((nparams, nparams), dtype=np.float64)
    DIJ_filled = np.where(np.isfinite(DIJ), DIJ, 0)
    DIJt = np.einsum("ijlk->ijkl", DIJ_filled)
    H = np.einsum("ij,ijkl,ijlm->km", rho, DIJt, DIJ_filled) # rho[i, j] * DIJ[ij,:,:].T @ DIJ[ij,:,:]
    # Calculate the Hessian in a neighbor window
    # for i in range(ny):
    #     for j in range(nx):
    #         DIJ_slice = DIJ[i, j, :, :]
    #         # H += sAtA(rho[i * nx + j], DIJ_slice, nz, nparams)
    #         if utils.valid_values(DIJ_slice):
    #             H += rho[i, j] * DIJ_slice.T @ DIJ_slice
    
    return H


def inverse_hessian(H, nparams):
    """
    Function to compute the inverse of the Hessian

    Parameters:
    H (numpy.ndarray): input Hessian matrix
    nparams (int): number of parameters

    Returns:
    numpy.ndarray: inverse Hessian matrix
    """
    H_1 = np.zeros((nparams, nparams), dtype=np.float64)
    try:
        H_1 = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        # if the matrix is not invertible, set parameters to 0
        H_1 = np.zeros((nparams, nparams), dtype=np.float64)
    return H_1

