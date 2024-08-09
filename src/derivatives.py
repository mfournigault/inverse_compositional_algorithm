import numpy as np
from matrix_operators import AtA, sAtA
from transformation import TransformType


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

    J = np.zeros((2 * nx * ny, nparams))

    match transform_type:
        case TransformType.TRANSLATION:
            for i in range(nx * ny):
                c = 2 * i * nparams
                J[c] = 1.0
                J[c + 1] = 0.0
                J[c + 2] = 0.0
                J[c + 3] = 1.0

        case TransformType.EUCLIDEAN:
            for i in range(ny):
                for j in range(nx):
                    c = 2 * (i * nx + j) * nparams
                    J[c] = 1.0
                    J[c + 1] = 0.0
                    J[c + 2] = -i
                    J[c + 3] = 0.0
                    J[c + 4] = 1.0
                    J[c + 5] = j

        case TransformType.SIMILARITY:
            for i in range(ny):
                for j in range(nx):
                    c = 2 * (i * nx + j) * nparams
                    J[c] = 1.0
                    J[c + 1] = 0.0
                    J[c + 2] = j
                    J[c + 3] = -i
                    J[c + 4] = 0.0
                    J[c + 5] = 1.0
                    J[c + 6] = i
                    J[c + 7] = j

        case TransformType.AFFINITY:
            for i in range(ny):
                for j in range(nx):
                    c = 2 * (i * nx + j) * nparams
                    J[c] = 1.0
                    J[c + 1] = 0.0
                    J[c + 2] = j
                    J[c + 3] = i
                    J[c + 4] = 0.0
                    J[c + 5] = 0.0
                    J[c + 6] = 0.0
                    J[c + 7] = 1.0
                    J[c + 8] = 0.0
                    J[c + 9] = 0.0
                    J[c + 10] = j
                    J[c + 11] = i

        case TransformType.HOMOGRAPHY:
            for i in range(ny):
                for j in range(nx):
                    c = 2 * (i * nx + j) * nparams
                    J[c] = j
                    J[c + 1] = i
                    J[c + 2] = 1.0
                    J[c + 3] = 0.0
                    J[c + 4] = 0.0
                    J[c + 5] = 0.0
                    J[c + 6] = -j * j
                    J[c + 7] = -j * i
                    J[c + 8] = 0.0
                    J[c + 9] = 0.0
                    J[c + 10] = 0.0
                    J[c + 11] = j
                    J[c + 12] = i
                    J[c + 13] = 1.0
                    J[c + 14] = -j * i
                    J[c + 15] = -i * i

    return J

    
def hessian(DIJ, nparams, nx, ny, nz):
    """
    Function to compute the Hessian matrix.
    The Hessian is equal to DIJ^T * DIJ.
    """
    # Initialize the Hessian to zero
    H = np.zeros((nparams, nparams))
    
    # Calculate the Hessian in a neighbor window
    for i in range(ny):
        for j in range(nx):
            DIJ_slice = DIJ[(i * nx + j) * nz * nparams : (i * nx + j + 1) * nz * nparams]
            H += AtA(DIJ_slice, nz, nparams)
    
    return H

def hessian_robust(DIJ, rho, nparams, nx, ny, nz):
    """
    Function to compute the Hessian matrix with robust error functions.
    The Hessian is equal to rho' * DIJ^T * DIJ.
    """
    # Initialize the Hessian to zero
    H = np.zeros((nparams, nparams))
    
    # Calculate the Hessian in a neighbor window
    for i in range(ny):
        for j in range(nx):
            DIJ_slice = DIJ[(i * nx + j) * nz * nparams : (i * nx + j + 1) * nz * nparams]
            H += sAtA(rho[i * nx + j], DIJ_slice, nz, nparams)
    
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
    H_1 = np.zeros((nparams, nparams))
    try:
        H_1 = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        # if the matrix is not invertible, set parameters to 0
        H_1 = np.zeros((nparams, nparams))
    return H_1

