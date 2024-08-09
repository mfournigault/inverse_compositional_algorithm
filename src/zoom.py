import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates

from transformation import TransformType
import constants as cts

def zoom_size(nx: int, ny: int, factor: float) -> tuple[int, int]:
    """
    Calculates the new size after zooming an image.

    Args:
        nx (int): The original width of the image.
        ny (int): The original height of the image.
        factor (float): The zoom factor.

    Returns:
        tuple[int, int]: The new width and height after zooming the image.
    """
    nxx = int(np.round(nx * factor))
    nyy = int(np.round(ny * factor))
    return nxx, nyy


def bicubic_interpolation(image, x, y, nx, ny, nz, channel):
    coords = np.array([y, x])
    return map_coordinates(image[:, :, channel], coords, order=3, mode='nearest')

def zoom_out(I: np.ndarray, factor: float) -> np.ndarray:
    """
    Downsamples the image using Gaussian smoothing and bicubic interpolation.

    Args:
        I (np.ndarray): Input image with shape (ny, nx, nz).
        factor (float): Zoom factor between 0 and 1.

    Returns:
        np.ndarray: Zoomed image.
    """
    ny, nx, nz = I.shape
    nxx, nyy = zoom_size(nx, ny, factor)
    Iout = np.zeros((nyy, nxx, nz))

    # Compute the Gaussian sigma for smoothing
    sigma = cts.ZOOM_SIGMA_ZERO * np.sqrt(1.0 / (factor * factor) - 1.0)

    # Pre-smooth the image
    Is = np.copy(I)
    for channel in range(nz):
        Is[:, :, channel] = gaussian_filter(Is[:, :, channel], sigma=sigma)

    # Re-sample the image using bicubic interpolation
    for index_color in range(nz):
        for i1 in range(nyy):
            for j1 in range(nxx):
                i2 = i1 / factor
                j2 = j1 / factor
                Iout[i1, j1, index_color] = bicubic_interpolation(Is, j2, i2, nx, ny, nz, index_color)

    return Iout

def zoom_in_parameters(p: np.ndarray, transformation_type: TransformType, nx: int, ny: int, nxx: int, nyy: int) -> np.ndarray:
    """
    Upsamples the parameters of the transformation.

    Args:
        p (np.ndarray): Input parameters.
        transformation_type (TransformationType): Type of transformation.
        nx (int): Width of the original image.
        ny (int): Height of the original image.
        nxx (int): Width of the zoomed image.
        nyy (int): Height of the zoomed image.

    Returns:
        np.ndarray: Upsampled parameters.
    """
    # Compute the zoom factor
    factorx = nxx / nx
    factory = nyy / ny
    nu = max(factorx, factory)

    # Initialize the output parameters
    pout = np.zeros_like(p)

    # Adjust parameters based on the transformation type
    match transformation_type:
        case TransformType.TRANSLATION:
            pout[0] = p[0] * nu
            pout[1] = p[1] * nu
        case TransformType.EUCLIDEAN:
            pout[0] = p[0] * nu
            pout[1] = p[1] * nu
            pout[2] = p[2]
        case TransformType.SIMILARITY:
            pout[0] = p[0] * nu
            pout[1] = p[1] * nu
            pout[2] = p[2]
            pout[3] = p[3]
        case TransformType.AFFINITY:
            pout[0] = p[0] * nu
            pout[1] = p[1] * nu
            pout[2] = p[2]
            pout[3] = p[3]
            pout[4] = p[4]
            pout[5] = p[5]
        case TransformType.HOMOGRAPHY:
            pout[0] = p[0]
            pout[1] = p[1]
            pout[2] = p[2] * nu
            pout[3] = p[3]
            pout[4] = p[4]
            pout[5] = p[5] * nu
            pout[6] = p[6] / nu
            pout[7] = p[7] / nu
        case _:
            raise ValueError("Unsupported transformation type")

    return pout