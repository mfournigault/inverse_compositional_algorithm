import numpy as np
from numba import jit, prange

import transformation as tf

@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def neumann_bc(x: int, nx: int) -> tuple[int, bool]:
    """
    Apply Neumann boundary conditions to the given value.

    Args:
        x (int): The value to apply the boundary conditions to.
        nx (int): The maximum value allowed.

    Returns:
        tuple[int, bool]: A tuple containing the modified value and a flag indicating if the value was modified due to the boundary conditions.
    """
    if x < 0:
        x = 0
    elif x >= nx:
        x = nx - 1
    return x


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def cubic_interpolation(v: np.ndarray, x: float) -> float:
    """
    Performs cubic interpolation using the given control points and the interpolation parameter.

    Parameters:
        v (np.ndarray): The array of control points.
        x (float): The interpolation parameter.

    Returns:
        float: The interpolated value.

    """
    return v[1] + 0.5 * x * (v[2] - v[0]
                             + x * (2.0 * v[0] - 5.0 * v[1] + 4.0 * v[2] - v[3]
                                    + x * (3.0 * (v[1] - v[2]) + v[3] - v[0])))


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def bicubic_interpolation_array(p, x, y):
    """
    Bicubic interpolation in two dimensions.

    Parameters:
    p (np.ndarray): 4x4 array containing the interpolation points
    x (float): x position to be interpolated
    y (float): y position to be interpolated

    Returns:
    float: Interpolated value
    """
    v = np.zeros(4)
    v[0] = cubic_interpolation(p[0], y)
    v[1] = cubic_interpolation(p[1], y)
    v[2] = cubic_interpolation(p[2], y)
    v[3] = cubic_interpolation(p[3], y)
    return cubic_interpolation(v, x)


@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def bicubic_interpolation_point(input, uu, vv, nx, ny, nz, k):
    """
    Performs bicubic interpolation at a given point.

    Args:
        input (ndarray): Input array of shape (nx * ny * nz,) containing the image data.
        uu (float): The x-coordinate of the point to interpolate.
        vv (float): The y-coordinate of the point to interpolate.
        nx (int): The width of the image.
        ny (int): The height of the image.
        nz (int): The number of channels in the image.
        k (int): The channel index to interpolate.

    Returns:
        float: The interpolated value at the given point.
    """
    sx = -1 if uu < 0 else 1
    sy = -1 if vv < 0 else 1
    
    x = neumann_bc(int(uu), nx)
    y = neumann_bc(int(vv), ny)
    mx = neumann_bc(int(uu) - sx, nx)
    my = neumann_bc(int(vv) - sy, ny)
    dx = neumann_bc(int(uu) + sx, nx)
    dy = neumann_bc(int(vv) + sy, ny)
    ddx = neumann_bc(int(uu) + 2 * sx, nx)
    ddy = neumann_bc(int(vv) + 2 * sy, ny)
    
    p11 = input[my, mx, k]
    p12 = input[my, x, k]
    p13 = input[my, dx, k]
    p14 = input[my, ddx, k]
    p21 = input[y, mx, k]
    p22 = input[y, x, k]
    p23 = input[y, dx, k]
    p24 = input[y, ddx, k]
    p31 = input[dy, mx, k]
    p32 = input[dy, x, k]
    p33 = input[dy, dx, k]
    p34 = input[dy, ddx, k]
    p41 = input[ddy, mx, k]
    p42 = input[ddy, x, k]
    p43 = input[ddy, dx, k]
    p44 = input[ddy, ddx, k]
    
    pol = np.array([
        [p11, p21, p31, p41],
        [p12, p22, p32, p42],
        [p13, p23, p33, p43],
        [p14, p24, p34, p44]
    ])
    
    return bicubic_interpolation_array(pol, uu - x, vv - y)

@jit(nopython=True, fastmath=True, nogil=True, cache=True, parallel=True)
def bicubic_interpolation_image(
    input, 
    params,
    nparams,
    # transform_type, 
    nanifoutside, 
    delta):
    
    ny, nx, nz = input.shape
    output = np.zeros((ny, nx, nz), dtype=np.float64)

    # nparams = transform_type.nparams()

    if nanifoutside:
        out_value = np.nan
    else:
        out_value = 0.0
    
    for i in prange(ny):
        for j in prange(nx):
            p = i * nx + j
            # x, y = tf.project(j, i, params, transform_type)
            x, y = tf.project(j, i, params, nparams)
            out = (x < delta) or (x > nx - 1 - delta) or (y < delta) or (y > ny - 1 - delta)

            if out:
                output[i, j, :] = out_value
            else:
                for k in range(nz):
                    output[i, j, k] = bicubic_interpolation_point(input, x, y, nx, ny, nz, k)
    
    return output


