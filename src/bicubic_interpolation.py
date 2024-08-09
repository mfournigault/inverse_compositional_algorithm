import numpy as np
import transformation as tf

def neumann_bc(x: int, nx: int) -> tuple[int, bool]:
    """
    Apply Neumann boundary conditions to the given value.

    Args:
        x (int): The value to apply the boundary conditions to.
        nx (int): The maximum value allowed.

    Returns:
        tuple[int, bool]: A tuple containing the modified value and a flag indicating if the value was modified due to the boundary conditions.
    """
    out = False
    if x < 0:
        if x < -2:
            out = True
        x = 0
    elif x >= nx:
        if x > nx + 1:
            out = True
        x = nx - 1
    return x, out


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


def bicubic_interpolation_point(input, uu, vv, nx, ny, nz, k, border_out):
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
        border_out (bool): Flag indicating whether to return 0 for points outside the image.

    Returns:
        float: The interpolated value at the given point.
    """
    sx = -1 if uu < 0 else 1
    sy = -1 if vv < 0 else 1
    out = False
    
    x = neumann_bc(int(uu), nx, out)
    y = neumann_bc(int(vv), ny, out)
    mx = neumann_bc(int(uu) - sx, nx, out)
    my = neumann_bc(int(vv) - sy, ny, out)
    dx = neumann_bc(int(uu) + sx, nx, out)
    dy = neumann_bc(int(vv) + sy, ny, out)
    ddx = neumann_bc(int(uu) + 2 * sx, nx, out)
    ddy = neumann_bc(int(vv) + 2 * sy, ny, out)
    
    if out and border_out:
        return 0
    else:
        p11 = input[(mx  + nx * my) * nz + k]
        p12 = input[(x   + nx * my) * nz + k]
        p13 = input[(dx  + nx * my) * nz + k]
        p14 = input[(ddx + nx * my) * nz + k]
        p21 = input[(mx  + nx * y) * nz + k]
        p22 = input[(x   + nx * y) * nz + k]
        p23 = input[(dx  + nx * y) * nz + k]
        p24 = input[(ddx + nx * y) * nz + k]
        p31 = input[(mx  + nx * dy) * nz + k]
        p32 = input[(x   + nx * dy) * nz + k]
        p33 = input[(dx  + nx * dy) * nz + k]
        p34 = input[(ddx + nx * dy) * nz + k]
        p41 = input[(mx  + nx * ddy) * nz + k]
        p42 = input[(x   + nx * ddy) * nz + k]
        p43 = input[(dx  + nx * ddy) * nz + k]
        p44 = input[(ddx + nx * ddy) * nz + k]
        
        pol = np.array([
            [p11, p21, p31, p41],
            [p12, p22, p32, p42],
            [p13, p23, p33, p43],
            [p14, p24, p34, p44]
        ])
        
        return bicubic_interpolation_array(pol, uu - x, vv - y)


def bicubic_interpolation_image(input, output, params, transform_type, nx, ny, nz, border_out):
    
    nparams = transform_type.nparams()
    
    for i in range(ny):
        for j in range(nx):
            p = i * nx + j
            x, y = tf.project(j, i, params, nparams)
            
            for k in range(nz):
                if border_out and (x < 0 or x >= nx or y < 0 or y >= ny):
                    output[p * nz + k] = 0
                else:
                   #TODO: replace with the function from scipy.ndimage (map_coordinates) 
                    output[p * nz + k] = bicubic_interpolation_point(input, x, y, nx, ny, nz, k, border_out)
