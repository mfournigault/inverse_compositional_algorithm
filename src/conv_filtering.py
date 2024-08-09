import numpy as np

def gradient(input: np.ndarray, nx: int, ny: int, nz: int):
    """
    Compute the gradient of an input array.

    Parameters:
    - input (np.ndarray): The input array.
    - nx (int): The number of columns in the input array.
    - ny (int): The number of rows in the input array.
    - nz (int): The number of color channels in the input array.

    Returns:
    - dx (np.ndarray): The gradient in the x-direction.
    - dy (np.ndarray): The gradient in the y-direction.
    """

   #TODO: remove params nx, ny, nz and define them from the shape of input
   #TODO: remove the reshaping of the input array?
    
    input = input.reshape((ny, nx, nz))
    dx = np.zeros_like(input)
    dy = np.zeros_like(input)
    
   #TODO: validate the gradient computation with some test on images 
    for index_color in range(nz):
        # Gradient in the center body of the image
        dx[1:ny-1, 1:nx-1, index_color] = 0.5 * (input[1:ny-1, 2:nx, index_color] - input[1:ny-1, 0:nx-2, index_color])
        dy[1:ny-1, 1:nx-1, index_color] = 0.5 * (input[2:ny, 1:nx-1, index_color] - input[0:ny-2, 1:nx-1, index_color])
        
        # Gradient in the first and last rows
        dx[0, 1:nx-1, index_color] = 0.5 * (input[0, 2:nx, index_color] - input[0, 0:nx-2, index_color])
        dy[0, 1:nx-1, index_color] = 0.5 * (input[1, 1:nx-1, index_color] - input[0, 1:nx-1, index_color])
        dx[ny-1, 1:nx-1, index_color] = 0.5 * (input[ny-1, 2:nx, index_color] - input[ny-1, 0:nx-2, index_color])
        dy[ny-1, 1:nx-1, index_color] = 0.5 * (input[ny-1, 1:nx-1, index_color] - input[ny-2, 1:nx-1, index_color])
        
        # Gradient in the first and last columns
        dx[1:ny-1, 0, index_color] = 0.5 * (input[1:ny-1, 1, index_color] - input[1:ny-1, 0, index_color])
        dy[1:ny-1, 0, index_color] = 0.5 * (input[2:ny, 0, index_color] - input[0:ny-2, 0, index_color])
        dx[1:ny-1, nx-1, index_color] = 0.5 * (input[1:ny-1, nx-1, index_color] - input[1:ny-1, nx-2, index_color])
        dy[1:ny-1, nx-1, index_color] = 0.5 * (input[2:ny, nx-1, index_color] - input[0:ny-2, nx-1, index_color])
        
        # Gradient in the corners
        dx[0, 0, index_color] = 0.5 * (input[0, 1, index_color] - input[0, 0, index_color])
        dy[0, 0, index_color] = 0.5 * (input[1, 0, index_color] - input[0, 0, index_color])
        
        dx[0, nx-1, index_color] = 0.5 * (input[0, nx-1, index_color] - input[0, nx-2, index_color])
        dy[0, nx-1, index_color] = 0.5 * (input[1, nx-1, index_color] - input[0, nx-1, index_color])
        
        dx[ny-1, 0, index_color] = 0.5 * (input[ny-1, 1, index_color] - input[ny-1, 0, index_color])
        dy[ny-1, 0, index_color] = 0.5 * (input[ny-1, 0, index_color] - input[ny-2, 0, index_color])
        
        dx[ny-1, nx-1, index_color] = 0.5 * (input[ny-1, nx-1, index_color] - input[ny-1, nx-2, index_color])
        dy[ny-1, nx-1, index_color] = 0.5 * (input[ny-1, nx-1, index_color] - input[ny-2, nx-1, index_color])
    
    return dx, dy


def gaussian(I, xdim, ydim, zdim, sigma, bc, precision):
    """
    Applies Gaussian smoothing to the input image.

    Args:
        I (ndarray): Input image.
        xdim (int): Width of the input image.
        ydim (int): Height of the input image.
        zdim (int): Number of channels in the input image.
        sigma (float): Standard deviation of the Gaussian kernel.
        bc (int): Boundary condition type. 0 for Dirichlet, 1 for Reflecting, 2 for Periodic.
        precision (float): Precision factor for determining the size of the kernel.

    Returns:
        ndarray: Smoothed image.

    Raises:
        ValueError: If the sigma value is too large for the specified boundary condition.

    """
   #TODO: remove params xdim, ydim, zdim and define them from the shape of I 

    den = 2 * sigma * sigma
    size = int(precision * sigma) + 1
    bdx = xdim + size
    bdy = ydim + size

    if bc and size > xdim:
        raise ValueError("GaussianSmooth: sigma too large for this bc")

    # Compute the coefficients of the 1D convolution kernel
    B = np.array([1 / (sigma * np.sqrt(2.0 * np.pi)) * np.exp(-i * i / den) for i in range(size)])
    norm = np.sum(B) * 2 - B[0]
    B /= norm

    R = np.zeros(size + xdim + size)
    T = np.zeros(size + ydim + size)

    # Loop for every channel
    for index_color in range(zdim):
        # Convolution of each line of the input image
        for k in range(ydim):
            R[size:bdx] = I[k * xdim:(k + 1) * xdim, index_color]
            if bc == 0:  # Dirichlet boundary conditions
                R[:size] = R[bdx:] = 0
            elif bc == 1:  # Reflecting boundary conditions
                R[:size] = I[k * xdim + size - np.arange(size), index_color]
                R[bdx:] = I[k * xdim + xdim - 1 - np.arange(size), index_color]
            elif bc == 2:  # Periodic boundary conditions
                R[:size] = I[k * xdim + xdim - size + np.arange(size), index_color]
                R[bdx:] = I[k * xdim + np.arange(size), index_color]

            for i in range(size, bdx):
                sum_val = B[0] * R[i]
                for j in range(1, size):
                    sum_val += B[j] * (R[i - j] + R[i + j])
                I[k * xdim + i - size, index_color] = sum_val

        # Convolution of each column of the input image
        for k in range(xdim):
            T[size:bdy] = I[size * xdim + k::xdim, index_color]
            if bc == 0:  # Dirichlet boundary conditions
                T[:size] = T[bdy:] = 0
            elif bc == 1:  # Reflecting boundary conditions
                T[:size] = I[(size - np.arange(size)) * xdim + k, index_color]
                T[bdy:] = I[(ydim - 1 - np.arange(size)) * xdim + k, index_color]
            elif bc == 2:  # Periodic boundary conditions
                T[:size] = I[(ydim - size + np.arange(size)) * xdim + k, index_color]
                T[bdy:] = I[np.arange(size) * xdim + k, index_color]

            for i in range(size, bdy):
                sum_val = B[0] * T[i]
                for j in range(1, size):
                    sum_val += B[j] * (T[i - j] + T[i + j])
                I[(i - size) * xdim + k, index_color] = sum_val

    return I