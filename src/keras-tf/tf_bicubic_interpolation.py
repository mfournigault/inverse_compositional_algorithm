import tensorflow as tf


# @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),))
def cubic(x: tf.Tensor) -> tf.Tensor:
    """
    Computes the cubic interpolation function for the given input tensor.

    This function implements the cubic interpolation function based on Keys' 
    cubic convolution interpolation. It is used for smooth interpolation of 
    data points.

    Parameters:
    x (tf.Tensor): A tensor of input values for which the cubic interpolation 
                   function is to be computed.

    Returns:
    tf.Tensor: A tensor containing the result of applying the cubic interpolation 
               function to the input tensor.
    """
    absx = tf.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    # Fonction de base cubique de Keys
    return tf.where(absx <= 1,
                    (1.5 * absx3 - 2.5 * absx2 + 1.0),
                    tf.where(absx < 2,
                             (-0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0),
                             tf.zeros_like(x)))

# @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
#                               tf.TensorSpec(shape=[None, None, None], dtype=tf.int32),
#                               tf.TensorSpec(shape=[None, None, None], dtype=tf.int32)))
def get_pixel_value(img: tf.Tensor, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Retrieves the pixel values from the image tensor at the specified (x, y) coordinates.
    Args:
        img (tf.Tensor): A 4D tensor of shape [batch, H, W, C] representing the image batch.
        x (tf.Tensor): A 2D tensor of shape [batch, H, W] containing the x-coordinates.
        y (tf.Tensor): A 2D tensor of shape [batch, H, W] containing the y-coordinates.
    Returns:
        tf.Tensor: A tensor containing the pixel values at the specified coordinates, with shape [batch, H, W, C].
    """
    # img : [batch, H, W, C]
    batch_size = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    
    # Create a tensor of indices with shape [batch, H, W]
    batch_idx = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
    batch_idx = tf.tile(batch_idx, [1, H, W])
    
    # We stack indices to duplicate them for channels dim [batch, H, W, 3]
    indices = tf.stack([batch_idx, y, x], axis=-1)
    
    return tf.gather_nd(img, indices)

# @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
#                               tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)))
def bicubic_sampler(image: tf.Tensor, grid: tf.Tensor) -> tf.Tensor:
    """
    Perform bicubic interpolation on the given image using the provided grid.
    Args:
        image (tf.Tensor): A 4D tensor of shape [batch, H, W, C] representing the input image.
        grid (tf.Tensor): A 4D tensor of shape [batch, newH, newW, 2] containing the coordinates (y, x) for sampling.
    Returns:
        tf.Tensor: A 4D tensor of shape [batch, newH, newW, C] representing the interpolated image.
    Notes:
        - The function clips the grid values to avoid out-of-range values.
        - For each pixel, the function uses 16 neighbors to perform the bicubic interpolation.
        - The weights for each neighbor are computed using a cubic function.
        - The function accumulates the contributions of the 16 neighbors to produce the final interpolated image.
    """
    # image: [batch, H, W, C]
    # grid: [batch, newH, newW, 2] with coordinates (y, x)
    input_shape = tf.shape(image)
    batch_size, H, W, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    # grid shape is (batch, 3, H, W) 3 for x, y, 1
    grid_x = grid[:, 0, :, :]  
    grid_y = grid[:, 1, :, :]  

    # We clip the grid values to avoid out of range values
    x0 = tf.cast(tf.floor(grid_x), tf.int32)
    y0 = tf.cast(tf.floor(grid_y), tf.int32)

    # For each pixel, we need the 16 neighbors to perform the bicubic interpolation
    # So create 4 indices around (x0,y0) for each dim
    x_indexes = [x0 - 1, x0, x0 + 1, x0 + 2]
    y_indexes = [y0 - 1, y0, y0 + 1, y0 + 2]

    # Compute the weights for each neighbor
    x_diff = grid_x - tf.cast(x0, tf.float32)
    y_diff = grid_y - tf.cast(y0, tf.float32)

    weights_x = [cubic(x_diff + 1.0), cubic(x_diff), cubic(x_diff - 1.0), cubic(x_diff - 2.0)]
    weights_y = [cubic(y_diff + 1.0), cubic(y_diff), cubic(y_diff - 1.0), cubic(y_diff - 2.0)]

    # Convert list into tensors with shape [batch, d, 4]
    weights_x = tf.stack(weights_x, axis=-1)
    weights_y = tf.stack(weights_y, axis=-1)

    # Tensor with shape [batch, H, W, channels]
    output = tf.zeros([batch_size, tf.shape(grid)[2], tf.shape(grid)[3], channels], dtype=image.dtype)

    # Accumulation on the 16 neighbors
    for i in range(4):
        for j in range(4):
            x_i = x_indexes[i]
            y_j = y_indexes[j]
            # clip the indices to avoid out of range values
            x_i = tf.clip_by_value(x_i, 0, W - 1)
            y_j = tf.clip_by_value(y_j, 0, H - 1)
            pixel = get_pixel_value(image, x_i, y_j)  # [batch, newH, newW, channels]
            w = weights_y[:, :, :, j] * weights_x[:, :, :, i]
            output += pixel * tf.expand_dims(w, axis=-1)
    return output

