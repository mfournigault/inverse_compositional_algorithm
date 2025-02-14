import tensorflow as tf

from transformation import TransformType

@tf.function    
def tf_compute_gradients(I1: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """
    By using the central difference, computes the gradients of the input image
    tensor I1 along the x and y axes.

    Args:
        I1 (tf.Tensor): A 4D tensor representing the input image with shape 
                        (batch_size, height, width, channels).

    Returns:
        tuple: A tuple containing two 4D tensors:
            - Ix (tf.Tensor): The gradient of the input image along the x-axis.
            - Iy (tf.Tensor): The gradient of the input image along the y-axis.
    """
    Ix = tf.zeros_like(I1)
    Iy = tf.zeros_like(I1)
    Ix = (I1[:, :, 2:, :] - I1[:, :, :-2, :]) * 0.5
    Ix = tf.pad(Ix, [[0,0], [0,0], [1,1], [0,0]])
    Iy = (I1[:, 2:, :, :] - I1[:, :-2, :, :]) * 0.5
    Iy = tf.pad(Iy, [[0,0], [1,1], [0,0], [0,0]])
    return Ix, Iy


@tf.function
def tf_jacobian(
            transform_type: TransformType,
            nx: int,
            ny: int) -> tf.Tensor:
    """
    Compute the Jacobian matrix for the specified transformation type.

    The Jacobian matrix is computed based on the transformation type and the 
    dimensions of the input image. The supported transformation types are:
    - TRANSLATION
    - EUCLIDEAN
    - SIMILARITY
    - AFFINITY
    - HOMOGRAPHY
    Args:
        transform_type (TransformType): The type of transformation.
        nx (int): The number of columns in the image grid.
        ny (int): The number of rows in the image grid.

    Returns:
        tf.Tensor: A tensor representing the Jacobian matrix with an added 
        batch dimension.
    """
    x = tf.range(nx, dtype=tf.float32)
    y = tf.range(ny, dtype=tf.float32)
    X, Y = tf.meshgrid(x, y)
    ones = tf.ones_like(X)
    zeros = tf.zeros_like(X)
    if transform_type == TransformType.TRANSLATION:
        J = tf.stack([ones, zeros, zeros, ones], axis=-1)
    elif transform_type == TransformType.EUCLIDEAN:
        J = tf.stack([ones, zeros, -Y, zeros, ones, X], axis=-1)
    elif transform_type == TransformType.SIMILARITY:
        J = tf.stack([ones, zeros, X, -Y, zeros, ones, Y, X], axis=-1)
    elif transform_type == TransformType.AFFINITY:
        J = tf.stack([ones, zeros, X, Y, zeros, zeros, zeros, ones, zeros, zeros, X, Y], axis=-1)
    elif transform_type == TransformType.HOMOGRAPHY:
        J = tf.stack([X, Y, ones, zeros, zeros, zeros, -X*X, -X*Y, zeros, zeros, zeros, X, Y, ones, -X*Y, -Y*Y], axis=-1)
    else:
        raise ValueError("Unsupported transformation type")
    
    return tf.expand_dims(J, 0)  # Add batch dimension

