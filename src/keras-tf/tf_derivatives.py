import tensorflow as tf


# @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),))
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


# @tf.function(
#     input_signature=[
#         tf.TensorSpec(shape=[], dtype=tf.int32),  # transform_type value as int
#         tf.TensorSpec(shape=[], dtype=tf.int32),  # nx
#         tf.TensorSpec(shape=[], dtype=tf.int32)   # ny
#     ]
# )
def tf_jacobian(transform_type_value, nx, ny) -> tf.Tensor:    
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
        transform_type_value (tf.Tensor): An integer tensor representing the transformation type.
        nx (tf.Tensor): The number of columns in the image grid.
        ny (tf.Tensor): The number of rows in the image grid.

    Returns:
        tf.Tensor: A tensor representing the Jacobian matrix with an added 
        batch dimension.
    """
    x = tf.range(tf.cast(nx, dtype=tf.float32), dtype=tf.float32)
    y = tf.range(tf.cast(ny, dtype=tf.float32), dtype=tf.float32)
    X, Y = tf.meshgrid(x, y)
    ones = tf.ones_like(X)
    zeros = tf.zeros_like(X)
    
    def jac_translation():
        return tf.stack([ones, zeros, zeros, ones], axis=-1)
    def jac_euclidean():
        return tf.stack([ones, zeros, -Y, zeros, ones, X], axis=-1)
    def jac_similarity():
        return tf.stack([ones, zeros, X, -Y, zeros, ones, Y, X], axis=-1)
    def jac_affinity():
        return tf.stack([ones, zeros, X, Y, zeros, zeros, zeros, ones, zeros, zeros, X, Y], axis=-1)
    def jac_homography():
        return tf.stack([X, Y, ones,
                         zeros, zeros, zeros,
                         -X*X, -X*Y,
                         zeros, zeros, zeros,
                         X, Y, ones,
                         -X*Y, -Y*Y], axis=-1)
    
    branch_fns = {
        0: jac_translation,
        1: jac_euclidean,
        2: jac_similarity,
        3: jac_affinity,
        4: jac_homography,
    }
    
    # Use the integer tensor directly with tf.switch_case:
    J = tf.switch_case(branch_index=transform_type_value, branch_fns=branch_fns)
    return tf.expand_dims(J, 0)  # Add batch dimension

