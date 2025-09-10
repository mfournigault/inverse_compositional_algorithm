import tensorflow as tf


# @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
#                                 tf.TensorSpec(shape=[], dtype=tf.float32),
#                                 tf.TensorSpec(shape=[], dtype=tf.int32)])
def tf_rhop(t2, lambda_, type_) -> tf.Tensor:
    """
    Derivative of robust error functions, TensorFlow version.
    
    Parameters:
        t2: squared difference of both images, now a mtrix  
        lambda_: robust threshold
        type_: choice of the robust error function
        
    Returns:
        tf.Tensor: result of the robust error function derivative
    """
    lambda2 = lambda_ * lambda_
    def quadratic():  #RobustErrorFunctionType.QUADRATIC:
        return tf.ones_like(t2)
    def truncated_quadratic():  # RobustErrorFunctionType.TRUNCATED_QUADRATIC:
        return tf.where(t2 < lambda2, tf.ones_like(t2), tf.zeros_like(t2))
    def german_mcclure():  # RobustErrorFunctionType.GERMAN_MCCLURE:
        return lambda2 / tf.square(lambda2 + t2)
    def lorentzian():  # RobustErrorFunctionType.LORENTZIAN:
        return 1.0 / (lambda2 + t2)
    def charbonnier():  # RobustErrorFunctionType.CHARBONNIER:
        return 1.0 / tf.sqrt(t2 + lambda2)
    def error_fn():
        def raise_error():
            raise ValueError("Unsupported transformation type")
        return tf.py_function(raise_error, [], Tout=tf.float32)
    
    branch_fns = {
        0: quadratic,
        1: truncated_quadratic,
        2: german_mcclure,
        3: lorentzian,
        4: charbonnier
    }

    return tf.switch_case(type_, branch_fns=branch_fns, default=error_fn)


# @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
#                                 tf.TensorSpec(shape=[], dtype=tf.float32),
#                                 tf.TensorSpec(shape=[], dtype=tf.int32)])
def tf_robust_error_function(DI, lambda_, type_) -> tf.Tensor:
    """
    Apply a robust function to the difference of images.
    Parameters:
        DI: 3D tensor [ny, nx, nz] of image differences.
        lambda_: seuil robuste.
        type_: type de fonction robuste.
    Returns:
        tf.Tensor: 2D tensor [ny, nx] of robust error function values.
    """
    # Replace NaN values with zeros
    DI_filled = tf.where(tf.math.is_finite(DI), DI, tf.zeros_like(DI))
    # Compute the squared sum of the differences
    t2 = tf.reduce_sum(tf.square(DI_filled), axis=-1)
    t2 = tf.where(tf.math.is_finite(t2), t2, tf.zeros_like(t2))
    rho = tf_rhop(t2, lambda_, type_)

    return rho


# @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
#                               tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
#                               tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
def tf_steepest_descent_images(Ix, Iy, J) -> tf.Tensor:
    """
    Calculate the steepest descent images DI^t*J for optimization.
    Args:
        Ix (tf.Tensor): The gradient of the image along the x-axis with shape (b, ny, nx, nz).
        Iy (tf.Tensor): The gradient of the image along the y-axis with shape (b, ny, nx, nz).
        J (tf.Tensor): The Jacobian matrix with shape (b, ny, nx, 2*m), where m = nparams/2.
    Returns:
        tf.Tensor: The steepest descent images with shape (b, ny, nx, nz, m).
    """
    Jx, Jy = tf.split(J, num_or_size_splits=2, axis=-1)  # Chaque tenseur a la taille (b, ny, nx, m)

    # Expand Ix and Iy along the last dimension to be able to multiply with Jx and Jy.
    Ix_exp = tf.expand_dims(Ix, axis=-1)  # (b, ny, nx, nz, 1)
    Iy_exp = tf.expand_dims(Iy, axis=-1)  # (b, ny, nx, nz, 1)

    # Reshape Jx et Jy to get a "duplicate" for nz.
    Jx_exp = tf.expand_dims(Jx, axis=2)  # (b, ny, 1, nx, m) 
    Jy_exp = tf.expand_dims(Jy, axis=2)  # (b, ny, 1, nx, m)

    Jx_exp = tf.reshape(Jx, [tf.shape(Jx)[0], tf.shape(Jx)[1], tf.shape(Jx)[2], 1, tf.shape(Jx)[3]])
    Jy_exp = tf.reshape(Jy, [tf.shape(Jy)[0], tf.shape(Jy)[1], tf.shape(Jy)[2], 1, tf.shape(Jy)[3]])

    # DIJ is defined by the sum of the two contributions
    DIJ = Ix_exp * Jx_exp + Iy_exp * Jy_exp  # (b, ny, nx, nz, m)

    return DIJ

