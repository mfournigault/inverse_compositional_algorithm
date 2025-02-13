import tensorflow as tf

from image_optimisation import RobustErrorFunctionType


def tf_rhop(
            t2: tf.Tensor,
            lambda_: float,
            type_: RobustErrorFunctionType
        ) -> tf.Tensor:
    """
    Derivative of robust error functions, TensorFlow version.
    
    Parameters:
        t2 (tf.Tensor): squared difference of both images, now a mtrix  
        lambda_ (float): robust threshold
        type_ (int): choice of the robust error function
        
    Returns:
        tf.Tensor: result of the robust error function derivative
    """
    lambda2 = lambda_ * lambda_
    if type_ == RobustErrorFunctionType.QUADRATIC:
        result = tf.ones_like(t2)
    elif type_ == RobustErrorFunctionType.TRUNCATED_QUADRATIC:
        result = tf.where(t2 < lambda2, tf.ones_like(t2), tf.zeros_like(t2))
    elif type_ == RobustErrorFunctionType.GERMAN_MCCLURE:
        result = lambda2 / tf.square(lambda2 + t2)
    elif type_ == RobustErrorFunctionType.LORENTZIAN:
        result = 1.0 / (lambda2 + t2)
    elif type_ == RobustErrorFunctionType.CHARBONNIER:
        result = 1.0 / tf.sqrt(t2 + lambda2)
    else:
        raise ValueError("Unknown type for robust error function")
    return result

def tf_robust_error_function(
            DI: tf.Tensor,
            lambda_: float,
            type_: RobustErrorFunctionType
        ) -> tf.Tensor:
    """
    Apply a robust function to the difference of images.
    Parameters:
        DI (tf.Tensor): 3D tensor [ny, nx, nz] of image differences.
        lambda_ (float): seuil robuste.
        type_ (RobustErrorFunctionType): type de fonction robuste.
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


@tf.function
def tf_steepest_descent_images(Ix: tf.Tensor, 
                               Iy: tf.Tensor, 
                               J: tf.Tensor) -> tf.Tensor:
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

