import tensorflow as tf

from transformation import TransformType
from tf_bicubic_interpolation import bicubic_sampler



# @tf.function(input_signature=(tf.TensorSpec(shape=[], dtype=tf.int32),))
def tf_nparams(transform_type_value):
    """
    Returns the number of parameters for the given transformation type.
    """
    branch_fns = {
        0: lambda: tf.constant(2, tf.int32),  # TransformType.TRANSLATION
        1: lambda: tf.constant(3, tf.int32),  # TransformType.EUCLIDEAN
        2: lambda: tf.constant(4, tf.int32),  # TransformType.SIMILARITY
        3: lambda: tf.constant(6, tf.int32),  # TransformType.AFFINITY
        4: lambda: tf.constant(8, tf.int32)   # TransformType.HOMOGRAPHY
    }
    def error_fn():
        def raise_error():
            raise ValueError("Unsupported transformation type")
        return tf.py_function(raise_error, [], Tout=tf.int32)

    return tf.switch_case(transform_type_value, branch_fns, default=error_fn)


@tf.function(reduce_retracing=True,
             input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),
                              tf.TensorSpec(shape=[], dtype=tf.int32)))
def tf_params2matrix(p, transform_type) -> tf.Tensor:
    """
    Converts the given parameters `p` into a transformation matrix based on the specified `transform_type`.

    Parameters:
        p (tf.Tensor): The parameters for the transformation.
        transform_type (TransformType): The type of transformation.

    Returns:
        tf.Tensor: A 3x3 transformation matrix.
    """
    dtype = p.dtype
    one = tf.cast(1, dtype)
    zero = tf.cast(0, dtype)
    
    def translation():  #TransformType.TRANSLATION:
        return tf.convert_to_tensor([
            [one, zero, p[0]],
            [zero, one, p[1]],
            [zero, zero, one]
        ], dtype=dtype)
    
    def euclidean():  #TransformType.EUCLIDEAN:
        return tf.convert_to_tensor([
            [tf.cos(p[2]), -tf.sin(p[2]), p[0]],
            [tf.sin(p[2]),  tf.cos(p[2]), p[1]],
            [zero,         zero,          one]
        ], dtype=dtype)
    
    def similarity():  #TransformType.SIMILARITY:
        return tf.convert_to_tensor([
            [one + p[2], -p[3],      p[0]],
            [p[3],       one + p[2], p[1]],
            [zero,       zero,       one]
        ], dtype=dtype)
    
    def affinity():  #TransformType.AFFINITY:
        return tf.convert_to_tensor([
            [one + p[2], p[3],       p[0]],
            [p[4],       one + p[5],  p[1]],
            [zero,       zero,        one]
        ], dtype=dtype)
    
    def homography():  #TransformType.HOMOGRAPHY:
        return tf.convert_to_tensor([
            [one + p[0], p[1],      p[2]],
            [p[3],       one + p[4], p[5]],
            [p[6],       p[7],      one]
        ], dtype=dtype)
    
    branch_fns = {
        0: translation,
        1: euclidean,
        2: similarity,
        3: affinity,
        4: homography
    }
    def error_fn():
        def raise_error():
            raise ValueError("Unsupported transformation type")
        return tf.py_function(raise_error, [], Tout=tf.float32)
    
    return tf.switch_case(transform_type, branch_fns, default=error_fn)


# @tf.function(input_signature=(tf.TensorSpec(shape=[], dtype=tf.int32),
#                               tf.TensorSpec(shape=[], dtype=tf.int32),
#                               tf.TensorSpec(shape=[], dtype=tf.int32),
#                               tf.TensorSpec(shape=[], dtype=tf.int32),
#                               tf.TensorSpec(shape=[None, None], dtype=tf.float32)))
def tf_transformed_grid(
            batch_size, 
            ny,
            nx, 
            transform_type, 
            p) -> tf.Tensor:
    """
    Transforms a grid of coordinates using a batch of affine transformation parameters.
    Args:
        batch_size (int): The number of elements in the batch.
        ny (int): The number of rows in the grid.
        nx (int): The number of columns in the grid.
        transform_type (TransformType): The type of geometric transformation.
        p (tf.Tensor): A tensor of shape (batch_size, num_params) containing the affine 
        transformation parameters for each element in the batch.
    Returns:
        tf.Tensor: A tensor of shape (batch_size, 3, nx, ny) containing the transformed 
        grid coordinates for each element in the batch.
    """
    x = tf.linspace(0.0, tf.cast(nx-1, dtype=tf.float32), nx)
    y = tf.linspace(0.0, tf.cast(ny-1, dtype=tf.float32), ny)
    X, Y = tf.meshgrid(x, y)
    ones = tf.ones_like(X)
    coords = tf.stack([X, Y, ones], axis=0)

    affine_matrix = tf.map_fn(lambda params: tf_params2matrix(params, transform_type), p, dtype=tf.float32)
    transformed = tf.einsum('bij,jhw->bihw', affine_matrix, coords)

    return transformed


# @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
#                               tf.TensorSpec(shape=[], dtype=tf.int32),
#                               tf.TensorSpec(shape=[None, None], dtype=tf.float32),
#                               tf.TensorSpec(shape=[], dtype=tf.int32)))
def tf_warp_image(I2, transform_type, p, delta) -> tf.Tensor:
    """
    Warps the input image I2 according to the transformation parameters p.
    Parameters:
    -----------
    I2 : tf.Tensor
        The input image tensor to be warped. Expected shape is [batch, height, width, channels].
    transform_type : the geometric transformation type.
    p : tf.Tensor
        The transformation parameters tensor.
    Returns:
    --------
    tf.Tensor
        The warped image tensor with the same shape as I2, where the transformation has been applied.
        Pixels outside the valid region are set to NaN.
    Raises:
    -------
    ValueError
        If the transformation type is not supported.
    """
    def warp_fn():
        # batch_size, ny, nx, _ = I2.shape
        batch_size = tf.shape(I2)[0]
        nx = tf.shape(I2)[2]
        ny = tf.shape(I2)[1]
        grid = tf_transformed_grid(batch_size, ny, nx, transform_type, p)  # grid with shape [batch, ny, nx, 2]
        warped = bicubic_sampler(I2, grid)
        grid_int = tf.cast(tf.round(grid), tf.int32)

        # Define a mask to set pixels outside the valid region to NaN
        mask = tf.logical_and(
                    tf.logical_and(grid_int[:, 0, :, :] >= delta, 
                                   grid_int[:, 0, :, :] <= nx-delta),
                    tf.logical_and(grid_int[:, 1, :, :] >= delta, 
                                   grid_int[:, 1, :, :] <= ny-delta)
                )
        mask = tf.expand_dims(mask, axis=-1)
        warped = tf.where(mask == False,
                          tf.constant(float('nan'), dtype=I2.dtype),
                          warped)
        return warped

    def error_fn():
        def raise_error():
            raise ValueError("Unsupported transformation type")
        return tf.py_function(raise_error, [], Tout=tf.float32)
    
    # Test the value of transform_type using TensorFlow operations:
    return tf.cond(
           tf.less_equal(transform_type, tf.constant(5, dtype=tf.int32)),
           warp_fn,
           error_fn
    )

# @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),
#                               tf.TensorSpec(shape=[None], dtype=tf.float32),
#                               tf.TensorSpec(shape=[], dtype=tf.int32)))
def tf_update_transform(p, dp, transform_type) -> tf.Tensor:
    """
    Update the transformation parameters p with the update dp.
    
    Args:
        p (tf.Tensor): actual transformation parameters.
        dp (tf.Tensor): update to apply.
        transform_type (TransformType): transformation type.

    Returns:
        tf.Tensor: updated transformation parameters.
    """
    threshold = tf.constant(1e-10, dtype=tf.float32)
    
    def translation():  #TransformType.TRANSLATION:
        # We suppose that p and dp have 2 parameters here.
        new_p = tf.concat([p[:2] - dp[:2], p[2:]], axis=0)
        return new_p

    def euclidean():  #TransformType.EUCLIDEAN:
        # We suppose that p and dp have 3 parameters here.
        a = tf.cos(dp[2])
        b = tf.sin(dp[2])
        c = dp[0]
        d = dp[1]
        ap = tf.cos(p[2])
        bp = tf.sin(p[2])
        cp = p[0]
        dp_val = p[1]
        cost = a * ap + b * bp
        sint = a * bp - b * ap
        new_p0 = cp - bp * (b * c - a * d) - ap * (a * c + b * d)
        new_p1 = dp_val - bp * (a * c + b * d) + ap * (b * c - a * d)
        new_p2 = tf.atan2(sint, cost)
        new_p = tf.stack([new_p0, new_p1, new_p2])
        return new_p

    def similarity():  #TransformType.SIMILARITY:
        # We suppose that p and dp have 4 parameters here.
        a = dp[2]
        b = dp[3]
        c = dp[0]
        d = dp[1]
        det = (2 * a + tf.square(a) + tf.square(b) + 1)
        def similarity_update():
            ap = p[2]
            bp = p[3]
            cp = p[0]
            dp_val = p[1]
            new_p0 = cp - bp * (-d - a * d + b * c) / det + (ap + 1) * (-c - a * c - b * d) / det
            new_p1 = dp_val + bp * (-c - a * c - b * d) / det + (ap + 1) * (-d - a * d + b * c) / det
            new_p2 = b * bp / det + (a + 1) * (ap + 1) / det - 1
            new_p3 = -b * (ap + 1) / det + bp * (a + 1) / det
            return tf.stack([new_p0, new_p1, new_p2, new_p3])
        new_p = tf.cond(tf.greater(tf.square(det), threshold), similarity_update, lambda: p)
        return new_p

    def affinity():  #TransformType.AFFINITY:
        # We suppose that p and dp have 6 parameters here.
        a = dp[2]
        b = dp[3]
        c = dp[0]
        d = dp[4]
        e = dp[5]
        f = dp[1]
        det = (a - b * d + e + a * e + 1)
        def affinity_update():
            ap = p[2]
            bp = p[3]
            cp = p[0]
            dp_val = p[4]
            ep = p[5]
            fp = p[1]
            new_p0 = cp + (-f * bp - a * f * bp + c * d * bp) / det + (ap + 1) * (-c + b * f - c * e) / det
            new_p1 = fp + dp_val * (-c + b * f - c * e) / det + (-f + c * d - a * f - f * ep - a * f * ep + d * d * ep) / det
            new_p2 = ((1 + ap) * (1 + e) - d * bp) / det - 1
            new_p3 = (bp + a * bp - b - b * ap) / det
            new_p4 = (dp_val * (1 + e) - d - d * ep) / det
            new_p5 = (a + ep + a * ep + 1 - b * dp_val) / det - 1
            return tf.stack([new_p0, new_p1, new_p2, new_p3, new_p4, new_p5])
        new_p = tf.cond(tf.greater(tf.square(det), threshold), affinity_update, lambda: p)
        return new_p

    def homography():  #TransformType.HOMOGRAPHY:
        # We suppose that p and dp have 8 parameters here.
        a = dp[0]
        b = dp[1]
        c = dp[2]
        d = dp[3]
        e = dp[4]
        f = dp[5]
        g = dp[6]
        h = dp[7]
        ap = p[0]
        bp = p[1]
        cp = p[2]
        dp_val = p[3]
        ep = p[4]
        fp = p[5]
        gp = p[6]
        hp = p[7]
        det = f * hp + a * f * hp - c * d * hp + gp * (c - b * f + c * e) - a + b * d - e - a * e - 1
        def homography_update():
            new_p0 = ((d * bp - f * g * bp) + cp * (g - d * h + g * e) + (ap + 1) * (f * h - e - 1)) / det - 1
            new_p1 = (h * cp + a * h * cp - b * g * cp - bp - a * bp + c * g * bp + b - c * h + b * ap - c * h * ap) / det
            new_p2 = (f * bp + a * f * bp - c * d * bp + (ap + 1) * (c - b * f + c * e) + cp * (-a + b * d - e - a * e - 1)) / det
            new_p3 = (fp * (g - d * h + g * e) + d - f * g + d * ep - f * g * ep + dp_val * (f * h - e - 1)) / det
            new_p4 = (b * dp_val - c * h * dp_val + h * fp + a * h * fp - b * g * fp - ep - a * ep + c * g * ep - 1) / det - 1
            new_p5 = (dp_val * (c - b * f + c * e) + f + a * f - c * d + f * ep + a * f * ep - c * d * ep + fp * (-a + b * d - e - a * e - 1)) / det
            new_p6 = (d * hp - f * g * hp + g - d * h + g * e + gp * (f * h - e - 1)) / det
            new_p7 = (h + a * h - b * g + b * gp - c * h * gp - hp - a * hp + c * g * hp) / det
            return tf.stack([new_p0, new_p1, new_p2, new_p3, new_p4, new_p5, new_p6, new_p7])
        new_p = tf.cond(tf.greater(tf.square(det), threshold), homography_update, lambda: p)
        return new_p

    branch_fns = {
        0: translation,
        1: euclidean,
        2: similarity,
        3: affinity,
        4: homography
    }
    def error_fn():
        def raise_error():
            raise ValueError("Unsupported transformation type")
        return tf.py_function(raise_error, [], Tout=tf.float32)
    
    return tf.switch_case(transform_type, branch_fns, default=error_fn)


