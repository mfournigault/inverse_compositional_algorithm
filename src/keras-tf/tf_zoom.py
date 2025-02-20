import tensorflow as tf



def tf_zoom_in_parameters(
        p: tf.Tensor,  # size of p is supposed to be standardized to 8, the maximum of elements for homography
        transformation_type: tf.Tensor, 
        nx: int, 
        ny: int, 
        nxx: int, 
        nyy: int
        ) -> tf.Tensor:
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


    # Adjust parameters based on the transformation type
    def translation():  # TransformType.TRANSLATION:
        pout = tf.zeros_like(p)
        pout = tf.tensor_scatter_nd_update(pout, [[0]], [p[0] * nu])
        pout = tf.tensor_scatter_nd_update(pout, [[1]], [p[1] * nu])
        return pout
    def euclidean():  # TransformType.EUCLIDEAN:
        pout = tf.zeros_like(p)
        pout = tf.tensor_scatter_nd_update(pout, [[0]], [p[0] * nu])
        pout = tf.tensor_scatter_nd_update(pout, [[1]], [p[1] * nu])
        pout = tf.tensor_scatter_nd_update(pout, [[2]], [p[2]])
        return pout
    def similarity():  # TransformType.SIMILARITY:
        pout = tf.zeros_like(p)
        pout = tf.tensor_scatter_nd_update(pout, [[0]], [p[0] * nu])
        pout = tf.tensor_scatter_nd_update(pout, [[1]], [p[1] * nu])
        pout = tf.tensor_scatter_nd_update(pout, [[2]], [p[2]])
        pout = tf.tensor_scatter_nd_update(pout, [[3]], [p[3]])
        return pout
    def affinity():  # TransformType.AFFINITY:
        pout = tf.zeros_like(p)
        pout = tf.tensor_scatter_nd_update(pout, [[0]], [p[0] * nu])
        pout = tf.tensor_scatter_nd_update(pout, [[1]], [p[1] * nu])
        pout = tf.tensor_scatter_nd_update(pout, [[2]], [p[2]])
        pout = tf.tensor_scatter_nd_update(pout, [[3]], [p[3]])
        pout = tf.tensor_scatter_nd_update(pout, [[4]], [p[4]])
        pout = tf.tensor_scatter_nd_update(pout, [[5]], [p[5]])
        return pout
    def homography():  # TransformType.HOMOGRAPHY:
        pout = tf.zeros_like(p)
        pout = tf.tensor_scatter_nd_update(pout, [[0]], [p[0]])
        pout = tf.tensor_scatter_nd_update(pout, [[1]], [p[1]])
        pout = tf.tensor_scatter_nd_update(pout, [[2]], [p[2] * nu])
        pout = tf.tensor_scatter_nd_update(pout, [[3]], [p[3]])
        pout = tf.tensor_scatter_nd_update(pout, [[4]], [p[4]])
        pout = tf.tensor_scatter_nd_update(pout, [[5]], [p[5] * nu])
        pout = tf.tensor_scatter_nd_update(pout, [[6]], [p[6] / nu])
        pout = tf.tensor_scatter_nd_update(pout, [[7]], [p[7] / nu])
        return pout
    def default():
        def raise_error():
            raise ValueError("Unsupported transformation type")
        return tf.py_function(raise_error, [], tf.float32)

    branch_fns = {
        0: translation,
        1: euclidean,
        2: similarity,
        3: affinity,
        4: homography
    }
    return tf.switch_case(transformation_type, branch_fns, default)

