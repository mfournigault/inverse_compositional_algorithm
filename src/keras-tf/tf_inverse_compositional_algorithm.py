import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from enum import Enum

import image_optimisation as io
import transformation as tr
import constants as cts
import zoom as zm

from tf_image_optimisation import tf_robust_error_function, tf_steepest_descent_images
from tf_transformation import tf_update_transform, tf_params2matrix, tf_warp_image, tf_nparams
from tf_derivatives import tf_jacobian, tf_compute_gradients
from tf_zoom import tf_zoom_in_parameters

import imageio


def mark_boundaries_as_nan(tensor, delta):
    # tensor with shape [batch, H, W, C]
    shape = tf.shape(tensor)
    H = shape[1]
    W = shape[2]
    # create a mask for rows
    rows = tf.range(H)
    valid_rows = tf.logical_and(rows >= delta, rows < H - delta)
    # create a mask for columns
    cols = tf.range(W)
    valid_cols = tf.logical_and(cols >= delta, cols < W - delta)
    valid_mask = tf.logical_and(
        tf.expand_dims(valid_rows, axis=1),
        tf.expand_dims(valid_cols, axis=0)
    )  # shape [H, W]
    valid_mask = tf.reshape(valid_mask, [1, H, W, 1])
    valid_mask = tf.tile(valid_mask, [shape[0], 1, 1, shape[3]])
    # replace non valid pixels with NaN
    return tf.where(valid_mask, tensor, tf.constant(np.nan, dtype=tensor.dtype))


def save_image(img_tensor, name):
    prep = tf.where(tf.math.is_nan(img_tensor), tf.constant(0., dtype=img_tensor.dtype), img_tensor)
    min_val = tf.reduce_min(prep)
    max_val = tf.reduce_max(prep)
    normalized = (prep - min_val) / (max_val - min_val)
    
    # Convertir en uint8 [0,255] en utilisant tf.image.convert_image_dtype
    img_uint8 = tf.image.convert_image_dtype(normalized, tf.uint8).numpy()
    imageio.imwrite(name, img_uint8)


def pad_params(params, required_length):
    current_length = tf.shape(params)[0]
    pad_length = required_length - current_length
    return tf.cond(
        tf.greater(pad_length, 0),
        lambda: tf.pad(params, [[0, pad_length]]),
        lambda: params
    )


class InverseCompositional(Layer):
    """
        InverseCompositional is a custom TensorFlow layer that implements 
        the inverse compositional algorithm for image alignment. This class 
        supports various transformation types and (only) the quadratic error 
        function to
        handle different types of image transformations and noise.

        Most of the methods in this class are decorated with the @tf.function to
        enable TensorFlow graph mode execution for better performance.

        Attributes:
        TOL : float
            Tolerance used for the convergence in the iterations.
        nanifoutside : bool
            If True, the pixels outside the image are considered as NaN.
        delta : int
            Maximal distance to boundary to consider the pixel as NaN.
        verbose : bool
            If True, switch on messages.
        max_iter : int
            Maximal number of iterations.
        Methods:
        __init__(self, transform_type: tr.TransformType, ...)
        build(self, input_shape)
        call(self, inputs)
    """
    def __init__(
        self, 
        TOL: float = 1e-3,
        nanifoutside: bool = True,
        delta: int = 10,
        verbose: bool = True,
        max_iter: int = 30,
         **kwargs):
        """
        Initialize the InverseCompositional class.
        Parameters:
        -----------
        TOL : float, optional
            Tolerance used for the convergence in the iterations (default is 1e-3).
        nanifoutside : bool, optional
            If True, the pixels outside the image are considered as NaN (default is True).
        delta : int, optional
            Maximal distance to boundary to consider the pixel as NaN (default is 10).
        verbose : bool, optional
            If True, switch on messages (default is True).
        max_iter : int, optional
            Maximal number of iterations (default is 30).
        **kwargs : dict
            Additional keyword arguments.
        """
        
        super(InverseCompositional, self).__init__(trainable=False, **kwargs)
        self.TOL = tf.constant(TOL, dtype=tf.float32)
        self.nanifoutside = tf.constant(nanifoutside, dtype=tf.bool)
        self.delta = tf.constant(delta, dtype=tf.int32)
        self.verbose = tf.constant(verbose, dtype=tf.bool)
        self.max_iter = tf.constant(max_iter, dtype=tf.int32)

    def build(self, input_shape):
        """
        Builds the internal representation of the layer based on the input shape.

        Parameters:
        input_shape (tuple): A tuple representing the shape of the input tensor. 
                             The first element of the tuple is expected to be a list where:
                             - The first element is the batch size.
                             - The second, third, and fourth elements are the dimensions (ny, nx, nz) of the input tensor.
        """
        self.batch_size = tf.Variable(input_shape[0][0], dtype=tf.int32)
        self.ny = tf.Variable(input_shape[0][1], dtype=tf.int32)
        self.nx = tf.Variable(input_shape[0][2], dtype=tf.int32)
        self.nz = tf.Variable(input_shape[0][3], dtype=tf.int32)

    @tf.function(reduce_retracing=True)
    def call(self, inputs, transform_type=None, **kwargs):
        """
        Perform the inverse compositional algorithm on the input images.
        Args:
            inputs (tuple): A tuple containing three elements:
                - I1 (tf.Tensor): The first batch of images.
                - I2 (tf.Tensor): The second batch of images.
                - p (tf.Tensor): The initial parameters for the transformation.
            transform_type (tr.TransformType): The type of transformation to be used.
        Returns:
            tuple: A tuple containing four elements:
                - p_final (tf.Tensor): The final transformation parameters.
                - final_error (tf.Tensor): The final error value.
                - DI_final (tf.Tensor): The difference image after the final iteration.
                - Iw_final (tf.Tensor): The final warped image.
        Notes:
            - The function assumes that the input images are batches, where 
            each image in I1 is associated with an image in I2 and an initial parameter set p.
            - The function performs several steps including casting the input 
            images to float32, computing gradients, discarding boundary pixels 
            if necessary, and iteratively updating the transformation parameters using a robust error function.
            - The convergence criteria are based on the error norm and the maximum number of iterations.
        """
        # inputs are batches: every I1 in the batch is associated to a I2 in 
        # the batch and a p_init
        I1, I2, p = inputs

        # Moved transform_type from the init to the call in order to be able to change
        # it with new datasets without having to recompile the layer
        # We substract 1 to transform_type because the value of the enum starts at 1
        # and tf.switch_case starts at 0
        self.transform_type = tf.constant(transform_type.value-1, dtype=tf.int32)
        
        if I1.dtype != tf.float32:
            I1 = tf.cast(I1, tf.float32)
        if I2.dtype != tf.float32:
            I2 = tf.cast(I2, tf.float32)
        
        J = tf_jacobian(self.transform_type, self.nx, self.ny)

        Ix, Iy = tf_compute_gradients(I1)
        # Like in the modified version of the algorithm, we discard boundary pixels
        if (self.nanifoutside is True and self.delta > 0):
            Ix = mark_boundaries_as_nan(Ix, self.delta)
            Iy = mark_boundaries_as_nan(Iy, self.delta)
        
        DIJ = tf_steepest_descent_images(Ix, Iy, J)
        
        # Compute Hessian and b
        DIJ_filled = tf.where(tf.keras.ops.isfinite(DIJ), DIJ, 0)
        DIJt = tf.einsum("bhwcn->bhwnc", DIJ_filled)
        H = tf.einsum("bhwnc,bhwcm->bnm", DIJt, DIJ_filled)
        H_inv = tf.linalg.inv(H)

        # @tf.function
        def body(i, p, error, DI_init, Iw_init):
            # Warp I2 with current parameters
            Iw = tf_warp_image(I2, self.transform_type, p, self.delta)
            DI = Iw - I1

            DI_filled = tf.where(tf.keras.ops.isfinite(DI), DI, 0)
            
            # Compute the independant vector b
            prod = tf.einsum("bhwnc,bhwc->bhwn", DIJt, DI_filled)
            b = tf.einsum('bhwn->bn', prod)
            
            # Solve for dp
            dp = tf.einsum('bij,bj->bi', H_inv, b)  # dp is a batch of updates
            error = tf.norm(dp) # error is a batch of update norms
            dp_pad = tf.map_fn(lambda x: pad_params(x, 8), dp,
                               fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.float32))
            dp = dp_pad 
        
            updated_params = tf.map_fn(
                lambda x: tf_update_transform(x[0], x[1], self.transform_type),
                (p, dp),
                fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.float32)
            )
            p = updated_params
            
            if self.verbose:
                for b in range(self.batch_size):
                    tf.print(f"|Dp|={error}: p=(", end="")
                    for i in range(tf_nparams(self.transform_type) - 1):
                        tf.print(f"{p[b][i]} ", end="")

            return i+1, p, error, DI, Iw
        
        def cond(i, p, error, DI, Iw):
            # We assume that if a batch of images is provided to the layer,
            # images are homogenous in the sense that they originate from the
            # same scene and are captured by the same sensor. Therefore, we
            # consider that the convergence criteria is the same for all
            # images and we include the error of all images in the batch to
            # determine the convergence
            return (i < self.max_iter) and (error > self.TOL) 

        i = tf.constant(0)
        error = tf.constant(1e10, dtype=tf.float32)
        Iw_init = tf_warp_image(I2, self.transform_type, p, self.delta)
        DI_init = Iw_init - I1

        i, p_final, final_error, DI_final, Iw_final = tf.while_loop(
            cond, body, (i, p, error, DI_init, Iw_init),
            parallel_iterations=1,
            maximum_iterations=self.max_iter,
            shape_invariants=(
                tf.TensorShape([]),                      # i is scalair
                tf.TensorShape([None, None]),              # p: batch_size,  nparams 
                tf.TensorShape([]),                      # error is scalar
                tf.TensorShape(None),  # DI: batch, H, W, C
                tf.TensorShape(None)   # Iw: batch, H, W, C
            ))
        
        return p_final, final_error, DI_final, Iw_final



class RobustInverseCompositional(Layer):
    """
        RobustInverseCompositional is a custom TensorFlow layer that implements 
        the inverse compositional algorithm for image alignment. This class 
        supports various transformation types and robust error functions to
        handle different types of image transformations and noise.

        Most of the methods in this class are decorated with the @tf.function to
        enable TensorFlow graph mode execution for better performance.

        Attributes:
        TOL : float
            Tolerance used for the convergence in the iterations.
        robust_type : io.RobustErrorFunctionType
            The type of robust error function to be used.
        lambda_ : float
            Parameter of the robust error function.
        nanifoutside : bool
            If True, the pixels outside the image are considered as NaN.
        delta : int
            Maximal distance to boundary to consider the pixel as NaN.
        verbose : bool
            If True, switch on messages.
        max_iter : int
            Maximal number of iterations.
        Methods:
        __init__(self, transform_type: tr.TransformType, ...)
        build(self, input_shape)
        call(self, inputs)
    """
    def __init__(
        self, 
        TOL: float = 1e-3,
        robust_type: io.RobustErrorFunctionType = io.RobustErrorFunctionType.CHARBONNIER,
        lambda_: float = 0.0,
        nanifoutside: bool = True,
        delta: int = 10,
        verbose: bool = True,
        max_iter: int = 30,
         **kwargs):
        """
        Initialize the RobustInverseCompositional class.
        Parameters:
        -----------
        TOL : float, optional
            Tolerance used for the convergence in the iterations (default is 1e-3).
        robust_type : io.RobustErrorFunctionType, optional
            The type of robust error function to be used (default is CHARBONNIER).
        lambda_ : float, optional
            Parameter of the robust error function (default is 0.0).
        nanifoutside : bool, optional
            If True, the pixels outside the image are considered as NaN (default is True).
        delta : int, optional
            Maximal distance to boundary to consider the pixel as NaN (default is 10).
        verbose : bool, optional
            If True, switch on messages (default is True).
        max_iter : int, optional
            Maximal number of iterations (default is 30).
        **kwargs : dict
            Additional keyword arguments.
        """
        super(RobustInverseCompositional, self).__init__(trainable=False, **kwargs)
        self.TOL = tf.constant(TOL, dtype=tf.float32)
        self.nanifoutside = tf.constant(nanifoutside, dtype=tf.bool)
        self.delta = tf.constant(delta, dtype=tf.int32)
        self.verbose = tf.constant(verbose, dtype=tf.bool)
        self.max_iter = tf.constant(max_iter, dtype=tf.int32)
        self.robust_type = tf.constant(robust_type.value, dtype=tf.int32)
        self.lambda_ = tf.constant(lambda_, dtype=tf.float32)
        self.lambda_it = lambda_ if lambda_ > 0 else tf.constant(cts.LAMBDA_0, dtype=tf.float32)

    def build(self, input_shape):
        """
        Builds the internal representation of the layer based on the input shape.

        Parameters:
        input_shape (tuple): A tuple representing the shape of the input tensor. 
                             The first element of the tuple is expected to be a list where:
                             - The first element is the batch size.
                             - The second, third, and fourth elements are the dimensions (ny, nx, nz) of the input tensor.
        """
        self.batch_size = input_shape[0][0]
        self.ny, self.nx, self.nz = input_shape[0][1:4]

    @tf.function(reduce_retracing=True)
    def call(self, inputs, transform_type=None, **kwargs):
        """
        Perform the inverse compositional algorithm on the input images.
        Args:
            inputs (tuple): A tuple containing three elements:
                - I1 (tf.Tensor): The first batch of images.
                - I2 (tf.Tensor): The second batch of images.
                - p (tf.Tensor): The initial parameters for the transformation.
            transform_type (tr.TransformType): The type of transformation to be used.
        Returns:
            tuple: A tuple containing four elements:
                - p_final (tf.Tensor): The final transformation parameters.
                - final_error (tf.Tensor): The final error value.
                - DI_final (tf.Tensor): The difference image after the final iteration.
                - Iw_final (tf.Tensor): The final warped image.
        Notes:
            - The function assumes that the input images are batches, where 
            each image in I1 is associated with an image in I2 and an initial parameter set p.
            - The function performs several steps including casting the input 
            images to float32, computing gradients, discarding boundary pixels 
            if necessary, and iteratively updating the transformation parameters using a robust error function.
            - The convergence criteria are based on the error norm and the maximum number of iterations.
        """
        # inputs are batches: every I1 in the batch is associated to a I2 in 
        # the batch and a p_init
        I1, I2, p = inputs
        
        # Moved transform_type from the init to the call in order to be able to change
        # it with new datasets without having to recompile the layer
        # We substract 1 to transform_type because the value of the enum starts at 1
        # and tf.switch_case starts at 0
        self.transform_type = tf.constant(transform_type.value-1, dtype=tf.int32)

        if I1.dtype != tf.float32:
            I1 = tf.cast(I1, tf.float32)
        if I2.dtype != tf.float32:
            I2 = tf.cast(I2, tf.float32)

        J = tf_jacobian(self.transform_type, self.nx, self.ny)

        Ix, Iy = tf_compute_gradients(I1)
        # Like in the modified version of the algorithm, we discard boundary pixels
        if (self.nanifoutside is True and self.delta > 0):
            Ix = mark_boundaries_as_nan(Ix, self.delta)
            Iy = mark_boundaries_as_nan(Iy, self.delta)
        
        DIJ = tf_steepest_descent_images(Ix, Iy, J)
        
        # @tf.function
        def body(i, p, error, DI_init, Iw_init):
            # Warp I2 with current parameters
            Iw = tf_warp_image(I2, self.transform_type, p, self.delta)
            DI = Iw - I1

            # Compute robust error
            rho = tf.map_fn(
                lambda x: tf_robust_error_function(x, self.lambda_it, self.robust_type),
                DI,
                fn_output_signature=tf.TensorSpec(shape=[None, None], dtype=tf.float32)
            )
            # rho = tf_robust_error_function(DI, self.lambda_it, self.robust_type)
            
            # update lambda
            if self.lambda_ <= 0 and self.lambda_it > cts.LAMBDA_N:
                self.lambda_it *= cts.LAMBDA_RATIO
                if self.lambda_it < cts.LAMBDA_N:
                    self.lambda_it = cts.LAMBDA_N
            
            # Compute Hessian and b
            DIJ_filled = tf.where(tf.keras.ops.isfinite(DIJ), DIJ, 0)
            DIJt = tf.einsum("bhwcn->bhwnc", DIJ_filled)
            DI_filled = tf.where(tf.keras.ops.isfinite(DI), DI, 0)
            H = tf.einsum("bhw,bhwnc,bhwcm->bnm", rho, DIJt, DIJ_filled)
            H_inv = tf.linalg.inv(H)
        
            prod = tf.einsum("bhwnc,bhwc->bhwn", DIJt, DI_filled)
            prod = tf.einsum("bhw,bhwn->bhwn", rho, prod)
            b = tf.einsum('bhwn->bn', prod)
            
            # Solve for dp
            dp = tf.einsum('bij,bj->bi', H_inv, b)  # dp is a batch of updates
            error = tf.norm(dp) # error is a batch of update norms
        
            updated_params = tf.map_fn(
                lambda x: tf_update_transform(x[0], x[1], self.transform_type),
                (p, dp),
                fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.float32)
            )
            p = updated_params
            
            if self.verbose:
                for b in range(self.batch_size):
                    tf.print(f"|Dp|={error}: p=(", end="")
                    for i in range(tf_nparams(self.transform_type) - 1):
                        tf.print(f"{p[b][i]} ", end="")
                    tf.print(f"{p[b][tf_nparams(self.transform_type) - 1]}), lambda_={self.lambda_it}")

            return i+1, p, error, DI, Iw
        
        def cond(i, p, error, DI, Iw):
            # We assume that if a batch of images is provided to the layer,
            # images are homogenous in the sense that they originate from the
            # same scene and are captured by the same sensor. Therefore, we
            # consider that the convergence criteria is the same for all
            # images and we include the error of all images in the batch to
            # determine the convergence
            return (i < self.max_iter) and (error > self.TOL) 

        i = tf.constant(0)
        error = tf.constant(1e10, dtype=tf.float32)
        Iw_init = tf_warp_image(I2, self.transform_type, p, self.delta)
        DI_init = Iw_init - I1

        i, p_final, final_error, DI_final, Iw_final = tf.while_loop(
            cond, body, (i, p, error, DI_init, Iw_init),
            parallel_iterations=1,
            maximum_iterations=self.max_iter,
            shape_invariants=(
                tf.TensorShape([]),                      # i is scalair
                tf.TensorShape([None, None]),              # p: batch_size,  nparams 
                tf.TensorShape([]),                      # error is scalar
                tf.TensorShape(None),  # DI: batch, H, W, C
                tf.TensorShape(None)   # Iw: batch, H, W, C
            ))
        
        return p_final, final_error, DI_final, Iw_final

class PyramidalInverseCompositional(Layer):
    def __init__(
                self, 
                transform_type: tr.TransformType, # typeof transformation
                nscales: int=3, # number of scales
                nu: float=0.5,      # downsampling factor
                TOL: float=1e-3,     # stopping criterion threshold
                robust_type: io.RobustErrorFunctionType=io.RobustErrorFunctionType.CHARBONNIER,  # type of robust error function
                lambda_: float=0.0,  # parameter of robust error function
                nanifoutside: bool=True, # if True, the pixels outside the image are considered as NaN
                delta: int=10, # maximal distance to boundary to consider the pixel as NaN
                verbose: bool=True,  # switch on messages
                **kwargs
                ):
        super(PyramidalInverseCompositional, self).__init__(trainable=False, **kwargs)
        self.nscales = nscales
        self.nu = nu
        self.transform_type = transform_type
        self.verbose = verbose

        if robust_type == io.RobustErrorFunctionType.QUADRATIC:
            self.ic_layers = [InverseCompositional(
                                TOL,
                                nanifoutside,
                                delta,
                                verbose,
                                cts.MAX_ITER,
                                name=f"IC_{s}",
                                **kwargs) for s in range(nscales)]
        else:
            self.ic_layers = [RobustInverseCompositional(
                                    TOL,
                                    robust_type,
                                    lambda_,
                                    nanifoutside,
                                    delta,
                                    verbose,
                                    cts.MAX_ITER,
                                    name=f"RIC_{s}",
                                    **kwargs) for s in range(nscales)]

    def build(self, input_shape):
        self.pyramid_shapes = []
        print("input_shape: ", input_shape)
        self.batch_size = input_shape[0][0]
        current_shape = input_shape[0][1:]  # Assume I1, I2 have same shape
        print("current_shape: ", current_shape)
        for _ in range(self.nscales):
            self.pyramid_shapes.append(current_shape)
            current_shape = (int(current_shape[0]*self.nu),
                             int(current_shape[1]*self.nu),
                             current_shape[2])

    def call(self, inputs, transform_type=None, **kwargs):
        # Called for a new transform type, we update it
        if transform_type is not None:
            self.transform_type = transform_type

        I1, I2 = inputs
        if I1.dtype != tf.float32:
            I1 = tf.cast(I1, tf.float32)
        if I2.dtype != tf.float32:
            I2 = tf.cast(I2, tf.float32)
        nparams = self.transform_type.nparams()
        # p = [tf.zeros((self.batch_size, nparams), dtype=tf.float32) for _ in range(self.nscales)]
        p = [tf.zeros((self.batch_size, 8), dtype=tf.float32) for _ in range(self.nscales)]

        # Build pyramids
        I1_pyramid = [I1]
        I2_pyramid = [I2]
        for i in range(1, self.nscales):
            # Bug fix: use bicubic interpolation for downsampling
            I1_pyramid.append(tf.image.resize(I1_pyramid[-1], 
                                              self.pyramid_shapes[i][:2], 
                                              method=tf.image.ResizeMethod.BICUBIC
                                              ))
            I2_pyramid.append(tf.image.resize(I2_pyramid[-1],
                                              self.pyramid_shapes[i][:2], 
                                              method=tf.image.ResizeMethod.BICUBIC
                                              ))
        
        # Process from coarse to fine
        for scale in reversed(range(self.nscales)):
            if self.verbose:
                tf.print(f"Scale: {scale}")
            ic_layer = self.ic_layers[scale]
            # print("shape of p[scale]: ", p[scale].shape)
            p_scale, error, DI, Iw = ic_layer(inputs=[I1_pyramid[scale], I2_pyramid[scale], p[scale]], 
                                              transform_type=self.transform_type)
            p[scale] = p_scale

            # Upscale parameters for next scale
            if scale > 0:
                # updated_params = []
                upscaled_param = tf.map_fn(
                    lambda x: tf_zoom_in_parameters(
                        x,
                        tf.constant(self.transform_type.value-1, dtype=tf.int32),
                        self.pyramid_shapes[scale][2],
                        self.pyramid_shapes[scale][1],
                        self.pyramid_shapes[scale-1][2],
                        self.pyramid_shapes[scale-1][1]),
                    p[scale],
                    fn_output_signature=tf.TensorSpec(shape=[None], dtype=tf.float32)
                )
                p[scale-1] = upscaled_param
                # for b in range(self.batch_size):
                #     upscaled_param = tf_zoom_in_parameters(p[scale][b],
                #                             self.transform_type,
                #                             self.pyramid_shapes[scale][2],
                #                             self.pyramid_shapes[scale][1],
                #                             self.pyramid_shapes[scale-1][2],
                #                             self.pyramid_shapes[scale-1][1])
                #     updated_params.append(upscaled_param)
                # p[scale-1] = tf.stack(updated_params, axis=0)

        return p, error, DI, Iw

