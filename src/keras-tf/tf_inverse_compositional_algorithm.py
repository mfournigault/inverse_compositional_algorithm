import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from enum import Enum

import image_optimisation as io
import transformation as tr
import constants as cts
import zoom as zm
import bicubic_interpolation as bi
from tf_image_optimisation import tf_robust_error_function
from tf_bicubic_interpolation import bicubic_sampler
from tf_transformation import tf_update_transform

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
    tf.print("Étendue de prep : min =", min_val, ", max =", max_val)
    normalized = (prep - min_val) / (max_val - min_val)
    
    # Convertir en uint8 [0,255] en utilisant tf.image.convert_image_dtype
    img_uint8 = tf.image.convert_image_dtype(normalized, tf.uint8).numpy()
    imageio.imwrite(name, img_uint8)


class RobustInverseCompositional(Layer):
    
    def __init__(
        self, 
        transform_type: tr.TransformType, 
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
        transform_type : tr.TransformType
            The type of transformation to be used.
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
        self.transform_type = transform_type
        self.nparams = self.transform_type.nparams()
        self.TOL = TOL
        self.robust_type = robust_type
        self.lambda_ = lambda_
        self.lambda_it = lambda_ if lambda_ > 0 else cts.LAMBDA_0
        self.nanifoutside = nanifoutside
        self.delta = delta
        self.verbose = verbose
        self.max_iter = max_iter

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
        
    def compute_gradients(self, I1):
        """
        Computes the gradients of the input image tensor I1 along the x and y axes.

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

    def jacobian(self):
        """
        Compute the Jacobian matrix for the specified transformation type.

        The Jacobian matrix is computed based on the transformation type and the 
        dimensions of the input image. The supported transformation types are:
        - TRANSLATION
        - EUCLIDEAN
        - SIMILARITY
        - AFFINITY
        - HOMOGRAPHY

        Returns:
            tf.Tensor: A tensor representing the Jacobian matrix with an added 
            batch dimension.
        """
        x = tf.range(self.nx, dtype=tf.float32)
        y = tf.range(self.ny, dtype=tf.float32)
        X, Y = tf.meshgrid(x, y)
        ones = tf.ones_like(X)
        zeros = tf.zeros_like(X)
        match self.transform_type:
            case tr.TransformType.TRANSLATION:
                J = tf.stack([ones, zeros, zeros, ones], axis=-1)
            case tr.TransformType.EUCLIDEAN:
                J = tf.stack([ones, zeros, -Y, zeros, ones, X], axis=-1)
            case tr.TransformType.SIMILARITY:
                J = tf.stack([ones, zeros, X, -Y, zeros, ones, Y, X], axis=-1)
            case tr.TransformType.AFFINITY:
                J = tf.stack([ones, zeros, X, Y, zeros, zeros, zeros, ones, zeros, zeros, X, Y], axis=-1)
            case tr.TransformationType.HOMOGRAPHY:
                J = tf.stack([X, Y, ones, zeros, zeros, zeros, -X*X, -X*Y, zeros, zeros, zeros, X, Y, ones, -X*Y, -Y*Y], axis=-1)
        return tf.expand_dims(J, 0)  # Add batch dimension

    def steepest_descent_images(self, Ix, Iy, J):
        """
        Compute the steepest descent images for the inverse compositional algorithm.
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

    def warp_image(self, I2, p):
        """
        Warps the input image I2 according to the transformation parameters p.
        Parameters:
        -----------
        I2 : tf.Tensor
            The input image tensor to be warped. Expected shape is [batch, height, width, channels].
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
        if self.transform_type in [tr.TransformType.TRANSLATION,
                                   tr.TransformType.EUCLIDEAN,
                                   tr.TransformType.SIMILARITY,
                                   tr.TransformType.AFFINITY, 
                                   tr.TransformType.HOMOGRAPHY]:
            grid = self.transformed_grid(p)  # grid with shape [batch, self.ny, self.nx, 2]
            warped = bicubic_sampler(I2, grid)
            grid_int = tf.cast(tf.round(grid), tf.int32)

            # Define a mask to set pixels outside the valid region to NaN
            mask = tf.logical_and(
                        tf.logical_and(grid_int[:, 0, :, :] >= self.delta, 
                                       grid_int[:, 0, :, :] <= I2.shape[2]-self.delta),
                        tf.logical_and(grid_int[:, 1, :, :] >= self.delta, 
                                       grid_int[:, 1, :, :] <= I2.shape[1]-self.delta)
                    )
            mask = tf.expand_dims(mask, axis=-1)
            warped = tf.where(mask == False,
                              tf.constant(float('nan'), dtype=I2.dtype),
                              warped)
            return warped
        else:
            raise ValueError("Unsupported transformation type")

    def transformed_grid(self, p):
        """
        Transforms a grid of coordinates using a batch of affine transformation parameters.
        Args:
            p (tf.Tensor): A tensor of shape (batch_size, num_params) containing the affine transformation parameters for each element in the batch.
        Returns:
            tf.Tensor: A tensor of shape (batch_size, 3, nx, ny) containing the transformed grid coordinates for each element in the batch.
        """
        x = tf.linspace(0.0, self.nx-1, self.nx)
        y = tf.linspace(0.0, self.ny-1, self.ny)
        X, Y = tf.meshgrid(x, y)
        ones = tf.ones_like(X)
        coords = tf.stack([X, Y, ones], axis=0)
        # Use of map_fn to apply the function params2matrix to each element of p (batch of parameters)
        affine_matrix = tf.map_fn(lambda params: tr.params2matrix(params, self.transform_type), p, dtype=tf.float32)
        transformed = tf.einsum('bij,jhw->bihw', affine_matrix, coords)
        
        return transformed

    def call(self, inputs):
        """
        Perform the inverse compositional algorithm on the input images.
        Args:
            inputs (tuple): A tuple containing three elements:
                - I1 (tf.Tensor): The first batch of images.
                - I2 (tf.Tensor): The second batch of images.
                - p (tf.Tensor): The initial parameters for the transformation.
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
        if I1.dtype != tf.float32:
            I1 = tf.cast(I1, tf.float32)
        if I2.dtype != tf.float32:
            I2 = tf.cast(I2, tf.float32)
        J = self.jacobian()
        Ix, Iy = self.compute_gradients(I1)
        # Like in the modified version of the algorithm, we discard boundary pixels
        if (self.nanifoutside is True and self.delta > 0):
            Ix = mark_boundaries_as_nan(Ix, self.delta)
            Iy = mark_boundaries_as_nan(Iy, self.delta)
        
        DIJ = self.steepest_descent_images(Ix, Iy, J)
        
        def body(i, p, error, DI_init, Iw_init):
            # Warp I2 with current parameters
            Iw = self.warp_image(I2, p)
            DI = Iw - I1

            # Compute robust error
            #TODO: apply map_fn to compute the robust error for each image in the batch
            rho = tf.map_fn(
                lambda x: tf_robust_error_function(x, self.lambda_it, self.robust_type),
                DI,
                dtype=tf.float32
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
                dtype=tf.float32
            )
            p = updated_params
            
            if self.verbose:
                for b in range(self.batch_size):
                    # tf.print(f"--- Batch {b}")
                    tf.print(f"|Dp|={error}: p=(", end="")
                    for i in range(self.nparams - 1):
                        tf.print(f"{p[b][i]} ", end="")
                    tf.print(f"{p[b][self.nparams - 1]}), lambda_={self.lambda_it}")

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
        Iw_init = self.warp_image(I2, p)
        DI_init = Iw_init - I1

        i, p_final, final_error, DI_final, Iw_final = tf.while_loop(
            cond, body, (i, p, error, DI_init, Iw_init),
            parallel_iterations=1,
            maximum_iterations=self.max_iter)
        
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
        self.ric_layers = [RobustInverseCompositional(
                                transform_type,
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
        self.batch_size = input_shape[0][0]
        current_shape = input_shape[0][1:]  # Assume I1, I2 have same shape
        for _ in range(self.nscales):
            self.pyramid_shapes.append(current_shape)
            current_shape = (int(current_shape[0]*self.nu),
                             int(current_shape[1]*self.nu),
                             current_shape[2])

    def call(self, inputs):
        I1, I2 = inputs
        if I1.dtype != tf.float32:
            I1 = tf.cast(I1, tf.float32)
        if I2.dtype != tf.float32:
            I2 = tf.cast(I2, tf.float32)
        nparams = self.transform_type.nparams()
        p = [tf.zeros((self.batch_size, nparams), dtype=tf.float32) for _ in range(self.nscales)]

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
            ric_layer = self.ric_layers[scale]
            # print("shape of p[scale]: ", p[scale].shape)
            p_scale, error, DI, Iw = ric_layer([I1_pyramid[scale], I2_pyramid[scale], p[scale]])
            p[scale] = p_scale

            # Upscale parameters for next scale
            if scale > 0:
                updated_params = []
                for b in range(self.batch_size):
                    upscaled_param = zm.zoom_in_parameters(p[scale][b],
                                            self.transform_type,
                                            self.pyramid_shapes[scale][2],
                                            self.pyramid_shapes[scale][1],
                                            self.pyramid_shapes[scale-1][2],
                                            self.pyramid_shapes[scale-1][1])
                    updated_params.append(upscaled_param)
                p[scale-1] = tf.stack(updated_params, axis=0)

        return p, error, DI, Iw

    def upscale_parameters(self, p, old_shape, new_shape):
        # Implement parameter upscaling logic based on transform type
        scale_factor = (new_shape[0]/old_shape[0], new_shape[1]/old_shape[1])
        if self.transform_type == TransformType.AFFINE:
            # Scale translation parameters
            scaled_p = p * tf.constant([scale_factor[1], scale_factor[0], 
                                      scale_factor[1], scale_factor[0], 
                                      1.0, 1.0], dtype=tf.float32)
            return scaled_p
        # Add other transform types as needed
        return p