import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from enum import Enum
import image_optimisation as io
import transformation as tr


class TransformType(Enum):
    AFFINE = 6
    # Add other transform types as needed

class RobustErrorFunctionType(Enum):
    QUADRATIC = 1
    HUBER = 2

class RobustInverseCompositional(Layer):
    
    def __init__(
        self, 
        transform_type: tr.TransformType, 
        TOL=1e-3: float,    # Tolerance used for the convergence in the iterations
        robust_type: io.RobustErrorFunctionType, # type (RobustErrorFunctionType) of robust error function
        lambda_=0.0, # parameter of robust error function
        nanifoutside=True: bool, # if True, the pixels outside the image are considered as NaN
        delta=0: int, # maximal distance to boundary to consider the pixel as NaN 
        max_iter=30:int, # maximal number of iterations
         **kwargs):
        
        super(RobustInverseCompositional, self).__init__(**kwargs)
        self.transform_type = transform_type
        self.TOL = TOL
        self.robust_type = robust_type
        self.lambda_ = lambda_
        self.nanifoutside = nanifoutside
        self.delta = delta
        self.max_iter = max_iter

    def build(self, input_shape):
        self.nparams = self.transform_type.nparams()
        self.batch_size = input_shape[0][0]
        self.ny, self.nx, self.nz = input_shape[0][1:4]
        
    def compute_gradients(self, I1):
        Ix = (I1[:, :, 2:, :] - I1[:, :, :-2, :]) * 0.5
        Ix = tf.pad(Ix, [[0,0], [0,0], [1,1], [0,0]])
        Iy = (I1[:, 2:, :, :] - I1[:, :-2, :, :]) * 0.5
        Iy = tf.pad(Iy, [[0,0], [0,0], [1,1], [0,0]])
        return Ix, Iy

    def jacobian(self):
        # Example for affine transform, implement other transforms as needed
        x = tf.range(self.nx, dtype=tf.float32)
        y = tf.range(self.ny, dtype=tf.float32)
        X, Y = tf.meshgrid(x, y)
        ones = tf.ones_like(X)
        zeros = tf.zeros_like(X)
        match self.transform_type:
            case TransformType.TRANSLATION:
                J = tf.stack([ones, tf.zeros_like(X), tf.zeros_like(X), ones], axis=-1)
            case TransformType.EUCLIDEAN:
                J = tf.stack([ones, tf.zeros_like(X), -Y, tf.zeros_like(X), ones, X], axis=-1)
            case TransformType.SIMILARITY:
                J = tf.stack([ones, X, -Y, ones, Y, X], axis=-1)
            case TransformType.AFFINITY:
                J = tf.stack([ones, zeros, X, Y, zeros, zeros, zeros, ones, zeros, zeros, X, Y], axis=-1)
            case TransformationType.HOMOGRAPHY:
                J = tf.stack([X, Y, ones, zeros, zeros, zeros, -X*X, -X*Y, zeros, zeros, zeros, X, Y, ones, -X*Y, -Y*Y], axis=-1)
        
        J = J[..., :self.nparams]
        return tf.expand_dims(J, 0)  # Add batch dimension

    def steepest_descent_images(self, Ix, Iy, J):
        gradients = tf.stack([Ix, Iy], axis=-1)
        return tf.einsum('bhwci,bhwji->bhwj', gradients, J) #TODO: Check if this is correct

    def warp_image(self, I2, p):
        # Generate flow field based on transform parameters
        if self.transform_type == TransformType.AFFINE:
            flow = self.affine_flow(p)
        # Add other transform types here
        return tf.image.dense_image_warp(I2, flow) #TODO: dense image wrap should not be useful as the transformation is linear

    def affine_flow(self, p):
        x = tf.linspace(0.0, self.nx-1, self.nx)
        y = tf.linspace(0.0, self.ny-1, self.ny)
        X, Y = tf.meshgrid(x, y)
        ones = tf.ones_like(X)
        coords = tf.stack([X, Y, ones], axis=0)
        
        affine_matrix = tf.reshape(p, [-1, 2, 3])
        transformed = tf.einsum('bij,jhw->bhw', affine_matrix, coords)
        X_new = transformed[:, 0, :, :]
        Y_new = transformed[:, 1, :, :]
        
        flow = tf.stack([X_new - X, Y_new - Y], axis=-1)
        return flow

    def robust_error(self, DI):
        if self.robust_type == RobustErrorFunctionType.HUBER:
            return tf.where(tf.abs(DI) < self.lambda_, 
                            0.5 * tf.square(DI),
                            self.lambda_ * (tf.abs(DI) - 0.5 * self.lambda_))
        else:  # Quadratic
            return 0.5 * tf.square(DI)

    def call(self, inputs):
        I1, I2, p_init = inputs
        p = tf.identity(p_init)
        J = self.jacobian()
        Ix, Iy = self.compute_gradients(I1)
        DIJ = self.steepest_descent_images(Ix, Iy, J)
        
        def body(i, p, error):
            # Warp I2 with current parameters
            Iw = self.warp_image(I2, p)
            DI = Iw - I1
            
            # Compute robust error
            rho = self.robust_error(DI)
            
            # Compute Hessian and b
            weighted_DIJ = DIJ * tf.expand_dims(rho, -1)
            H = tf.einsum('bhwij,bhwik->bjk', DIJ, weighted_DIJ)
            b = tf.einsum('bhwij,bhw->bj', DIJ, DI * rho)
            
            # Solve for dp
            H_inv = tf.linalg.inv(H)
            dp = tf.einsum('bij,bj->bi', H_inv, b)
            
            # Update parameters
            p = p - dp
            error = tf.norm(dp)
            return i+1, p, error
        
        def cond(i, p, error):
            return (i < self.max_iter) & (error > self.TOL)
        
        i = tf.constant(0)
        error = tf.constant(1e10, dtype=tf.float32)
        _, p_final, final_error = tf.while_loop(
            cond, body, [i, p, error],
            maximum_iterations=self.max_iter)
        
        # Compute final warped image
        Iw_final = self.warp_image(I2, p_final)
        DI_final = Iw_final - I1
        
        return p_final, final_error, DI_final, Iw_final

class PyramidalInverseCompositional(Layer):
    def __init__(self, nscales=3, nu=0.5, **kwargs):
        super(PyramidalInverseCompositional, self).__init__(**kwargs)
        self.nscales = nscales
        self.nu = nu
        self.ric_layers = [RobustInverseCompositional(**kwargs) for _ in range(nscales)]
        
    def build(self, input_shape):
        self.pyramid_shapes = []
        current_shape = input_shape[0][1:]  # Assume I1, I2 have same shape
        for _ in range(self.nscales):
            self.pyramid_shapes.append(current_shape)
            current_shape = (int(current_shape[0]*self.nu), 
                            int(current_shape[1]*self.nu), 
                            current_shape[2])
        
    def call(self, inputs):
        I1, I2 = inputs
        p = tf.zeros((self.batch_size, self.ric_layers[0].nparams))
        
        # Build pyramids
        I1_pyramid = [I1]
        I2_pyramid = [I2]
        for i in range(1, self.nscales):
            I1_pyramid.append(tf.image.resize(I1_pyramid[-1], self.pyramid_shapes[i][:2]))
            I2_pyramid.append(tf.image.resize(I2_pyramid[-1], self.pyramid_shapes[i][:2]))
        
        # Process from coarse to fine
        for scale in reversed(range(self.nscales)):
            ric_layer = self.ric_layers[scale]
            p, error, DI, Iw = ric_layer([I1_pyramid[scale], I2_pyramid[scale], p])
            
            # Upscale parameters for next scale
            if scale > 0:
                p = self.upscale_parameters(p, self.pyramid_shapes[scale], 
                                          self.pyramid_shapes[scale-1])
        
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