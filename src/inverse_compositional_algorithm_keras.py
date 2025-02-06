import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from enum import Enum
import image_optimisation as io
import transformation as tr
import constants as cts
import zoom as zm


def cubic(x):
    absx = tf.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    # Fonction de base cubique de Keys
    return tf.where(absx <= 1,
                    (1.5 * absx3 - 2.5 * absx2 + 1.0),
                    tf.where(absx < 2,
                             (-0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0),
                             tf.zeros_like(x)))


def get_pixel_value(img, x, y):
    # img : [batch, H, W, C]
    xt = tf.squeeze(x)
    yt = tf.squeeze(y)

    batch_size = tf.shape(img)[0]  
    batch_idx = tf.range(batch_size)                        # (batch_size,)
    B, Y, X = tf.meshgrid(batch_idx, yt, xt, indexing="ij")
    indices = tf.stack([B, Y, X], axis=-1)  # (batch, ny, nx, channels)
    
    return tf.gather_nd(img, indices)


def bicubic_sampler(image, grid):
    # image: [batch, H, W, C]
    # grid: [batch, newH, newW, 2] avec coordonnées (y, x) en flottants
    input_shape = tf.shape(image)
    batch_size, H, W, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    grid_y = grid[:, :, 0]  # extract y coordinates with shape [batch, H]
    grid_x = grid[:, 0, :]  # extract x coordinates with shape [batch, W]

    # Calcul des coordonnées entières pour le voisinage (début)
    x0 = tf.cast(tf.floor(grid_x), tf.int32)  # shape [batch, W]
    y0 = tf.cast(tf.floor(grid_y), tf.int32)  # shape [batch, H]

    # Pour l'interpolation bicubique, on a besoin de 4 pixels dans chaque direction.
    # On crée donc les 4 indices autour de (x0,y0)
    x_indexes = [x0 - 1, x0, x0 + 1, x0 + 2]  # list of 4 indices with shape [batch, W]
    y_indexes = [y0 - 1, y0, y0 + 1, y0 + 2]  # list of 4 indices with shape [batch, H]

    # Calcul des poids selon la distance
    x_diff = grid_x - tf.cast(x0, tf.float32)  # shape [batch, W]
    y_diff = grid_y - tf.cast(y0, tf.float32)  # shape [batch, H]

    # list of 4 indices with shape [batch, W]
    weights_x = [cubic(x_diff + 1.0), cubic(x_diff), cubic(x_diff - 1.0), cubic(x_diff - 2.0)]
    # list of 4 indices with shape [batch, H]
    weights_y = [cubic(y_diff + 1.0), cubic(y_diff), cubic(y_diff - 1.0), cubic(y_diff - 2.0)]

    # Convert list into tensors with shape [batch, d, 4]
    weights_x = tf.stack(weights_x, axis=-1)  # list of 4 tensors with shape [batch, W] -> [batch, W, 4]
    weights_y = tf.stack(weights_y, axis=-1)  # list of 4 tensors with shape [batch, H] -> [batch, H, 4]

    # Tensor with shape [batch, H, W, channels]
    output = tf.zeros([batch_size, tf.shape(grid)[1], tf.shape(grid)[2], channels], dtype=image.dtype)

    # Accumulation on the 16 neighbors
    for i in range(4):
        for j in range(4):
            # Sélection des indices i, j
            x_i = x_indexes[i]
            y_j = y_indexes[j]
            # Pour éviter de sortir des bornes, on clippe les indices
            x_i = tf.clip_by_value(x_i, 0, W - 1)
            y_j = tf.clip_by_value(y_j, 0, H - 1)
            pixel = get_pixel_value(image, x_i, y_j)  # [batch, newH, newW, channels]
            w = tf.expand_dims(weights_y[:, :, j], axis=-1) * tf.expand_dims(weights_x[:, :, i], axis=1)
            output += pixel * tf.expand_dims(w, axis=-1)  # expand_dims of w to replicate it for all channels
    return output

class RobustInverseCompositional(Layer):
    
    def __init__(
        self, 
        transform_type: tr.TransformType, 
        TOL: float = 1e-3,    # Tolerance used for the convergence in the iterations
        robust_type: io.RobustErrorFunctionType = io.RobustErrorFunctionType.CHARBONNIER, # type (RobustErrorFunctionType) of robust error function
        lambda_: float = 0.0, # parameter of robust error function
        nanifoutside: bool = True, # if True, the pixels outside the image are considered as NaN
        delta: int = 10, # maximal distance to boundary to consider the pixel as NaN 
        max_iter: int = 30, # maximal number of iterations
         **kwargs):
        
        super(RobustInverseCompositional, self).__init__(trainable=False, **kwargs)
        self.transform_type = transform_type
        self.nparams = self.transform_type.nparams()
        self.TOL = TOL
        self.robust_type = robust_type
        self.lambda_ = lambda_ if lambda_ > 0 else cts.LAMBDA_0
        self.nanifoutside = nanifoutside
        self.delta = delta
        self.max_iter = max_iter

    def build(self, input_shape):
        self.batch_size = input_shape[0][0]
        self.ny, self.nx, self.nz = input_shape[0][1:4]
        
    def compute_gradients(self, I1):
        Ix = tf.zeros_like(I1)
        Iy = tf.zeros_like(I1)
        Ix = (I1[:, :, 2:, :] - I1[:, :, :-2, :]) * 0.5
        Ix = tf.pad(Ix, [[0,0], [0,0], [1,1], [0,0]])
        Iy = (I1[:, 2:, :, :] - I1[:, :-2, :, :]) * 0.5
        Iy = tf.pad(Iy, [[0,0], [1,1], [0,0], [0,0]])
        return Ix, Iy

    def jacobian(self):
        # Example for affine transform, implement other transforms as needed
        x = tf.range(self.nx, dtype=tf.float32)
        y = tf.range(self.ny, dtype=tf.float32)
        X, Y = tf.meshgrid(x, y)
        ones = tf.ones_like(X)
        zeros = tf.zeros_like(X)
        match self.transform_type:
            case tr.TransformType.TRANSLATION:
                J = tf.stack([ones, tf.zeros_like(X), tf.zeros_like(X), ones], axis=-1)
            case tr.TransformType.EUCLIDEAN:
                J = tf.stack([ones, tf.zeros_like(X), -Y, tf.zeros_like(X), ones, X], axis=-1)
            case tr.TransformType.SIMILARITY:
                J = tf.stack([ones, X, -Y, ones, Y, X], axis=-1)
            case tr.TransformType.AFFINITY:
                J = tf.stack([ones, zeros, X, Y, zeros, zeros, zeros, ones, zeros, zeros, X, Y], axis=-1)
            case tr.TransformationType.HOMOGRAPHY:
                J = tf.stack([X, Y, ones, zeros, zeros, zeros, -X*X, -X*Y, zeros, zeros, zeros, X, Y, ones, -X*Y, -Y*Y], axis=-1)
        print("--- J shape: ", J.shape)
        # J = J[..., :self.nparams]
        # print("----- J shape: ", J.shape)
        #TODO: fix bug J shape should be (ny, nx, 2*nparams) and not (ny, nx, nparams)
        return tf.expand_dims(J, 0)  # Add batch dimension

    def steepest_descent_images(self, Ix, Iy, J):
        # gradients = tf.stack([Ix, Iy], axis=-1)
        # return tf.einsum('bhwci,bhwji->bhwj', gradients, J) #TODO: Check if this is correct
        # Supposons que J a une taille (b, ny, nx, 2*m) avec m = nparams/2
        print("J shape: ", J.shape)
        Jx, Jy = tf.split(J, num_or_size_splits=2, axis=-1)  # Chaque tenseur a la taille (b, ny, nx, m)
        print("Jx shape: ", Jx.shape)
        print("Jy shape: ", Jy.shape)

        # Étendre Ix et Iy pour pouvoir multiplier par broadcast avec Jx et Jy.
        # Ix et Iy ont la taille (b, ny, nx, nz). On les étend sur la dernière dimension.
        Ix_exp = tf.expand_dims(Ix, axis=-1)  # (b, ny, nx, nz, 1)
        Iy_exp = tf.expand_dims(Iy, axis=-1)  # (b, ny, nx, nz, 1)
        print("Ix_exp shape: ", Ix_exp.shape)
        print("Iy_exp shape: ", Iy_exp.shape)

        # Reshape Jx et Jy pour qu'ils aient un axe "duplicate" pour nz.
        Jx_exp = tf.expand_dims(Jx, axis=2)  # (b, ny, 1, nx, m) ou réarranger selon l'ordre souhaité
        Jy_exp = tf.expand_dims(Jy, axis=2)  # (b, ny, 1, nx, m)
        print("Jx_exp shape: ", Jx_exp.shape)
        print("Jy_exp shape: ", Jy_exp.shape)

        # Pour correspondre aux dimensions, on peut réarranger Jx et Jy en (b, ny, nx, 1, m)
        Jx_exp = tf.reshape(Jx, [tf.shape(Jx)[0], tf.shape(Jx)[1], tf.shape(Jx)[2], 1, tf.shape(Jx)[3]])
        Jy_exp = tf.reshape(Jy, [tf.shape(Jy)[0], tf.shape(Jy)[1], tf.shape(Jy)[2], 1, tf.shape(Jy)[3]])
        print("Jx_exp shape: ", Jx_exp.shape)
        print("Jy_exp shape: ", Jy_exp.shape)

        # Alors, DIJ est défini par la somme des deux contributions :
        DIJ = Ix_exp * Jx_exp + Iy_exp * Jy_exp  # Résultat de taille (b, ny, nx, nz, m)
        print("DIJ shape: ", DIJ.shape)
        return DIJ

    def warp_image(self, I2, p):
        if self.transform_type in [tr.TransformType.TRANSLATION, tr.TransformType.EUCLIDEAN,
                                   tr.TransformType.AFFINITY, tr.TransformType.HOMOGRAPHY]:
            grid = self.transformed_grid(p)  # grid de forme [batch, self.ny, self.nx, 2]
            warped = bicubic_sampler(I2, grid)

            # Création d'un masque indiquant où I2 n'est pas NaN
            mask = tf.cast(~tf.math.is_nan(I2), I2.dtype)
            # Pour le masque, on utilise une interpolation en nearest neighbor.
            # Ici, nous pouvons utiliser tf.gather_nd sur les indices entiers.
            grid_int = tf.cast(tf.round(grid), tf.int32)
        
            def sample_mask(mask, grid_int):
                # Supposons que mask a la forme (batch, H, W, channels)
                # Créer une grille d'indices spatiaux (B, Y, X) de forme (batch, H, W)
                B, Y, X, C = tf.meshgrid(tf.range(tf.shape(mask)[0]), 
                                    tf.range(tf.shape(mask)[1]), 
                                    tf.range(tf.shape(mask)[2]), 
                                    tf.range(tf.shape(mask)[3]),
                                    indexing="ij")

                # Empiler pour obtenir des indices de forme (batch, H, W, channels, 4)
                indices = tf.stack([B, Y, X, C], axis=-1)
                # indices = tf.stack([B, Y, X], axis=-1)
                return tf.gather_nd(mask, indices)

            mask_warped = sample_mask(mask, grid_int)
            warped = tf.where(mask_warped < 0.5,
                                tf.constant(float('nan'), dtype=I2.dtype),
                                warped)
            return warped
        else:
            raise ValueError("Unsupported transformation type")

    def transformed_grid(self, p):
        x = tf.linspace(0.0, self.nx-1, self.nx)
        y = tf.linspace(0.0, self.ny-1, self.ny)
        X, Y = tf.meshgrid(x, y)
        ones = tf.ones_like(X)
        coords = tf.stack([X, Y, ones], axis=0)
        # Use of map_fn to apply the function params2matrix to each element of p (batch of parameters)
        affine_matrix = tf.map_fn(lambda params: tr.params2matrix(params, self.transform_type), p, dtype=tf.float32)
        transformed = tf.einsum('bij,jhw->bhw', affine_matrix, coords)
        
        return transformed

    def robust_error(self, DI):
        # apply our robust error function, considering that in input we may have a batch of images
        rho_list = []
        for image in tf.range(self.batch_size):
            rho = io.robust_error_function(DI.numpy()[image, :, :, :], self.lambda_, self.robust_type)
            # adding the robust error to the tensor of robust errors
            rho_list.append(rho)
        rho_tensor = tf.stack(rho_list, axis=0)
        return rho_tensor

    def call(self, inputs):
        # inputs are batches: every I1 in the batch is associated to a I2 in 
        # the batch and a p_init
        I1, I2, p = inputs
        J = self.jacobian()
        print("Jacobian shape: ", J.shape)
        print("I1 shape: ", I1.shape)
        print("I2 shape: ", I2.shape)
        Ix, Iy = self.compute_gradients(I1)
        print("Ix shape: ", Ix.shape)
        print("Iy shape: ", Iy.shape)
        DIJ = self.steepest_descent_images(Ix, Iy, J)
        print("DIJ shape: ", DIJ.shape)
        # DIJ = tf.squeeze(DIJ, axis=4)
        # print("DIJ shape: ", DIJ.shape)
        print("p shape: ", p.shape)
        
        def body(i, p, error):
            # Warp I2 with current parameters
            Iw = self.warp_image(I2, p)
            DI = Iw - I1
            
            # Compute robust error
            rho = self.robust_error(DI)
            print("rho shape: ", rho.shape)
            # rho = tf.squeeze(rho, axis=2)
            # print("rho shape: ", rho.shape)
            
            # Compute Hessian and b
            weighted_DIJ = DIJ * tf.expand_dims(rho, -1)
            DIJt = tf.einsum("bhwcn->bhwnc", DIJ)
            #TODO: fix the calculation of H and b by following the numpy implementation
            H = tf.einsum('bhwij,bhwik->bjk', DIJ, weighted_DIJ)
            b = tf.einsum('bhwij,bhw->bj', DIJ, DI * rho)
            
            # Solve for dp
            H_inv = tf.linalg.inv(H)
            dp = tf.einsum('bij,bj->bi', H_inv, b)  # dp is a batch of updates
            
            # Update parameters
            p = p - dp
            error = tf.norm(dp) # error is a batch of update norms
            return i+1, p, error
        
        def cond(i, p, error):
            # We assume that if a batch of images is provided to the layer,
            # images are homogenous in the sense that they originate from the
            # same scene and are captured by the same sensor. Therefore, we
            # consider that the convergence criteria is the same for all
            # images and we include the error of all images in the batch to
            # determine the convergence
            return (i < self.max_iter) & (error > self.TOL) 
            # For now, we choose to process batch of images, and executing the maximum number of iterations
            # TODO: state on the convergence criteria - we may consider that the batch of images is homogenous and that the convergence criteria is the same for all images
            # return i < self.max_iter

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
    def __init__(self, 
                    transform_type: tr.TransformType, # typeof transformation
                    nscales: int=3, # number of scales
                    nu: float=0.5,      # downsampling factor
                    TOL: float=1e-3,     # stopping criterion threshold
                    robust_type: io.RobustErrorFunctionType=io.RobustErrorFunctionType.CHARBONNIER,  # type of robust error function
                    lambda_: float=0.0,  # parameter of robust error function
                    nanifoutside: bool=True, # if True, the pixels outside the image are considered as NaN
                    delta: int=10, # maximal distance to boundary to consider the pixel as NaN
                    verbose: bool=True,  # switch on messages
                    **kwargs):
        super(PyramidalInverseCompositional, self).__init__(trainable=False, **kwargs)
        self.nscales = nscales
        self.nu = nu
        self.ric_layers = [RobustInverseCompositional(
                                transform_type,
                                TOL,
                                robust_type,
                                lambda_,
                                nanifoutside,
                                delta,
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
        p = tf.zeros((self.nscales, self.batch_size, self.ric_layers[0].nparams))

        # Build pyramids
        I1_pyramid = [I1]
        I2_pyramid = [I2]
        for i in range(1, self.nscales):
            I1_pyramid.append(tf.image.resize(I1_pyramid[-1], self.pyramid_shapes[i][:2]))
            I2_pyramid.append(tf.image.resize(I2_pyramid[-1], self.pyramid_shapes[i][:2]))
        
        # Process from coarse to fine
        for scale in reversed(range(self.nscales)):
            ric_layer = self.ric_layers[scale]
            print("shape of p[scale]: ", p[scale].shape)
            p[scale], error, DI, Iw = ric_layer([I1_pyramid[scale], I2_pyramid[scale], p[scale]])

            # Upscale parameters for next scale
            if scale > 0:
                for b in self.batch_size:
                    p[scale-1][b] = zm.zoom_in_parameters(p[scale][b],
                                            self.transform_type,
                                            self.pyramid_shapes[scale][2],
                                            self.pyramid_shapes[scale][1],
                                            self.pyramid_shapes[scale-1][2],
                                            self.pyramid_shapes[scale-1][1])
                # p = self.upscale_parameters(p,
                #                             self.pyramid_shapes[scale],
                #                             self.pyramid_shapes[scale-1])

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