import numpy as np
from enum import Enum
from skimage import transform

class TransformType(Enum):
    TRANSLATION = 1
    EUCLIDEAN = 2
    SIMILARITY = 3
    AFFINITY = 4
    HOMOGRAPHY = 5

    def nparams(self):
        """
        Returns the number of parameters for the given transformation type.
        """
        match self:
            case TransformType.TRANSLATION:
                return 2
            case TransformType.EUCLIDEAN:
                return 3
            case TransformType.SIMILARITY:
                return 4
            case TransformType.AFFINITY:
                return 6
            case TransformType.HOMOGRAPHY:
                return 8
            case _:
                raise ValueError("Unknown transform type")


def update_transform(p, dp, transform_type):
    """
    Update the transformation parameters based on the given type of transformation.
    Args:
        p (numpy.ndarray): The current transformation parameters.
        dp (numpy.ndarray): The update to be applied to the transformation parameters.
        transform_type (TransformType): The type of transformation.
    Returns:
        numpy.ndarray: The updated transformation parameters.
    Raises:
        None
    """
    match transform_type:
        case TransformType.TRANSLATION:
            nparams = 2
            p[:nparams] -= dp[:nparams]
        
        case TransformType.EUCLIDEAN:
            nparams = 3
            a = np.cos(dp[2])
            b = np.sin(dp[2])
            c = dp[0]
            d = dp[1]
            ap = np.cos(p[2])
            bp = np.sin(p[2])
            cp = p[0]
            dp_val = p[1]
            cost = a * ap + b * bp
            sint = a * bp - b * ap
            p[0] = cp - bp * (b * c - a * d) - ap * (a * c + b * d)
            p[1] = dp_val - bp * (a * c + b * d) + ap * (b * c - a * d)
            p[2] = np.arctan2(sint, cost)
        
        case TransformType.SIMILARITY:
            nparams = 4
            a = dp[2]
            b = dp[3]
            c = dp[0]
            d = dp[1]
            det = (2 * a + a * a + b * b + 1)
            if det * det > 1E-10:
                ap = p[2]
                bp = p[3]
                cp = p[0]
                dp_val = p[1]
                p[0] = cp - bp * (-d - a * d + b * c) / det + (ap + 1) * (-c - a * c - b * d) / det
                p[1] = dp_val + bp * (-c - a * c - b * d) / det + (ap + 1) * (-d - a * d + b * c) / det
                p[2] = b * bp / det + (a + 1) * (ap + 1) / det - 1
                p[3] = -b * (ap + 1) / det + bp * (a + 1) / det
        
        case TransformType.AFFINITY:
            nparams = 6
            a = dp[2]
            b = dp[3]
            c = dp[0]
            d = dp[4]
            e = dp[5]
            f = dp[1]
            det = (a - b * d + e + a * e + 1)
            if det * det > 1E-10:
                ap = p[2]
                bp = p[3]
                cp = p[0]
                dp_val = p[4]
                ep = p[5]
                fp = p[1]
                p[0] = cp + (-f * bp - a * f * bp + c * d * bp) / det + (ap + 1) * (-c + b * f - c * e) / det
                p[1] = fp + dp_val * (-c + b * f - c * e) / det + (-f + c * d - a * f - f * ep - a * f * ep + d * d * ep) / det
                p[2] = ((1 + ap) * (1 + e) - d * bp) / det - 1
                p[3] = (bp + a * bp - b - b * ap) / det
                p[4] = (dp_val * (1 + e) - d - d * ep) / det
                p[5] = (a + ep + a * ep + 1 - b * dp_val) / det - 1
        
        case TransformType.HOMOGRAPHY:
            nparams = 8
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
            if det * det > 1E-10:
                p[0] = ((d * bp - f * g * bp) + cp * (g - d * h + g * e) + (ap + 1) * (f * h - e - 1)) / det - 1
                p[1] = (h * cp + a * h * cp - b * g * cp - bp - a * bp + c * g * bp + b - c * h + b * ap - c * h * ap) / det
                p[2] = (f * bp + a * f * bp - c * d * bp + (ap + 1) * (c - b * f + c * e) + cp * (-a + b * d - e - a * e - 1)) / det
                p[3] = (fp * (g - d * h + g * e) + d - f * g + d * ep - f * g * ep + dp_val * (f * h - e - 1)) / det
                p[4] = (b * dp_val - c * h * dp_val + h * fp + a * h * fp - b * g * fp - ep - a * ep + c * g * ep - 1) / det - 1
                p[5] = (dp_val * (c - b * f + c * e) + f + a * f - c * d + f * ep + a * f * ep - c * d * ep + fp * (-a + b * d - e - a * e - 1)) / det
                p[6] = (d * hp - f * g * hp + g - d * h + g * e + gp * (f * h - e - 1)) / det
                p[7] = (h + a * h - b * g + b * gp - c * h * gp - hp - a * hp + c * g * hp) / det
    
    return p


def project(x, y, p, transform_type):
    """
    Applies a transformation to the given coordinates (x, y) based on the specified transform type and parameters.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        p (tuple): The transformation parameters.
        transform_type (TransformType): The type of transformation.

    Returns:
        tuple: The transformed coordinates (xp, yp).
    """
    if transform_type == TransformType.TRANSLATION:
        # p = (tx, ty)
        xp = x + p[0]
        yp = y + p[1]
    elif transform_type == TransformType.EUCLIDEAN:
        # p = (tx, ty, theta)
        xp = np.cos(p[2]) * x - np.sin(p[2]) * y + p[0]
        yp = np.sin(p[2]) * x + np.cos(p[2]) * y + p[1]
    elif transform_type == TransformType.SIMILARITY:
        # p = (tx, ty, a, b)
        xp = (1 + p[2]) * x - p[3] * y + p[0]
        yp = p[3] * x + (1 + p[2]) * y + p[1]
    elif transform_type == TransformType.AFFINITY:
        # p = (tx, ty, a00, a01, a10, a11)
        xp = (1 + p[2]) * x + p[3] * y + p[0]
        yp = p[4] * x + (1 + p[5]) * y + p[1]
    elif transform_type == TransformType.HOMOGRAPHY:
        # p = (h00, h01, ..., h21)
        d = p[6] * x + p[7] * y + 1
        xp = ((1 + p[0]) * x + p[1] * y + p[2]) / d
        yp = (p[3] * x + (1 + p[4]) * y + p[5]) / d
    else:
        raise ValueError("Invalid transformation type")
    
    return xp, yp

def params2matrix(p, transform_type):
    """
    Converts the given parameters `p` into a transformation matrix based on the specified `transform_type`.

    Parameters:
        p (list): The parameters for the transformation.
        transform_type (TransformType): The type of transformation.

    Returns:
        numpy.ndarray: The transformation matrix.

    """
    matrix = np.identity(3)
    
    if transform_type == TransformType.TRANSLATION:
        matrix[0, 2] = p[0]
        matrix[1, 2] = p[1]
    elif transform_type == TransformType.EUCLIDEAN:
        matrix[0, 0] = np.cos(p[2])
        matrix[0, 1] = -np.sin(p[2])
        matrix[0, 2] = p[0]
        matrix[1, 0] = np.sin(p[2])
        matrix[1, 1] = np.cos(p[2])
        matrix[1, 2] = p[1]
    elif transform_type == TransformType.SIMILARITY:
        matrix[0, 0] = 1 + p[2]
        matrix[0, 1] = -p[3]
        matrix[0, 2] = p[0]
        matrix[1, 0] = p[3]
        matrix[1, 1] = 1 + p[2]
        matrix[1, 2] = p[1]
    elif transform_type == TransformType.AFFINITY:
        matrix[0, 0] = 1 + p[2]
        matrix[0, 1] = p[3]
        matrix[0, 2] = p[0]
        matrix[1, 0] = p[4]
        matrix[1, 1] = 1 + p[5]
        matrix[1, 2] = p[1]
    elif transform_type == TransformType.HOMOGRAPHY:
        matrix[0, 0] = 1 + p[0]
        matrix[0, 1] = p[1]
        matrix[0, 2] = p[2]
        matrix[1, 0] = p[3]
        matrix[1, 1] = 1 + p[4]
        matrix[1, 2] = p[5]
        matrix[2, 0] = p[6]
        matrix[2, 1] = p[7]
    
    return matrix


def transform_image(image, transformation_type, gt):
    """
    Transforms an image based on the specified transformation type and geometric transformation parameters.

    Parameters:
    - `image` (ndarray): The input image to be transformed.
    - `transformation_type` (TransformType): The type of transformation to apply. It can be one of the following:
    - `TransformType.AFFINITY`: Affine transformation.
    - `TransformType.SIMILARITY`: Similarity transformation.
    - `TransformType.EUCLIDEAN`: Euclidean transformation.
    - `gt` (list or array): The geometric transformation parameters. The expected format depends on the `transformation_type`:
        - For `TransformType.AFFINITY`: [tx, ty, a00, a01, a10, a11]
        - For `TransformType.SIMILARITY`: [tx, ty, scale, rotation]
        - For `TransformType.EUCLIDEAN`: [tx, ty, theta]
        - angles of rotation are in radians counter-clockwise.
    """
    if all(np.abs(value) < 1e-10 for value in gt):
        # Identity transformation
        tform = transform.AffineTransform(matrix=np.eye(3))
    else:
        hmatrix = params2matrix(gt, transformation_type)
        if transformation_type == TransformType.AFFINITY:
            tform = transform.AffineTransform(matrix=hmatrix)
        elif transformation_type == TransformType.SIMILARITY:
            # Similarity transformation, we expect gt as [tx, ty, a, b]
            # skimage considers the rotation in clockwise direction, so we need to negate it
            # tform = transform.SimilarityTransform(scale=gt[2], rotation=gt[3], translation=(gt[0], gt[1]))
            tform = transform.SimilarityTransform(matrix=hmatrix)
        elif transformation_type == TransformType.EUCLIDEAN:
            # Euclidean transformation, we expect gt as [tx, ty, theta]
            # skimage considers the rotation in clockwise direction, so we need to negate it
            tform = transform.EuclideanTransform(rotation=gt[2], translation=(gt[0], gt[1]))
            # tform = transform.EuclideanTransform(matrix=hmatrix)
        elif transformation_type == TransformType.HOMOGRAPHY:
            tform = transform.ProjectiveTransform(matrix=hmatrix)
        else:
            raise ValueError("Unsupported transformation type")

    print("gt", gt)
    print("tform", tform.params)
    transformed_image = transform.warp(image, tform.inverse)
    return transformed_image