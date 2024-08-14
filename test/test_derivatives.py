import numpy as np
import os, sys

sys.path.append(os.path.abspath("../src/"))

import derivatives as der

def test_jacobian():
    nx = 2
    ny = 2
    transform_type = der.TransformType.TRANSLATION
    J = der.jacobian(transform_type, nx, ny)
    print("--------------------")
    print("nx, ny", nx, ny)
    print("Transform type", transform_type)
    print(J)
    assert J.shape == (ny, nx, 2 * transform_type.nparams())
    # assert np.all(J == np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))

    transform_type = der.TransformType.EUCLIDEAN
    J = der.jacobian(transform_type, nx, ny)
    print("--------------------")
    print("nx, ny", nx, ny)
    print("Transform type", transform_type)
    print(J)
    assert J.shape == (ny, nx, 2 * transform_type.nparams())
    # assert np.all(J == np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]))

    transform_type = der.TransformType.SIMILARITY
    J = der.jacobian(transform_type, nx, ny)
    print("--------------------")
    print("nx, ny", nx, ny)
    print("Transform type", transform_type)
    print(J)
    assert J.shape == (ny, nx, 2 * transform_type.nparams())
    # assert np.all(J == np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]]))

    transform_type = der.TransformType.AFFINITY
    J = der.jacobian(transform_type, nx, ny)
    print("--------------------")
    print("nx, ny", nx, ny)
    print("Transform type", transform_type)
    print(J)
    assert J.shape == (ny, nx, 2 * transform_type.nparams())

    nx = 2
    ny = 3
    transform_type = der.TransformType.TRANSLATION
    J = der.jacobian(transform_type, nx, ny)
    print("--------------------")
    print("nx, ny", nx, ny)
    print("Transform type", transform_type)
    print(J)
    assert J.shape == (ny, nx, 2 * transform_type.nparams())
    # assert np.all(J == np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))

    transform_type = der.TransformType.EUCLIDEAN
    J = der.jacobian(transform_type, nx, ny)
    print("--------------------")
    print("nx, ny", nx, ny)
    print("Transform type", transform_type)
    print(J)
    assert J.shape == (ny, nx, 2 * transform_type.nparams())
    # assert np.all(J == np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]))

    transform_type = der.TransformType.SIMILARITY
    J = der.jacobian(transform_type, nx, ny)
    print("--------------------")
    print("nx, ny", nx, ny)
    print("Transform type", transform_type)
    print(J)
    assert J.shape == (ny, nx, 2 * transform_type.nparams())
    # assert np.all(J == np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, -1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]]))

    transform_type = der.TransformType.AFFINITY
    J = der.jacobian(transform_type, nx, ny)
    print("--------------------")
    print("nx, ny", nx, ny)
    print("Transform type", transform_type)
    print(J)
    assert J.shape == (ny, nx, 2 * transform_type.nparams())


def main():
    print("Running tests")
    test_jacobian()

if __name__ == '__main__':
    main()