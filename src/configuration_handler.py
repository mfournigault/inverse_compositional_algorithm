import configparser
from transformation import TransformType
from image_optimisation import RobustErrorFunctionType

def create_config_file(filename):
    config = configparser.ConfigParser()

    config['InverseCompositionalAlgorithm'] = {
        'TOL': '1e-3',
        'transform_type': 'EUCLIDEAN', #TRANSLATION, EUCLIDEAN, SIMILARITY, AFFINITY, HOMOGRAPHY
        'verbose': 'False'
    }

    config['RobustInverseCompositionalAlgorithm'] = {
        'TOL': '1e-3',
        'transform_type': 'EUCLIDEAN', #TRANSLATION, EUCLIDEAN, SIMILARITY, AFFINITY, HOMOGRAPHY
        'robust_type': 'CHARBONNIER', #QUADRATIC, TRUNCATED_QUADRATIC, GERMAN_MCCLURE, LORENTZIAN, CHARBONNIER
        'lambda': '0.0',
        'verbose': 'False'
    }

    config['PyramidalInverseCompositionalAlgorithm'] = {
        'TOL': '1e-3',
        'transform_type': 'EUCLIDEAN', #TRANSLATION, EUCLIDEAN, SIMILARITY, AFFINITY, HOMOGRAPHY
        'pyramid_levels': '2',
        'nu': '0.5',
        'robust_type': 'QUADRATIC', #QUADRATIC, TRUNCATED_QUADRATIC, GERMAN_MCCLURE, LORENTZIAN, CHARBONNIER
        'lambda': '0.0',
        'verbose': 'False'
    }

    with open(filename, 'w') as configfile:
        config.write(configfile)

def read_config_file(filename):
    config = configparser.ConfigParser()
    config.read(filename)

    inverse_compositional_params = {
        'TOL': float(config['InverseCompositionalAlgorithm']['TOL']),
        'transform_type': TransformType[config['InverseCompositionalAlgorithm']['transform_type']],
        'verbose': config.getboolean('InverseCompositionalAlgorithm', 'verbose')
    }

    robust_inverse_compositional_params = {
        'TOL': float(config['RobustInverseCompositionalAlgorithm']['TOL']),
        'transform_type': TransformType[config['RobustInverseCompositionalAlgorithm']['transform_type']],
        'robust_type': RobustErrorFunctionType[config['RobustInverseCompositionalAlgorithm']['robust_type']],
        'lambda': float(config['RobustInverseCompositionalAlgorithm']['lambda']),
        'verbose': config.getboolean('RobustInverseCompositionalAlgorithm', 'verbose')
    }

    pyramidal_inverse_compositional_params = {
        'TOL': float(config['PyramidalInverseCompositionalAlgorithm']['TOL']),
        'transform_type': TransformType[config['PyramidalInverseCompositionalAlgorithm']['transform_type']],
        'pyramid_levels': int(config['PyramidalInverseCompositionalAlgorithm']['pyramid_levels']),
        'nu': float(config['PyramidalInverseCompositionalAlgorithm']['nu']),
        'robust_type': RobustErrorFunctionType[config['PyramidalInverseCompositionalAlgorithm']['robust_type']],
        'lambda': float(config['PyramidalInverseCompositionalAlgorithm']['lambda']),
        'verbose': config.getboolean('PyramidalInverseCompositionalAlgorithm', 'verbose')
    }

    return {
        'inverse_compositional_algorithm': inverse_compositional_params,
        'robust_inverse_compositional_algorithm': robust_inverse_compositional_params,
        'pyramidal_inverse_compositional_algorithm': pyramidal_inverse_compositional_params
    }

# Example usage:
# create_config_file('config.ini')
# params = read_config_file('config.ini')
# print(params)