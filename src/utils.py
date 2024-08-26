import numpy as np

def valid_values(array):
    """
    Function to check if the array contains NaN or Inf values

    Parameters:
    array (numpy.ndarray): input array

    Returns:
    bool: True if the array contains NaN or Inf values, False otherwise
    """
    if np.any(np.isnan(array)):
        return False
    elif np.any(np.isinf(array)):
        return False
    else: 
        return True


