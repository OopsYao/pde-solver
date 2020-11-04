import numpy as np
from scipy import integrate


def derivative(func_sample, x):
    """ Differentiation along with the last axis """
    return np.gradient(func_sample, x, edge_order=2, axis=-1)


def integral(func_sample, x):
    """ Integrate along with the last axis """
    return integrate.simps(func_sample, x)
