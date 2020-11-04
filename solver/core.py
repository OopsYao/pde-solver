import numpy as np
import scipy.integrate as integrate
from .utils import derivative, integral

x = np.linspace(-1, 1, 501)
dt = 10 ** (-3)


# internal energy
def U(x): return x ** 2 / 2
# potential
def V(x): return 0
# interaction potential
def W(x): return 0
# initial density
def rho0(x): return 2 / np.sqrt(2 * np.pi) * np.exp(- x ** 2 / 2)
def Psi(x): return x * U(1 / x)


def step():
    """ yield the diffeomorphism Phi at every time step """
    # initial diffeomorphism
    def init(rho0, x):
        if isinstance(rho0, (list, tuple, np.ndarray)):
            # density sample
            pass
        else:
            # density function
            Phi = np.zeros_like(x)
            eps = 10 ** (-4)
            incre = np.ones_like(x)
            # TODO can be optmized?
            while (np.abs(incre) > eps).any():
                incre = - (integrate.quad(rho0, 0, Phi) - x) / rho0(x)
                Phi = Phi + incre
            return rho0(x)
    Phi = init(rho0, x)
    yield Phi

    # every time step
    while True:
        D_Phi = derivative(Phi, x)
        Phi_xy = np.expand_dims(Phi, -1) - Phi
        inc = derivative(derivative(Psi(D_Phi), D_Phi) *
                         D_Phi, x) - derivative(V(Phi), Phi) + integral(derivative(W(Phi_xy), Phi_xy), x)
        Phi = Phi + dt * inc
        yield Phi


def recover(Phi):
    """ recover the density rho (and its underlying x, here it is Phi) """
    return 1 / derivative(Phi, x), Phi
