import numpy as np
import matplotlib.pyplot as plt
from .core import Solver
from scipy import integrate
import tikzplotlib

dx = 1
sigma = 1
M = 1

# plt.style.use("grayscale")
# plt.style.use("seaborn-deep")


def init_diffeo(rho0, a, b):
    @np.vectorize
    def Rho0(upper):
        return integrate.quad(rho0, a, upper)[0]
    M = Rho0(b)
    # Omega_tilde := [0, M] (equidistant)
    Omega_tilde = np.linspace(0, M, N)
    # Initial guess of Omega = Phi0(Omega_tilde)
    Omega = np.linspace(a, b, N)

    unfulfill = np.full_like(Omega, True, dtype=bool)
    Omega[0], Omega[-1] = a, b
    unfulfill[0], unfulfill[-1] = False, False
    while unfulfill.any():
        # Correction
        invalid = (Omega < a) | (b < Omega)
        Omega[invalid] = np.random.uniform(a, b, invalid.sum())
        # Increment (where unfulfilled)
        inc = - (Rho0(Omega[unfulfill]) -
                 Omega_tilde[unfulfill]) / rho0(Omega[unfulfill])
        Omega[unfulfill] += inc
        unfulfill[unfulfill] = (np.abs(inc) >= 1e-8)
    return Omega, M


def rho0(x): return M / np.sqrt(2 * np.pi * sigma ** 2) * \
    np.exp(- x ** 2 / 2 / sigma ** 2)


def rho1(x):
    res = np.ones_like(x)
    res[x < 0] = 0
    return res


def rho2(x):
    return 2 * x


def rho3(x):
    return rho0(x)


def recover(Phi):
    return Phi


N = 400
# solver = Solver(rho2, N, 0, 0, 0, 0, 0, 0, 0, 0, 0)
# Phi = next(solver.step(-6, 6, 1))
Phi, M = init_diffeo(rho2, 0, 1)
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\Phi(\eta)$')
plt.plot(np.linspace(0, M, N), Phi)
# plt.plot([0, 0, 6], [-6, 0, 6])
tikzplotlib.clean_figure()
tikzplotlib.save('Phi2.tikz')

plt.figure()
x = np.linspace(-0, 1, N)
plt.plot(x, rho2(x))
plt.xlabel(r'$x$')
plt.ylabel(r'$\rho(x)$')
tikzplotlib.clean_figure()
tikzplotlib.save('rho2.tikz')

h = np.sqrt(5 / 2)
a = h / 2 / 5
print(np.sqrt(2 / 5))
plt.figure()
plt.plot([-4 * a, -3 * a, -3 * a, -a, -a, a, a, 3 * a, 3 * a, 4 * a], [0, 0, h, h, 0, 0, h, h, 0, 0])
plt.xlabel(r'$x$')
plt.ylabel(r'$\rho(x)$')
tikzplotlib.clean_figure()
tikzplotlib.save('rho-wave.tikz')

plt.figure()
plt.plot([0, 0, 2 * a * h, 2 * a * h, 4 * a * h , 4 * a * h], [-4 * a, -3 * a, -a, a, 3 * a, 4 * a])
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\Phi(\eta)$')
tikzplotlib.clean_figure()
tikzplotlib.save('Phi-wave.tikz')

plt.show()
