import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad
from scipy.optimize import newton
from .core import Solver

# Parameters
dt = 1e-3
a = -2
b = 2
nu = 0.05
N = 501

# internal energy


def U(x): return x ** 2 / 2 * nu
# potential
def V(x): return x ** 4 / 4 - x ** 2 / 2
# interaction potential
def W(x): return np.zeros_like(x)
# initial density
def rho0(x): return np.exp(- x ** 2 / 2 / 0.2 ** 2) / \
    np.sqrt(2 * np.pi * 0.2 ** 2)

# Positive part


def p(x): return (x + np.abs(x)) / 2


def Psi(x): return x * U(1 / x)

# Derivatives


def U_p(x): return x * nu
def U_pp(x): return nu
def V_p(x): return x ** 3 - x
def V_pp(x): return 3 * x ** 2 - 1
def W_p(x): return np.zeros_like(x)
def W_pp(x): return np.zeros_like(x)

# Determine rho_inf


def rho_inf(C, x):
    return p(C - V(x)) / nu

solver = Solver(rho0, N, U, U_p, U_pp, V, V_p, V_pp, W, W_p, W_pp)

C = newton(lambda C: quad(lambda x: rho_inf(C, x), -np.inf, np.inf)[0] - solver.M, 0)
def rho_inff(x): return rho_inf(C, x)


Einf = quad(lambda x: U(rho_inff(x)) + V(x) * rho_inff(x), -np.inf, np.inf)[0]

t_arr = dt * np.arange(0, 9718)
en = np.zeros_like(t_arr)

x = np.linspace(a, b, N)
plt.plot(x, rho0(x), label=r'init')
for i, (t, Phi) in enumerate(zip(tqdm(t_arr), solver.step(a, b, dt))):
    en[i] = solver.entropy(Phi)
    if i in [0, len(t_arr) - 1]:
        plt.plot(Phi, solver.recover(Phi), label=f't={t}')
plt.plot(x, rho_inff(x), label=r'inf')
plt.legend()
plt.figure()
plt.plot(t_arr, en - Einf)
plt.show()
