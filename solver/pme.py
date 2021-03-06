from .core import Solver
import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad
from tqdm import tqdm
import matplotlib.pyplot as plt

# Parameters
m = 4
t0 = 1e-3
alpha = 1 / (m + 1)
dt = 1e-5
N = 501


def pp(x): return (x + np.abs(x)) / 2

def bpp_c(c, x, t):
    return pp(c - alpha * (m - 1) / 2 / m * x ** 2 / t ** (2 * alpha)) ** (1 / (m - 1)) / t ** alpha


# Determine c
c = newton(lambda c: quad(lambda x: bpp_c(c, x, t0), -1, 1)[0] - 2, 0)
def rho0(x): return bpp_c(c, x, t0)
def bpp(x, t): return bpp_c(c, x, t)


# Support boundary
s0 = t0 ** alpha * np.sqrt(2 * m * c / alpha / (m - 1))


def U(x): return x ** m / (m - 1)
def V(x): return np.zeros_like(x)
def W(x): return np.zeros_like(x)


def U_p(x): return x ** (m - 1) * m / (m - 1)
def V_p(x): return np.zeros_like(x)
def W_p(x): return np.zeros_like(x)


def U_pp(x): return x ** (m - 2) * m  # In Python, 0 ** 0 == 0
def V_pp(x): return np.zeros_like(x)
def W_pp(x): return np.zeros_like(x)


solver = Solver(rho0, N, U, U_p, U_pp, V, V_p, V_pp, W, W_p, W_pp)
t_arr = dt * np.arange(t0 / dt, 2101)
x = np.linspace(-1, 1, N)
en = np.zeros_like(t_arr)
plt.plot(x, rho0(x), label='Initial value')
for i, (t, Phi) in enumerate(zip(tqdm(t_arr), solver.step(-s0, s0, dt))):
    if i in [len(t_arr) - 1]:
        plt.plot(Phi, solver.recover(Phi), 'x-', label='Computed solution')
    en[i] = solver.entropy(Phi)
plt.plot(x, bpp(x, t_arr[-1]), label=f'BPP at t={t_arr[-1]}')
plt.legend()

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(t_arr, en, label="Entropy")
plt.plot(t_arr, t_arr ** (- alpha * (m - 1)), label="Expected decay")
plt.xlabel('t')
plt.ylabel('E')
plt.legend()
plt.show()
