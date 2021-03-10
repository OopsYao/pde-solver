import numpy as np
from .core import Solver
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad, dblquad

# Parameters
a = 2
b = 0
sigma = 1
M = 1
N = 501
dt = 1e-3


def rho0(x): return M / np.sqrt(2 * np.pi * sigma ** 2) * \
    np.exp(- x ** 2 / 2 / sigma ** 2)


def rho_inf(x):
    if b == 0:
        mask = np.abs(x) <= np.sqrt(2)
        r = np.zeros_like(x)
        x_mask = x[mask] if type(x) == np.ndarray else x
        r[mask] = np.sqrt(2 - x_mask ** 2) / np.pi
        return r
    else:
        mask = np.abs(x) <= 1
        r = np.zeros_like(x)
        r[mask] = 2
        return r


def U(x): return np.zeros_like(x)
def V(x): return np.zeros_like(x)


def W(x):
    if b == 0:
        return np.abs(x) ** a / a - np.log(np.abs(x))
    else:
        return np.abs(x) ** a / a - np.abs(x) ** b / b


def U_p(x): return np.zeros_like(x)
def V_p(x): return np.zeros_like(x)
def W_p(x): return np.sign(x) * (np.abs(x) ** (a - 1) - np.abs(x) ** (b - 1))


def U_pp(x): return np.zeros_like(x)
def V_pp(x): return np.zeros_like(x)


def W_pp(x): return (a - 1) * np.abs(x) ** (a - 2) - \
    (b - 1) * np.abs(x) ** (b - 2)


solver = Solver(rho0, N, U, U_p, U_pp, V, V_p, V_pp, W, W_p, W_pp)
t_arr = dt * np.arange(4892)
en = np.zeros_like(t_arr)
x = np.linspace(-6, 6, N)
plt.plot(x, rho0(x), label='init')
plt.plot(x, rho_inf(x), label='inf')
for i, (t, Phi) in enumerate(zip(tqdm(t_arr), solver.step(-6, 6, dt))):
    if i in [0, len(t_arr) - 1]:
        plt.plot(Phi, solver.recover(Phi), label=f't={t}')
    en[i] = solver.entropy(Phi)
plt.legend()

xy = np.expand_dims(x, -1) - x
mask = ~np.eye(N, dtype=bool)
Wxy = np.zeros_like(xy)
Wxy[mask] = W(xy[mask])
rho_x = rho_inf(x)
E_inf = (Wxy * np.expand_dims(rho_x, -1) * rho_x).sum() * (12 / N) ** 2 / 2
plt.figure()
plt.yscale('log')
plt.plot(t_arr, en - E_inf, label='entropy')
plt.plot(t_arr, np.exp(-3 * t_arr), label='expected decay')
plt.show()
