import numpy as np
from .core import Solver
from tqdm import tqdm
import matplotlib.pyplot as plt

M0 = 2 * np.pi + 0.1
sigma = 1
chi = 1
N = 301
dt = 1e-3
t_star = 16.818399684193


def rho0(x): return M0 / np.sqrt(2 * np.pi * sigma ** 2) * \
    np.exp(- x ** 2 / 2 / sigma ** 2)


def rho01(x): return np.sqrt(600 / 2 / np.pi) * (
    (2 * np.pi - 0.1) * np.exp(-600 * (x - 2) ** 2 / 2) + (2 * np.pi - 0.5) * np.exp(-600 * (x + 2) ** 2 / 2))


def rho02(x): return np.sqrt(600 / 2 / np.pi) * (
    (2 * np.pi + 0.1) * np.exp(-600 * (x - 2) ** 2 / 2) + (2 * np.pi - 0.5) * np.exp(-600 * (x + 2) ** 2 / 2))


def U(x): return x * np.log(x) - x
def V(x): return np.zeros_like(x)
def W(x): return chi / np.pi * np.log(np.abs(x))


def U_p(x): return np.log(x)
def V_p(x): return np.zeros_like(x)
def W_p(x): return chi / np.pi / x


def U_pp(x): return 1 / x
def V_pp(x): return np.zeros_like(x)
def W_pp(x): return - chi / np.pi / x ** 2


solver = Solver(rho01, N, U, U_p, U_pp, V, V_p, V_pp, W, W_p, W_pp)
t_arr = dt * np.arange(1801)
x = np.linspace(-5, 5, N)
plt.plot(x, rho01(x), label='Initial value')
for i, (t, Phi) in enumerate(zip(tqdm(t_arr), solver.step(-5, 5, dt))):
    if i in [len(t_arr) - 1]:
        plt.plot(Phi, solver.recover(Phi), 'x-', label=r'Computed solution')
plt.legend()
plt.show()
