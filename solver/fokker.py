import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad
from scipy.optimize import newton
from .core import Solver
from .utils import save_tikz

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

C = newton(lambda C: quad(lambda x: rho_inf(
    C, x), -np.inf, np.inf)[0] - solver.M, 0)


def rho_inff(x): return rho_inf(C, x)


def plot_rho(Phi, *args, **kwargs):
    plt.figure('rho')
    plt.plot(Phi, solver.recover(Phi), *args, **kwargs)


tilde_Omega = np.linspace(0, 1, N)


def plot_Phi(Phi, *args, **kwargs):
    plt.figure('Phi')
    plt.plot(tilde_Omega, Phi, *args, **kwargs)


Einf = quad(lambda x: U(rho_inff(x)) + V(x) * rho_inff(x), -np.inf, np.inf)[0]

t_arr = dt * np.arange(0, 9718)
Phi_arr = np.empty((len(t_arr), N))

try:
    Phi_arr = np.load(f'fokker-nu={nu}.npy')
except:
    for i, (t, Phi) in enumerate(zip(tqdm(t_arr), solver.step(a, b, dt))):
        Phi_arr[i] = Phi
    np.save(f'fokker-nu={nu}.npy', Phi_arr)

x = np.linspace(a, b, N)
plt.figure('rho')
plt.plot(x, rho0(x), '--', label=r'初始密度$\rho_0$')
plot_Phi(Phi_arr[0], '--', label=f'$t=0$')

for i in [1000]:
    label = f'$t={t_arr[i]:.3f}$'
    plot_Phi(Phi_arr[i], '-.', label=label)
    plot_rho(Phi_arr[i], '-.', label=label)

if nu == 2:
    idx = np.arange(3, N - 3, 20)
    idx = [*np.arange(3), *idx, *np.arange(N - 3, N)]
else:
    idx1 = np.arange(3, N // 2 - 2, 10)
    idx2 = np.arange(N // 2 + 2, N - 3, 10)
    idx = [*np.arange(3), *idx1, N // 2 - 1, N // 2, N // 2 + 1, *idx2, *np.arange(N - 3, N)]
    # idx = np.arange(N)

plot_Phi(Phi_arr[-1], '-', label=f'$t={t_arr[-1]:.3f}$')
# plot_rho(Phi_arr[-1], 'x-', label=f'$t={t_arr[-1]:.3f}$')
plt.figure('rho')
plt.plot(Phi_arr[-1][idx], solver.recover(Phi_arr[-1])[idx], 'x-', label=f'$t={t_arr[-1]:.3f}$')

plt.figure('rho')
plt.plot(x, rho_inff(x), label=r'稳态解')

# Post plot
plt.figure('Phi')
plt.xlabel(r'$\tilde x$')
plt.legend()
save_tikz(f'fokker-nu={nu}-Phi.tikz')

plt.figure('rho')
plt.xlabel('$x$')
plt.legend()
save_tikz(f'fokker-nu={nu}-rho.tikz')

# en = np.zeros_like(t_arr)
# for i, Phi in enumerate(Phi_arr):
#     en[i] = solver.entropy(Phi)
# plt.figure()
# plt.plot(t_arr, en - Einf)
# plt.yscale('log')
# plt.xlabel('$t$')
# plt.ylabel(r'$E(\rho)-E(\rho_\infty)$')
# plt.legend()
# tikzplotlib.clean_figure()
# tikzplotlib.save(f'fokker-nu={nu}-ent.tikz')

plt.show()
