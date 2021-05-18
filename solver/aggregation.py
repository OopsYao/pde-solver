import numpy as np
from .core import Solver
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import quad, dblquad
from .utils import save_tikz

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


tilde_Omega = np.linspace(0, M, N)


def plot_Phi(Phi, *args, **kwargs):
    plt.figure('Phi')
    plt.plot(tilde_Omega, Phi, *args, **kwargs)


def plot_rho(Phi, *args, **kwargs):
    plt.figure('rho')
    plt.plot(Phi, solver.recover(Phi), *args, **kwargs)


solver = Solver(rho0, N, U, U_p, U_pp, V, V_p, V_pp, W, W_p, W_pp)
t_arr = dt * np.arange(4892)
Phi_arr = np.empty((len(t_arr), N))
en = np.zeros_like(t_arr)
x = np.linspace(-3, 3, N)
try:
    Phi_arr = np.load(f'aggregation-b={b}.npy')
except:
    for i, (t, Phi) in enumerate(zip(tqdm(t_arr), solver.step(-2, 2, dt))):
        Phi_arr[i] = Phi
    np.save(f'aggregation-b={b}.npy', Phi_arr)

plt.figure('rho')
plt.plot(x, rho0(x), '--', label=r'初始密度$\rho_0$')
plt.figure('Phi')
plt.plot(tilde_Omega, Phi_arr[0], '--', label='$t=0$')
for i in [800]:
    t = rf'$t={t_arr[i]:.3f}$'
    plot_rho(Phi_arr[i], '-.', label=t)
    plot_Phi(Phi_arr[i], '-.', label=t)
plt.figure('rho')
idx = np.arange(3, N - 3, 20)
idx = [*np.arange(3), *idx, *np.arange(N - 3, N)]
# plt.plot([Phi_arr[-1, 0], *(Phi_arr[-1, 2:-2:10]), Phi_arr[-1, -1]],
#          [0, *solver.recover(Phi_arr[-1])[2:-2:10], 0],
#          'x-', label=rf'$t={t_arr[2412]:.3f}$')
plt.plot(Phi_arr[-1][idx], solver.recover(Phi_arr[-1])[idx], 'x-', label='$t=4.890$')
plt.plot(x, rho_inf(x), label=r'稳态解')
plt.figure('Phi')
plt.plot(tilde_Omega, Phi_arr[-1], label=rf'$t={t_arr[-1]:.3f}$')

# en[i] = solver.entropy(Phi)
# xy = np.expand_dims(x, -1) - x
# mask = ~np.eye(N, dtype=bool)
# Wxy = np.zeros_like(xy)
# Wxy[mask] = W(xy[mask])
# rho_x = rho_inf(x)
# E_inf = (Wxy * np.expand_dims(rho_x, -1) * rho_x).sum() * (12 / N) ** 2 / 2
# plt.figure('ent')
# plt.yscale('log')
# plt.xlabel('t')
# plt.ylabel(r'$E(\rho)$')
# plt.plot(t_arr, en, label='Entropy')
# plt.plot(t_arr, np.exp(-7.5 * t_arr), label='Expected decay')
# plt.legend()

plt.figure('rho')
plt.xlabel(r'$x$')
plt.legend()
save_tikz(f'aggregation-b={b}-rho.tikz')

plt.figure('Phi')
plt.xlabel(r'$\tilde x$')
plt.legend()
save_tikz(f'aggregation-b={b}-Phi.tikz')

plt.show()
