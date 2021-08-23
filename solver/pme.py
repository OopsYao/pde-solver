from .core import Solver, rho2Phi
import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad
from tqdm import tqdm
import matplotlib.pyplot as plt
from .utils import save_tikz

plt.style.use("seaborn-deep")

# Parameters
m = 2
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


Phi_set = np.empty((2001, N))
solver = Solver(rho0, N, U, U_p, U_pp, V, V_p, V_pp, W, W_p, W_pp)
t_arr = dt * np.arange(t0 / dt, 2101)
x = np.linspace(-1, 1, N)
en = np.zeros_like(t_arr)
plt.plot(x, rho0(x), '--', label=r'初始密度$\rho_0$')

try:
    Phi_set = np.load(f'pme-m={m}.npy')
except:
    for i, (t, Phi) in enumerate(zip(tqdm(t_arr), solver.step(-s0, s0, dt))):
        Phi_set[i] = Phi
    np.save(f'pme-m={m}.npy', Phi_set)
for i, Phi in enumerate(Phi_set):
    en[i] = solver.entropy(Phi)
plt.plot(x, bpp(x, t_arr[-1]), label=rf'$t={t_arr[-1]}$时的真实解')
plt.plot(Phi_set[1000], solver.recover(Phi_set[1000]),
         '-.', label=rf'$t={t_arr[1000]:.3f}$')

idx = np.arange(5, N - 5, 10)
idx = [*np.arange(5), *idx, *np.arange(N - 5, N)]
plt.plot(Phi_set[2000][idx], solver.recover(Phi_set[2000])
         [idx], 'x-', label=rf'$t={t_arr[2000]:.3f}$')
# plt.plot(Phi_set[2000], solver.recover(Phi_set[2000]), 'x-', label=rf'$t={t_arr[2000]:.3f}$')
plt.xlabel(r'$x$')
plt.legend()
save_tikz(f'pme-m={m}-rho.tikz')


plt.figure()
plt.plot(np.linspace(0, 2, N), Phi_set[0], '-.', label='$t=0$')
plt.plot(np.linspace(0, 2, N), Phi_set[1000], '--', label='$t=0.01$')
plt.plot(np.linspace(0, 2, N), Phi_set[-1], label='$t=0.021$')
plt.legend()
plt.xlabel(r'$\eta$')
save_tikz(f'pme-m={m}-Phi.tikz')

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(t_arr, en, label="熵的变化曲线")
plt.plot(t_arr, t_arr ** (- alpha * (m - 1)), '--', label="预期衰退速率")
plt.xlabel('$t$')
plt.ylabel('$E$')
plt.legend()
save_tikz(f'pme-m={m}-ent.tikz')
# plt.show()

# for N in [61, 126, 251, 501]:
for dt in [8e-4, 4e-5, 2e-5, 1e-5, 0.5e-5]:
    # print(len(t_arr))
    def rho(x): return bpp(x, 0.021)
    Phi_ref = rho2Phi(rho, N, -s0, s0)
    M = quad(rho, -s0, s0)[0]
    solver = Solver(rho0, N, U, U_p, U_pp, V, V_p, V_pp, W, W_p, W_pp)
    Phi_end = None
    # t_arr = dt * np.arange(t0 / dt, 2101)
    t_arr = np.linspace(t0, 0.021, int((0.022 - t0) / dt))
    for i, (t, Phi) in enumerate(zip(tqdm(t_arr), solver.step(-s0, s0, dt))):
        if abs(t + dt - 0.021) > abs(t - 0.021):
            # if i == 2000:
            Phi_end = Phi
            print(i)
            break
    err = ((np.abs(Phi_end - Phi_ref)).sum() * M / N)
    # plt.plot(np.linspace(0, M, N), Phi_ref, label='ref')
    # plt.plot(np.linspace(0, M, N), Phi_end, '--', label='end')
    plt.plot(Phi_end, solver.recover(Phi_end))
    plt.plot(x, rho(x), '--')
    # plt.show()
    print(err)


plt.legend()
plt.show()
