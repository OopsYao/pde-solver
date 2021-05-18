import numpy as np
from .core import Solver
from tqdm import tqdm
import matplotlib.pyplot as plt
from .utils import save_tikz


sigma = 1
chi = 1
N = 501
dt = 1e-1

THEME = 'rho0-blowup'


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


def plot_rho(Phi, *args, **kwargs):
    plt.figure('rho')
    plt.plot(Phi, solver.recover(Phi), *args, **kwargs)


def plot_Phi(Phi, *args, **kwargs):
    plt.figure('Phi')
    plt.plot(tilde_Omega, Phi, *args, **kwargs)


def L(Phi1, Phi2, p):
    diff = np.abs(Phi1 - Phi2)
    if p == np.inf:
        return diff.max(-1)
    else:
        return (diff ** p).sum(-1) ** (1 / p)


if THEME == 'rho0-decay':
    M0 = 2 * np.pi - 0.15
    tilde_Omega = np.linspace(0, M0, N)

    def rho0(x): return M0 / np.sqrt(2 * np.pi * sigma ** 2) * \
        np.exp(- x ** 2 / 2 / sigma ** 2)
    N = 501
    solver = Solver(rho0, N, U, U_p, U_pp, V, V_p, V_pp, W, W_p, W_pp)
    x = np.linspace(-6, 6, N)

    dt = np.array([
        * 1e-1 * np.ones(101),
        * 1e0 * np.ones(60),
        * 1e1 * np.ones(100),
        * 1e2 * np.ones(100),
        * 1e3 * np.ones(100),
        * 1e4 * np.ones(100),
        * 1e5 * np.ones(100),
    ])
    t_arr = dt.cumsum()
    Phi_arr = np.empty((len(t_arr), N))
    try:
        Phi_arr = np.load(f'ks-{THEME}.npy')
    except:
        for i, (t, Phi) in enumerate(zip(tqdm(t_arr), solver.step(-6, 6, dt))):
            Phi_arr[i] = Phi
        np.save(f'ks-{THEME}.npy', Phi_arr)

    plt.figure('rho')
    plt.plot(x, rho0(x), '--', label=r'初始密度$\rho_0$')
    plot_Phi(Phi_arr[0], '--', label=f'$t=0$')

    for ii, i in enumerate([10, 250, 310]):
        label = f'$t={t_arr[i]:.0f}$'
        s = ['-.', ':', '-']
        plot_Phi(Phi_arr[i], s[ii], label=label)
        plot_rho(Phi_arr[i], s[ii], label=label)

    # Post plot
    plt.figure('Phi')
    plt.xlabel(r'$\eta$')
    plt.legend()
    save_tikz(f'ks-{THEME}-Phi.tikz')

    plt.figure('rho')
    plt.xlabel('$x$')
    plt.xlim((-6, 6))
    plt.legend()
    save_tikz(f'ks-{THEME}-rho.tikz')

    plt.figure('norm')
    plt.ylabel(r'$||\rho||$')
    plt.xlabel('$t$')
    plt.xscale('log')
    plt.yscale('log')
    rho_arr = np.empty_like(Phi_arr)
    for i, Phi in enumerate(Phi_arr):
        rho_arr[i] = solver.recover(Phi)
    plt.plot(t_arr, rho_arr.max(-1))
    save_tikz(f'ks-{THEME}-norm.tikz')

    Phi_inf = Phi_arr[-1]
    norm1_Phi = L(Phi_arr, Phi_inf, 1)
    norm2_Phi = L(Phi_arr, Phi_inf, 2)
    normI_Phi = L(Phi_arr, Phi_inf, np.inf)
    plt.figure('norm Phi')
    plt.plot(t_arr, norm1_Phi, label='$L_1$')
    plt.plot(t_arr, 5 * norm2_Phi, label='$5L_2$')
    plt.plot(t_arr, 10 * normI_Phi, label=r'$10L_\infty$')
    plt.xscale('log')
    plt.xlabel('$t$')
    plt.ylabel('$p$-范数')
    plt.legend()
    save_tikz(f'ks-{THEME}-norm_Phi.tikz')


if THEME == 'rho0-blowup':
    M0 = 2 * np.pi + 0.15
    tilde_Omega = np.linspace(0, M0, N)

    def rho0(x): return M0 / np.sqrt(2 * np.pi * sigma ** 2) * \
        np.exp(- x ** 2 / 2 / sigma ** 2)
    solver = Solver(rho0, N, U, U_p, U_pp, V, V_p, V_pp, W, W_p, W_pp)
    x = np.linspace(-6, 6, N)

    dt = np.array([
        * 1e-1 * np.ones(118),
        * 1e-3 * np.ones(54),
        * 1e-6 * np.ones(1200),
        * 1e-7 * np.ones(22),
        * 1e-8 * np.ones(12),
        * 1e-9 * np.ones(17),
        * 1e-10 * np.ones(43),
        * 1e-11 * np.ones(20),
        * 1e-12 * np.ones(70),
        * 1e-13 * np.ones(40),
    ])
    t_arr = dt.cumsum()

    Phi_arr = np.empty((len(t_arr), N))
    try:
        Phi_arr = np.load(f'ks-rho0-blowup.npy')
    except:
        for i, (t, Phi) in enumerate(zip(tqdm(t_arr), solver.step(-6, 6, dt))):
            Phi_arr[i] = Phi
        np.save(f'ks-rho0-blowup.npy', Phi_arr)

    plt.figure('rho')
    plt.plot(x, rho0(x), '--', label=r'初始密度$\rho_0$')
    plot_Phi(Phi_arr[0], '--', label=f'$t=0$')

    for ii, i in enumerate([10, 20, 40]):
        label = f'$t={t_arr[i]:.0f}$'
        s = ['-.', ':', '-']
        plot_Phi(Phi_arr[i], s[ii], label=label)
        plot_rho(Phi_arr[i], s[ii], label=label)

    # Post plot
    plt.figure('Phi')
    plt.xlabel(r'$\eta$')
    plt.legend()
    save_tikz(f'ks-rho0-blowup-Phi.tikz')

    plt.figure('rho')
    plt.xlabel('$x$')
    plt.legend()
    save_tikz(f'ks-rho0-blowup-rho.tikz')

    plt.figure('norm')
    plt.ylabel(r'$||\rho||$')
    plt.xlabel('$t$')
    plt.yscale('log')
    rho_arr = np.empty_like(Phi_arr)
    for i, Phi in enumerate(Phi_arr):
        rho_arr[i] = solver.recover(Phi)
    plt.plot(t_arr, rho_arr.max(-1))
    save_tikz(f'ks-rho0-blowup-norm.tikz')

    Phi_inf = Phi_arr[-1]
    norm1_Phi = L(Phi_arr, Phi_inf, 1)
    norm2_Phi = L(Phi_arr, Phi_inf, 2)
    normI_Phi = L(Phi_arr, Phi_inf, np.inf)
    plt.figure('norm Phi')
    plt.plot(t_arr, norm1_Phi, label=r'$L_1$')
    plt.plot(t_arr, 10 * norm2_Phi, label=r'$10L_2$')
    plt.plot(t_arr, 10 * normI_Phi, label=r'$10L_\infty$')
    plt.xlabel('$t$')
    plt.ylabel('$p$-范数')
    plt.legend()
    save_tikz(f'ks-{THEME}-norm_Phi.tikz')

if THEME == 'rho01':
    M0 = 4 * np.pi - 0.6
    N = 501
    tilde_Omega = np.linspace(0, M0, N)
    solver = Solver(rho01, N, U, U_p, U_pp, V, V_p, V_pp, W, W_p, W_pp)
    x = np.linspace(-5, 5, N)
    dt = np.array([
        * 1e-3 * np.ones(1801),
        * 1e-4 * np.ones(400),
        * 1e-5 * np.ones(190),
    ])
    t_arr = dt.cumsum()
    Phi_arr = np.empty((len(t_arr), N))
    try:
        Phi_arr = np.load(f'ks-{THEME}.npy')
    except:
        for i, (t, Phi) in enumerate(zip(tqdm(t_arr), solver.step(-5, 5, dt))):
            Phi_arr[i] = Phi
        np.save(f'ks-{THEME}.npy', Phi_arr)

    tilde_Omega = np.linspace(0, 4 * np.pi - 0.6, N)
    plt.figure('rho')
    plt.plot(x, rho01(x), '--', label=r'初始密度$\rho_0$')
    plot_Phi(Phi_arr[0], '--', label='$t=0$')

    for ii, i in enumerate([200, 800, 1800]):
        label = f'$t={t_arr[i]:.1f}$'
        s = ['-.', ':', '-', '-']
        plot_Phi(Phi_arr[i], s[ii], label=label)
        plot_rho(Phi_arr[i], s[ii], label=label)
    plt.figure('norm')
    rho_arr = np.empty_like(Phi_arr)
    for i, Phi in enumerate(Phi_arr):
        rho_arr[i] = solver.recover(Phi)
    plt.plot(t_arr, rho_arr.max(-1))
    plt.xlabel('$t$')
    plt.ylabel(r'$||\rho||$')
    plt.yscale('log')
    save_tikz(f'ks-{THEME}-norm.tikz')

    plt.figure('Phi')
    plt.xlabel(r'$\tilde x$')
    plt.legend()
    save_tikz(f'ks-{THEME}-Phi.tikz')
    plt.figure('rho')
    plt.xlabel('$x$')
    plt.legend()
    save_tikz(f'ks-{THEME}-rho.tikz')

    Phi_inf = Phi_arr[-1]
    norm1_Phi = L(Phi_arr, Phi_inf, 1)
    norm2_Phi = L(Phi_arr, Phi_inf, 2)
    normI_Phi = L(Phi_arr, Phi_inf, np.inf)
    plt.figure('norm Phi')
    plt.plot(t_arr, norm1_Phi, label=r'$L_1$')
    plt.plot(t_arr, 10 * norm2_Phi, label=r'$10L_2$')
    plt.plot(t_arr, 100 * normI_Phi, label=r'$100L_\infty$')
    plt.legend()
    save_tikz(f'ks-{THEME}-norm_Phi.tikz')


if THEME == 'rho02':
    M0 = 4 * np.pi - 0.4
    N = 301
    solver = Solver(rho02, N, U, U_p, U_pp, V, V_p, V_pp, W, W_p, W_pp)
    x = np.linspace(-5, 5, N)
    dt = np.array([
        * 1e-4 * np.ones(741),
        * 1e-5 * np.ones(140),
        * 1e-6 * np.ones(230),
    ])
    t_arr = dt.cumsum()
    tilde_Omega = np.linspace(0, 4 * np.pi - 0.4, N)
    Phi_arr = np.empty((len(t_arr), N))
    try:
        Phi_arr = np.load(f'ks-{THEME}.npy')
    except:
        for i, (t, Phi) in enumerate(zip(tqdm(t_arr), solver.step(-5, 5, dt))):
            Phi_arr[i] = Phi
        np.save(f'ks-{THEME}.npy', Phi_arr)

    plt.figure('Phi')
    plt.figure('rho')

    plt.figure('rho')
    plt.plot(x, rho02(x), '--', label=r'初始密度$\rho_0$')
    plot_Phi(Phi_arr[0], '--', label='$t=0$')
    tilde_Omega = np.linspace(0, 4 * np.pi - 0.4, N)
    for ii, i in enumerate([100, 200, 300]):
        label = f'$t={t_arr[i]:.2f}$'
        s = ['-.', ':', '-', '-', '-']
        plot_Phi(Phi_arr[i], s[ii], label=label)
        plot_rho(Phi_arr[i], s[ii], label=label)

    plt.figure('Phi')
    plt.legend()
    plt.xlabel(r'$\eta$')
    plt.savefig(f'ks-{THEME}-Phi.pdf')

    plt.figure('rho')
    plt.legend()
    plt.xlabel(r'$x$')
    save_tikz(f'ks-{THEME}-rho.tikz')

    plt.figure('norm')
    rho_arr = np.empty_like(Phi_arr)
    for i, Phi in enumerate(Phi_arr):
        rho_arr[i] = solver.recover(Phi)
    plt.plot(t_arr, rho_arr.max(-1))
    plt.xlabel('$t$')
    plt.yscale('log')
    plt.ylabel(r'$||\rho||$')
    save_tikz(f'ks-{THEME}-norm.tikz')

    plt.figure('norm Phi')
    Phi_inf = Phi_arr[-1]
    norm1_Phi = L(Phi_arr, Phi_inf, 1)
    norm2_Phi = L(Phi_arr, Phi_inf, 2)
    normI_Phi = L(Phi_arr, Phi_inf, np.inf)
    plt.plot(t_arr, norm1_Phi, label='$L_1$')
    plt.plot(t_arr, 10 * norm2_Phi, label='$10L_2$')
    plt.plot(t_arr, 10 * normI_Phi, label=r'$10L_\infty$')
    plt.ylabel('$p$-范数')
    plt.xlabel('$t$')
    plt.legend()
    save_tikz(f'ks-{THEME}-norm_Phi.tikz')


plt.show()
