import numpy as np
from .core import Solver
from tqdm import tqdm
import matplotlib.pyplot as plt

M0 = 2 * np.pi - 0.15
sigma = 1
chi = 1
N = 501
dt = 1e-1


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


solver = Solver(rho0, N, U, U_p, U_pp, V, V_p, V_pp, W, W_p, W_pp)
x = np.linspace(-6, 6, N)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(0 * np.ones_like(x), x, rho0(x), label='t=0')

# dt = np.array([
#     * 1e-1 * np.ones(118),
#     * 1e-3 * np.ones(54),
#     * 1e-6 * np.ones(1200),
#     * 1e-7 * np.ones(22),
#     * 1e-8 * np.ones(12),
#     * 1e-9 * np.ones(17),
#     * 1e-10 * np.ones(43),
#     * 1e-11 * np.ones(20),
#     * 1e-12 * np.ones(70),
# ])
dt = np.array([
    * 1e-1 * np.ones(100),
    * 1e0 * np.ones(60),
    * 1e1 * np.ones(54),
    * 1e2 * np.ones(54),
])
t_arr = dt.cumsum()

for i, (t, Phi) in enumerate(zip(tqdm(t_arr), solver.step(-6, 6, dt))):
    if i in [len(t_arr) - 46 , len(t_arr) - 26, len(t_arr) - 1]:
    # if i in [len(t_arr) - 45, len(t_arr) - 30, len(t_arr) - 15, len(t_arr) - 1]:
        plt.plot(t * np.ones_like(Phi), Phi,
                 # solver.recover(Phi), label=rf't=t*+{70 - 1 - len(t_arr) + i}e-12')
                 solver.recover(Phi), label=rf't={t:.0f}')
        # plt.plot(Phi, solver.recover(Phi), 'x-', label='Computed solution')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel(r'$\rho$')
plt.legend()

# plt.plot(x, rho02(x), label='Initial value')
# plt.xlabel('x')
# plt.ylabel(r'$\rho$')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel(r'$\rho$')
plt.legend()
plt.show()
