import numpy as np
from tqdm import tqdm
from scipy import integrate


class Solver:
    def __init__(self, rho0, N, U, U_p, U_pp, V, V_p, V_pp, W, W_p, W_pp):
        self.rho0 = rho0
        M = integrate.quad(rho0, -np.inf, np.inf)[0]
        self.dx = M / N
        self.M = M
        self.N = N

        self.U = U
        self.U_p = U_p
        self.U_pp = U_pp
        self.V = V
        self.V_p = V_p
        self.V_pp = V_pp
        self.W = W
        self.W_p = W_p
        self.W_pp = W_pp

    def recover(self, Phi):
        """ recover the density rho (and its underlying x, here it is Phi) """
        return np.array([0, *(2 * self.dx / (Phi[2:] - Phi[:-2])), 0])

    def entropy(self, Phi):
        rho = self.recover(Phi)
        E = np.trapz(self.U(rho), Phi)
        E += np.trapz(self.V(Phi) * rho, Phi)

        mask = ~np.eye(self.N, dtype=bool)
        Phi_ij = np.expand_dims(Phi, -1) - Phi
        E += self.W(Phi_ij[mask]).sum() * self.dx ** 2 / 2
        return E

    def Psi(self, x): return x * self.U(1 / x)

    def Psi_p(self, x):
        inv = 1 / x
        return self.U(inv) - inv * self.U_p(inv)

    def Psi_pp(self, x):
        inv = 1 / x
        return inv ** 3 * self.U_pp(inv)

    def _init_diffeo(self, a, b):
        # Omega_tilde := [0, M] (equidistant)
        Omega_tilde = np.linspace(0, self.M, self.N)
        # Initial guess of Omega = Phi0(Omega_tilde)
        Omega = np.linspace(a, b, self.N)
        # Loop version
        # for i, x in enumerate(Omega_tilde):
        #     while True:
        #         cur = Omega[i]
        #         if a < cur < b:
        #             inc = - (integrate.quad(rho0, a, cur)[0] - x) / rho0(cur)
        #             Omega[i] += inc
        #             if abs(inc) < 10e-8:
        #                 break
        #         else:
        #             Omega[i] = np.random.uniform(a, b)

        @np.vectorize
        def Rho0(upper):
            return integrate.quad(self.rho0, a, upper)[0]

        unfulfill = np.full_like(Omega, True, dtype=bool)
        Omega[0], Omega[-1] = a, b
        unfulfill[0], unfulfill[-1] = False, False
        while unfulfill.any():
            # Correction
            invalid = (Omega < a) | (b < Omega)
            Omega[invalid] = np.random.uniform(a, b, invalid.sum())
            # Increment (where unfulfilled)
            inc = - (Rho0(Omega[unfulfill]) -
                     Omega_tilde[unfulfill]) / self.rho0(Omega[unfulfill])
            Omega[unfulfill] += inc
            unfulfill[unfulfill] = (np.abs(inc) >= 10e-8)
        return Omega

    def step(self, a, b, dt):
        Phi = self._init_diffeo(a, b)
        yield Phi

        # Every time step

        # Some utils
        PN = (np.diag(-np.ones(self.N)) +
              np.diag(np.ones(self.N - 1), 1))[:-1] / self.dx

        def Pn(y):
            return (y[1:] - y[:-1]) / self.dx
        while True:
            Phi_n = Phi.copy()  # Given Phi_n, find Phi_n+1
            # Boundary points of Phi (support boundary of rho)
            v = -(self.U_p(0) - self.U_p(2 * self.dx / (Phi[-1] - Phi[-3]))) / (Phi[-1] - Phi[-2]) - self.V_p(
                Phi[-1]) - np.trapz([0, *(self.W_p(Phi[-1] - Phi[1:-1]) / (Phi[2:] - Phi[:-2]) * 2 * self.dx), 0], Phi)
            right = Phi[-1] + v * dt
            v = -(self.U_p(0) - self.U_p(2 * self.dx / (Phi[2] - Phi[0]))) / (Phi[0] - Phi[1]) - self.V_p(
                Phi[0]) - np.trapz([0, *(self.W_p(Phi[0] - Phi[1:-1]) / (Phi[2:] - Phi[:-2]) * 2 * self.dx), 0], Phi)
            left = Phi[0] + v * dt
            # Interior points of Phi
            Phi[0], Phi[-1] = left, right
            # Newton's method loop, with initial guess Phi = Phi_n (only for interior points)
            # Be careful with the `+=` operator! It also can be slow sometimes
            while True:
                F = (Phi - Phi_n)[1:-1] / dt
                F_p = np.eye(self.N - 2) / dt

                Phi_p = Pn(Phi)  # Phi_p at 0.5, 1.5, 2.5, ...
                F = F - Pn(self.Psi_p(Phi_p))
                F_p = F_p - \
                    Pn(np.expand_dims(self.Psi_pp(Phi_p), -1) * PN[:, 1:-1])

                F = F + self.V_p(Phi[1:-1])
                F_p = F_p + np.diag(self.V_pp(Phi[1:-1]))

                # Phi_ij[i][j] = Phi[i] - Phi[j]
                Phi_ij = np.expand_dims(Phi[1:-1], -1) - Phi[1:-1]
                mask = ~np.eye(self.N - 2, dtype=bool)
                Wp = np.zeros_like(Phi_ij)
                Wpp = np.zeros_like(Phi_ij)
                Wp[mask] = self.W_p(Phi_ij[mask])
                Wpp[mask] = self.W_pp(Phi_ij[mask])
                F = F + Wp.sum(axis=1) * self.dx
                F_p = F_p + (np.diag(Wpp.sum(axis=1)) - Wpp) * self.dx

                inc = np.linalg.solve(F_p, -F)
                Phi = Phi + [0, *inc, 0]
                if np.abs(inc).max() < 1e-8:
                    break
            yield Phi
