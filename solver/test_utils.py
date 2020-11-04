from .utils import derivative, integral
import numpy as np
import numpy.testing as npt
import unittest


x = np.array([0, 1/6, 1/4, 1/3, 1/2, 3/4, 5/6, 1, 7/6]) * np.pi
Phi = np.sin(x)
f = Phi ** 2


class UtilsTest(unittest.TestCase):
    def test_outter_derivative(self):
        ff = np.vstack((f, f, f))
        num_grad = derivative(ff, Phi)
        real_grad = 2 * np.sin(x)
        real_grad = np.array([real_grad, real_grad, real_grad])
        npt.assert_almost_equal(num_grad, real_grad)

    def test_inner_derivative(self):
        # actually, if we calculate the derivative directly by `derivative(f, x)`,
        # the result has a large deviation. So instead, we use chainrule here.
        num_grad = derivative(f, Phi) * derivative(Phi, x)
        real_grad = np.sin(2 * x)
        npt.assert_almost_equal(num_grad, real_grad, 1)

    def test_integral(self):
        p = np.vstack((Phi, Phi, Phi))
        num_int = integral(p, x)
        def prim_func(x): return -np.cos(x)
        real_int = prim_func(7/6 * np.pi) - prim_func(0)
        real_int = np.array([real_int, real_int, real_int])
        npt.assert_almost_equal(num_int, real_int, 2)


if __name__ == "__main__":
    unittest.main()
