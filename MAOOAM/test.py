
import unittest
import numpy as np
from class_da_system import da_system
from module_constants import read_xb_yo_xa, get_static_b
from module_obs_network import get_h, plot_mat, model_state_example, get_grid_val
from read_r_matrix import get_r
from exp_params import SEED

class TestLorenz(unittest.TestCase):
    @unittest.skip
    def test_lorenz(self):
        R = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]

        i = [0,2]
        j = i

        R = np.array(R)
        R2 = R[i,:]
        R2 = R2[:,j]

        print('R = ')
        print(R)
        print('R2 = ')
        print(R2)

class TestObsNetwork(unittest.TestCase):
    def test_difference_u_v(self):
        n = 1.5
        state = model_state_example()
        np.random.seed(SEED * 5)
        eps = 1.0e-8
        for is_atm in [True, False]:
            for i in range(1000):
                L = 5000000.0 / np.pi
                pivot_x = np.random.uniform(0, 2.0 * np.pi / n)
                pivot_y = np.random.uniform(0, np.pi)
                ptb_x = pivot_x + eps
                ptb_y = pivot_y + eps
                if ptb_x < 0.0 or ptb_x > 2.0 * np.pi / n or ptb_y < 0.0 or ptb_y > np.pi:
                    continue
                psi_pivot = get_grid_val(state, pivot_x, pivot_y, is_atm, "psi")
                psi_ptb_x = get_grid_val(state, ptb_x, pivot_y, is_atm, "psi")
                psi_ptb_y = get_grid_val(state, pivot_x, ptb_y, is_atm, "psi")
                u = get_grid_val(state, pivot_x, pivot_y, is_atm, "u")
                v = get_grid_val(state, pivot_x, pivot_y, is_atm, "v")
                self.assertTrue(np.isclose((psi_ptb_x - psi_pivot) / eps / L, v, atol=1e-6))
                self.assertTrue(np.isclose(- (psi_ptb_y - psi_pivot) / eps / L, u, atol=1e-6))

    def test_h_matrix_conditional_number(self):
        h = get_h()
        plot_mat(h)

    def test_get_grid_val(self):
        state = model_state_example()
        n = 1.5
        x = 1.2 * np.pi / n  # 0.0 <= x <= 2.0 * pi / n
        y = 0.8 * np.pi      # 0.0 <= y <= pi

        # get_grid_val() returns one of four variables
        # {atmosphere|ocean} x {streamfunction|temperature} at point (x, y)
        # unit: [m^2/s] for streamfunction and [K] for temperature
        a_psi = get_grid_val(state, x, y, True, "psi")
        a_tmp = get_grid_val(state, x, y, True, "tmp")
        o_psi = get_grid_val(state, x, y, False, "psi")
        o_tmp = get_grid_val(state, x, y, False, "tmp")
        self.assertTrue(np.isclose(-28390877.979826435, a_psi))
        self.assertTrue(np.isclose(-6.915992899895785 * 2.0, a_tmp))
        self.assertTrue(np.isclose(-16019.881464394632, o_psi))
        self.assertTrue(np.isclose(-39.234272164275836, o_tmp))

class TestTdvar(unittest.TestCase):
    @unittest.skip("constant files are old and inaccurate")
    def test_tdvar_with_cheng(self):
        n = p = 36
        xb, yo, xa_cheng = read_xb_yo_xa()
        R, B = self.read_cheng_r_b()
        B *= 10
        H = get_h()
        assert R.shape == (p, p)
        assert np.all(H == np.identity(n))
        das = da_system(x0=xb, yo=yo)
        das.setMethod("3DVar")
        das.setB(B)
        das.setR(R)
        das.setH(H)
        xa_das, KH = das.compute_analysis(xb, yo)
        xa_oi = self.oi(xb, yo, R, B, H)
        # print(xa_das, xa_oi, xa_cheng)
        di = xa_das - xa_cheng
        self.assertLess(np.max(np.abs(di)), 1.0e-5)

    @classmethod
    def oi(cls, xb, yo, R, B, H):
        K = B @ H.T @ np.linalg.inv(R + H @ B @ H.T)
        assert np.allclose(K, B @ np.linalg.inv(R + B))
        xa = xb + K @ (yo - H @ xb)
        return xa

    @classmethod
    def read_cheng_r_b(cls):
        dir = "/lustre/tyoshida/prgm/Cheng_MAOOAM"
        n = 36

        r = np.zeros((n, n))
        with open(f"{dir}/fort.202", "r") as f:
            for i in range(n):
                r[i, i] = float(f.readline().replace("D", "E").strip()) ** 2

        b = np.empty((n, n))
        with open(f"{dir}/fort.205", "r") as f:
            for i in range(n):
                bsli = f.readline().replace("D", "E").split()
                bfli = list(map(float, bsli))
                b[i, :] = np.array(bfli)
        assert np.allclose(b, b.T)
        return r, b

if __name__ == "__main__":
    unittest.main()
