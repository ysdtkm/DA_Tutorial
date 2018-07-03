
import unittest
import numpy as np
from module_obs_network import get_h, plot_mat, model_state_example, get_grid_val
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

