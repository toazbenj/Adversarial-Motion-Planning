"""
Cost_to_go.py
Created by Ben Toaz on 6-7-24

Unit tests for nash_equilibrium.py
"""

import unittest
import numpy as np
from nash_equilibrium import scipy_solve, itr_solve

class TestScipySolve(unittest.TestCase):

    def test_square_matrix_2x2(self):
        payoff_matrix = np.array([[3, 1], [0, 2]])
        player_A_strategy_scipy, player_B_strategy_scipy, game_value_scipy = scipy_solve(payoff_matrix)
        player_A_strategy_itr, player_B_strategy_itr, game_value_itr = itr_solve(payoff_matrix)

        np.testing.assert_almost_equal(game_value_scipy, game_value_itr, 1)
        np.testing.assert_almost_equal(player_A_strategy_scipy, player_A_strategy_itr, 1)
        np.testing.assert_almost_equal(player_B_strategy_scipy, player_B_strategy_itr, 1)

    def test_rectangular_matrix_3x2(self):
        # sus
        payoff_matrix = np.array([[3, 1], [0, 2], [2, 4]])
        player_A_strategy_scipy, player_B_strategy_scipy, game_value_scipy = scipy_solve(payoff_matrix)
        player_A_strategy_itr, player_B_strategy_itr, game_value_itr = itr_solve(payoff_matrix)

        np.testing.assert_almost_equal(game_value_scipy, game_value_itr, 1)
        np.testing.assert_almost_equal(player_A_strategy_scipy, player_A_strategy_itr, 1)
        np.testing.assert_almost_equal(player_B_strategy_scipy, player_B_strategy_itr, 1)

    def test_rectangular_matrix_2x3(self):
        # sus
        payoff_matrix = np.array([[3, 1, 2], [0, 2, 4]])
        player_A_strategy_scipy, player_B_strategy_scipy, game_value_scipy = scipy_solve(payoff_matrix)
        player_A_strategy_itr, player_B_strategy_itr, game_value_itr = itr_solve(payoff_matrix)

        np.testing.assert_almost_equal(game_value_scipy, game_value_itr, 1)
        np.testing.assert_almost_equal(player_A_strategy_scipy, player_A_strategy_itr, 1)
        np.testing.assert_almost_equal(player_B_strategy_scipy, player_B_strategy_itr, 1)
    def test_square_matrix_3x3(self):
        # sus
        payoff_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        player_A_strategy_scipy, player_B_strategy_scipy, game_value_scipy = scipy_solve(payoff_matrix)
        player_A_strategy_itr, player_B_strategy_itr, game_value_itr = itr_solve(payoff_matrix)

        np.testing.assert_almost_equal(game_value_scipy, game_value_itr, 1)
        np.testing.assert_almost_equal(player_A_strategy_scipy, player_A_strategy_itr, 1)
        np.testing.assert_almost_equal(player_B_strategy_scipy, player_B_strategy_itr, 1)

    def test_single_strategy_matrix_1x2(self):
        payoff_matrix = np.array([[1, 2]])
        player_A_strategy_scipy, player_B_strategy_scipy, game_value_scipy = scipy_solve(payoff_matrix)
        player_A_strategy_itr, player_B_strategy_itr, game_value_itr = itr_solve(payoff_matrix)

        np.testing.assert_almost_equal(game_value_scipy, game_value_itr, 1)
        np.testing.assert_almost_equal(player_A_strategy_scipy, player_A_strategy_itr, 1)
        np.testing.assert_almost_equal(player_B_strategy_scipy, player_B_strategy_itr, 1)


if __name__ == '__main__':
    unittest.main()
