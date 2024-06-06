import unittest
from cost_to_go import collision_check, cost, cost_to_go, control_inputs
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_collision_check(self):
        # stationary
        state = np.array([[0, 0],
                          [0, 1],
                          [0, 0]])
        control_input = np.array([[0, 0],
                                  [0, 0]])

        self.assertEqual(collision_check(state, control_input), False)

        # same position
        state = np.array([[0, 0],
                          [0, 0],
                          [0, 0]])
        control_input = np.array([[0, 0],
                                  [0, 0]])
        self.assertEqual(collision_check(state, control_input), True)

        # same maneuver, lined up
        state = np.array([[0, 0],
                          [0, 1],
                          [0, 0]])
        control_input = np.array([[1, -1],
                                  [1, 1]])

        self.assertEqual(collision_check(state, control_input), True)

        # same maneuver, one in the lead
        state = np.array([[0, 1],
                          [0, 1],
                          [0, 0]])
        control_input = np.array([[1, -1],
                                  [1, 1]])

        self.assertEqual(collision_check(state, control_input), False)

    def test_cost(self):
        # no maneuvers, adjacent
        state = np.array([[0, 0],
                          [0, 1],
                          [0, 0]])
        control_input = np.array([[0, 0],
                                  [0, 0]])
        penalty_lst = [0, 1, 2, 1, 10]
        test_mat = cost(state, control_input, penalty_lst)
        np.testing.assert_array_equal(test_mat, np.array([[0, 0],
                                                          [1, 1]]))
        # collision
        control_input = np.array([[1, -1],
                                  [1, 1]])
        test_mat = cost(state, control_input, penalty_lst)
        np.testing.assert_array_equal(test_mat, np.array([[10, 10],
                                                          [10, 10]]))
        # p1 passes
        control_input = np.array([[1, 0],
                                  [1, 0]])
        np.testing.assert_array_equal(cost(state, control_input, penalty_lst), np.array([[3, 0],
                                                                                         [0, 1]]))
        # p2 passes
        control_input = np.array([[0, -1],
                                  [0, 1]])
        np.testing.assert_array_equal(cost(state, control_input, penalty_lst), np.array([[0, 3],
                                                                                         [1, 0]]))
        # both move forward
        control_input = np.array([[0, 0],
                                  [1, 1]])
        np.testing.assert_array_equal(cost(state, control_input, penalty_lst), np.array([[2, 2],
                                                                                         [1, 1]]))

    def test_control_inputs(self):
        state = np.array([[0, 0],
                          [0, 1],
                          [0, 0]])
        acceleration_maneuver_range = range(-1, 2)
        inputs = [np.array([[-1, -1]]), np.array([[-1, 0]]), np.array([[-1, 1]]), np.array([[0, -1]]),
                  np.array([[0, 0]]), np.array([[0, 1]]), np.array([[1, -1]]), np.array([[1, 0]]), np.array([[1, 1]])]
        np.testing.assert_array_equal(control_inputs(state, acceleration_maneuver_range), inputs)

if __name__ == '__main__':
    unittest.main()
