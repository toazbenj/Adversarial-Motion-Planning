import unittest
import numpy as np

from potential_function import potential_function


class TestPotentialFunction(unittest.TestCase):

    def test_case_1(self):
        A = np.array([[3, 6, 1],
                      [8, -1, 5],
                      [10, -2, 10]])

        B = np.array([[-2, 5, -1],
                      [10, 1, 5],
                      [10, 2, 8]])

        phi, is_exact = potential_function(A, B)
        print("Test Case 1: Potential Function:\n", phi)
        print("Is Exact Potential Function:", is_exact)
        self.assertFalse(is_exact)  # Assuming this case does not satisfy exact potential condition

    def test_case_2(self):
        A = np.array([[2, 30],
                      [0, 8]])

        B = np.array([[2, 0],
                      [30, 8]])

        phi, is_exact = potential_function(A, B)
        print("Test Case 2: Potential Function:\n", phi)
        print("Is Exact Potential Function:", is_exact)
        self.assertTrue(is_exact)  # Assuming this case also does not satisfy exact potential condition

    def test_case_3(self):
        A = np.array([[4, 5],
                      [1, 3]])

        B = np.array([[1, 2],
                      [4, 5]])

        phi, is_exact = potential_function(A, B)
        print("Test Case 3: Potential Function:\n", phi)
        print("Is Exact Potential Function:", is_exact)
        self.assertFalse(is_exact)  # Assuming this case satisfies exact potential condition

    def test_case_4(self):
        A = np.array([[2.45, 3.95],
                      [2.55, 4.05]])

        B = np.array([[1, 2],
                      [4, 5]])

        phi, is_exact = potential_function(A, B)
        print("Test Case 4: Potential Function:\n", phi)
        print("Is Exact Potential Function:", is_exact)
        self.assertTrue(is_exact)  # Assuming this case satisfies exact potential condition

    def test_case_5(self):
        A = np.array([[0, 0, 0, 0],
                      [2, 10, 2, 2],
                      [1, 1, 1, 1],
                      [3, 10, 3, 10]])

        B = np.array([[1, 3, 0, 2],
                      [1, 10, 0, 2],
                      [1, 3, 0, 2],
                      [1, 10, 0, 10]])

        phi, is_exact = potential_function(A, B)
        print("Test Case 5: Potential Function:\n", phi)
        print("Is Exact Potential Function:", is_exact)
        self.assertFalse(is_exact)  # Assuming this case does not satisfy exact potential condition

    def test_case_6(self):
        A = np.array([[-2, 1],
                      [0, -1]])

        B = np.array([[-3, 1],
                      [2, -2]])

        phi, is_exact = potential_function(A, B)
        print("Test Case 6: Potential Function:\n", phi)
        print("Is Exact Potential Function:", is_exact)
        self.assertFalse(is_exact)  # Assuming this case does not satisfy exact potential condition

    def test_case_7(self):
        A = np.array([[2, 1, 3],
                      [2, 4, 3],
                      [5, 4, 6]])

        B = np.array([[6, 5, 4],
                      [3, 4, 2],
                      [2, 1, 3]])

        phi, is_exact = potential_function(A, B)
        print("Test Case 7: Potential Function:\n", phi)
        print("Is Exact Potential Function:", is_exact)
        self.assertFalse(is_exact)  # Assuming this case does not satisfy exact potential condition

    def test_case_8(self):
        A = np.array([[3, 1],
                      [2, 0]])

        B = np.array([[2, 5],
                      [0, 1]])

        phi, is_exact = potential_function(A, B)
        print("Test Case 8: Potential Function:\n", phi)
        print("Is Exact Potential Function:", is_exact)
        self.assertFalse(is_exact)  # Assuming this case does not satisfy exact potential condition


if __name__ == '__main__':
    unittest.main()
