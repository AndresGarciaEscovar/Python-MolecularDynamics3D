"""
    File that contains the tests for the molecule utilities.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General
import unittest
from numpy import array, pi

# User defined.
import code.utilities.utilities_math as umath

# ##############################################################################
# Classes
# ##############################################################################


class TestUtilitiesMolecule(unittest.TestCase):

    def test_symmetrize(self):
        """
            Tests that the function symmetrize is working properly.
        """
        # Define a non-square matrix.
        matrix = array([
            [1, 2, 3],
            [4, 5, 6]
        ], dtype=float)

        # Wrong dimensions.
        with self.assertRaises(ValueError):
            umath.symmetrize(matrix, passes=3)

        # Define square matrix.
        matrix = array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=int)

        with self.assertRaises(TypeError):
            umath.symmetrize(matrix)

        # Wrong type of matrix.
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        with self.assertRaises(TypeError):
            umath.symmetrize(matrix)

        # Define square matrix.
        matrix = array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=float)

        # Correct dimensions.
        rmatrix = umath.symmetrize(matrix, passes=3)

        # Must be the same dimensions.
        self.assertEqual(matrix.shape, rmatrix.shape)

        for i in range(len(matrix)):
            for j in range(len(matrix)):
                self.assertEqual(rmatrix[i, j], rmatrix[j, i])


# ##############################################################################
# Main Program
# ##############################################################################

if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
