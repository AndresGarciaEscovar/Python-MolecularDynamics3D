"""
    File that contains the tests for the math utilities.
"""

 ##############################################################################
# Imports
# ##############################################################################

# General
import unittest
from numpy import array

# User defined.
import code.utilities.utilities_math as umath

# ##############################################################################
# Global Constants
# ##############################################################################

ITERATIONS = 100

# ##############################################################################
# Classes
# ##############################################################################


class TestAtom(unittest.TestCase):

    def test_get_skew_symmetric_matrix(self):
        """
            Tests that the get_skew_symmetric_matrix method is working properly.
        """

        # Loop through the different types of arrays.
        for mtype in (list, tuple):
            tarray = mtype([1, 2, 3])
            with self.assertRaises(TypeError):
                umath.get_skew_symmetric_matrix(tarray)

            tarray = array(tarray, dtype=float)
            rmatrix = umath.get_skew_symmetric_matrix(tarray)

            compmatrix = array(
                [
                    [0.0, -tarray[2], tarray[1]],
                    [tarray[2], 0.0, -tarray[0]],
                    [-tarray[1], tarray[0], 0.0]
                ], dtype=float
            )

            # Matrices must be equal, element by element.
            for i, row in enumerate(compmatrix):
                for j, column in enumerate(row):
                    self.assertEqual(column, rmatrix[i, j])

            tarray = array([1, 2, 3, 4], dtype=float)
            with self.assertRaises(ValueError):
                umath.get_skew_symmetric_matrix(tarray)

    def test_intersect_spheres(self):
        """
            Tests that the function intersect_spheres
        """

# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
