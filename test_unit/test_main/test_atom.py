"""
    File that contains the unit test for setting up the atom.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General
import numpy as np
import random as rd
import unittest

# User defined.
import code.main.atom as atom

# ##############################################################################
# Global Constants
# ##############################################################################

ITERATIONS = 100

# ##############################################################################
# Classes
# ##############################################################################


class TestAtom(unittest.TestCase):

    def test_creation(self):
        """
            Tests the creation of the atom and the basic quantities.
        """

        for _ in range(ITERATIONS):
            # Choose a random mass and radius
            mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
            coordinate = np.array(
                [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
            )

            # Create an atom.
            matom = atom.Atom(radius, mass, coordinate)

            # Check that the atom is properly created.
            


# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
