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
import code.utilities.utilities_molecule

# ##############################################################################
# Classes
# ##############################################################################


class TestUtilitiesMolecule(unittest.TestCase):

    def test_get_bounding_radius(self):
        """
            Tests that the get_bounding_radius function is working properly.
        """
        self.assertTrue(False)

    def test_get_cod(self):
        """
            Tests that the get_cod function is working properly; where cod is
            center of diffusion.
        """
        self.assertTrue(False)

    def test_get_cog(self):
        """
            Tests that the get_cog function is working properly; where cog is
            center of geometry.
        """
        self.assertTrue(False)

    def test_get_com(self):
        """
            Tests that the get_com function is working properly; where cog is
            center of mass.
        """
        self.assertTrue(False)

    def test_get_dtensor(self):
        """
            Tests that the get_dtensor function is working properly.
        """
        self.assertTrue(False)

    def test_dtensor_and_orientation(self):
        """
            Tests that the dtensor_and_orientation function is working properly.
        """
        self.assertTrue(False)


# ##############################################################################
# Main Program
# ##############################################################################

if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
