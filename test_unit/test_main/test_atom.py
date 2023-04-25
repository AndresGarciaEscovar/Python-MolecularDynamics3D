"""
    File that contains the unit test for setting up the atom.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General
import mendeleev
import random as rd
import string
import unittest

from numpy import array, float64

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

    def test_change_aname(self):
        """
            Tests the creation of the atom and changing its name.
        """

        # Randomly choose a radius, mass and set of coordinates.
        mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
        coordinate = array(
            [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
        )

        # Create an atom.
        matom = atom.Atom(radius, mass, coordinate)

        # Try to change the name to an empty string.
        with self.assertRaises(ValueError):
            matom.aname = ""

        # Try to change the name to None.
        with self.assertRaises(TypeError):
            matom.aname = None

        # Try to change the name to anything other than a string.
        with self.assertRaises(TypeError):
            matom.aname = 6

        # Result must be a string.
        matom.aname = f"6"
        self.assertIsInstance(matom.aname, str)
        self.assertEqual(matom.aname, f"6")

    def test_change_atype(self):
        """
            Tests changing the type of the atom.
        """

        # Randomly choose a radius, mass and set of coordinates.
        mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
        coordinate = array(
            [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
        )

        # Create an atom.
        matom = atom.Atom(radius, mass, coordinate)

        # Try to change the name to an empty string.
        with self.assertRaises(ValueError):
            matom.atype = ""

        # Try to change the name to None.
        with self.assertRaises(TypeError):
            matom.atype = None

        # Try to change the name to anything other than a string.
        with self.assertRaises(TypeError):
            matom.atype = 6

        # Result must be a string.
        matom.atype = f"6"
        self.assertIsInstance(matom.atype, str)
        self.assertEqual(matom.atype, f"6")

    def test_change_coordinates(self):
        """
            Tests the creation of the atom and changing its coordinates.
        """

        # Randomly choose a radius, mass and set of coordinates.
        mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
        coordinate = array(
                [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
        )

        # Create an atom.
        matom = atom.Atom(radius, mass, coordinate)

        # Change the coordinates to an invalid dimension.
        with self.assertRaises(ValueError):
            matom.coordinates = [2, 3]

        with self.assertRaises(ValueError):
            matom.coordinates = [2, 3, 4, 5]

        # Change the coordinates to something valid.
        previous = len(matom.coordinates)
        ncoordinates = [2, 3, 4]
        matom.coordinates = ncoordinates

        # Check the length.
        self.assertEqual(previous, len(matom.coordinates))

        # Check the entriesa are consistent.
        for coordinate_0, coordinate_1 in zip(matom.coordinates, ncoordinates):
            self.assertEqual(float64, coordinate_0.dtype.type)
            self.assertEqual(coordinate_0, float64(coordinate_1))

    def test_change_mass(self):
        """
            Tests the creation of the atom and changing its mass.
        """

        # Randomly choose a radius, mass and set of coordinates.
        mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
        coordinate = array(
                [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
        )

        # Create an atom.
        matom = atom.Atom(radius, mass, coordinate)

        # Try to set a zero or negative mass.
        for tmass in [0.0, rd.uniform(0.01, 10.0)]:
            with self.assertRaises(ValueError):
                matom.mass = -tmass

        # Try to set with the wrong type.
        with self.assertRaises(TypeError):
            matom.mass = f"{tmass}"

    def test_change_radius(self):
        """
            Tests the creation of the atom and changing its radius.
        """

        # Randomly choose a radius, mass and set of coordinates.
        mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
        coordinate = array(
                [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
        )

        # Create an atom.
        matom = atom.Atom(radius, mass, coordinate)

        # Try to set a zero or negative mass.
        for tradius in [0.0, rd.uniform(0.01, 10.0)]:
            with self.assertRaises(ValueError):
                matom.radius = -tradius

        # Try to set with the wrong type.
        with self.assertRaises(TypeError):
            matom.radius = f"sfsfsfd"

    def test_setup_from_atype(self):
        """
            Tests changing the type of the atom.
        """

        # Randomly choose a radius, mass and set of coordinates.
        mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
        coordinate = array(
            [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
        )

        # Create an atom.
        matom = atom.Atom(radius, mass, coordinate)

        # Set the type to a random string.
        matom.atype = "".join(rd.choices(string.ascii_letters, k=5))

        # It must raise a warning.
        with self.assertWarns(UserWarning):
            matom.set_from_elements()

        # Change the type to carbon.
        element = mendeleev.element("C")
        matom.atype = "C"

        # Set the atom and test.
        matom.set_from_elements()
        self.assertEqual(element.mass, matom.mass)
        self.assertEqual(element.vdw_radius / 100.0, matom.radius)

    # --------------------------------------------------------------------------
    # Creation test.
    # --------------------------------------------------------------------------

    def test_creation(self):
        """
            Tests the creation of the atom and the basic quantities.
        """

        # Test several times.
        for _ in range(ITERATIONS):
            # Choose a random mass and radius
            mass, radius = rd.uniform(0.01, 10.0), rd.uniform(0.01, 10.0)
            coordinate = array(
                [rd.uniform(-10.0, 10.0) for _ in range(3)], dtype=float
            )

            # Create an atom.
            matom = atom.Atom(radius, mass, coordinate)

            # Check that the atom is properly created.
            self.assertEqual(mass, matom.mass)
            self.assertEqual(radius, matom.radius)

            # Check the remaining variables.
            self.assertEqual("---", matom.aname)
            self.assertEqual("---", matom.atype)


# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
