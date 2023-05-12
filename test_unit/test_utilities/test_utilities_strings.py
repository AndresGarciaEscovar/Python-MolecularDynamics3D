"""
    File that contains the tests for the molecule utilities.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General
import unittest

from numpy import array

# User defined.
import code.utilities.utilities_strings as ustrings

# ##############################################################################
# Classes
# ##############################################################################


class TestUtilitiesStrings(unittest.TestCase):

    # ##########################################################################
    # Auxiliary Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Auxiliary Print Test Methods
    # --------------------------------------------------------------------------

    def print_string_molecule(self):
        """
            Tests that printing the molecule information works.
        """
        # Print a section banner.
        self.print_banner("Test Printing a Molecule String")

        # Define the coordinates.
        coordinates = array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], dtype=float)
        radii = array([10, 11, 12], dtype=float)
        masses = array([13, 14, 15], dtype=float)

        print(ustrings.get_string_molecule(coordinates, radii, masses))
        print()

    def print_string_vector(self):
        """
            Tests that printing an array works.
        """
        # Print a section banner.
        self.print_banner("Test Printing a Vector")

        # Define a vector.
        arr = array([1, 2, 3], dtype=float)

        ustrings.print_vector(arr)
        print()

    # --------------------------------------------------------------------------
    # Auxiliary Print Methods
    # --------------------------------------------------------------------------

    def print_banner(self, title: str, character: str = "-") -> None:
        """
            Prints a banner with the corresponding title.

            :param title: The title of the banner.

            :param character: The character to use as the banner enclosure.
        """
        # Just to use self.
        self.assertTrue(True)

        print("# " + f"{character}" * 78)
        print(f"# {title}" )
        print("# " + f"{character}" * 78)
        print()

    # ##########################################################################
    # Tests Methods
    # ##########################################################################

    def test_print_strings(self):
        """
            Tests to print other strings.
        """
        self.print_string_molecule()
        self.print_string_vector()

# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
