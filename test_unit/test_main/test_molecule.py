"""
    File that contains the unit test for setting up a molecule.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General
import os
import unittest

# User defined.
import code.managers.context_manager_swd as cm_swd
import code.main.molecule as molecule

# ##############################################################################
# Classes
# ##############################################################################

class MoleculeManager:
    """
        Context manager that sets up the environment where the molecule file is
        created and destroyed to create a simple linear molecule.
    """




class TestMolecule(unittest.TestCase):

    def test_change_aname(self):
        """
            Tests the creation of the atom and changing its name.
        """
        # Randomly choose a radius, mass and set of coordinates.
        location = os.path.dirname(__file__)

        # Set the working directory in the current location.
        with cm_swd.SetWD(location) as wdir:
            print(wdir)



# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
