"""
    File that contains the unit test for setting up a molecule.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General
import os
import unittest

from numpy import array
from pathlib import Path

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

    # ##########################################################################
    # Functions
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Get Functions
    # --------------------------------------------------------------------------

    @staticmethod
    def get_top_dir(path: str):
        """
            Gets the top dir that is not a directory.

            :param path: The path to the directory.

            :return: The top directory that is not a directory.
        """
        # Auxiliary variables.
        sep = MoleculeManager.get_sep()

        # Get the separated path.
        tpath = path.split(sep)

        # Remove the zeroth entry.
        del tpath[0]
        tpath[0] = f"{sep}{tpath[0]}"
        length = len(tpath) + 1

        # Set the top path.
        fpath = f"{Path(*tpath)}"

        # Set the path if neeed.
        for i in range(length):
            if not Path(*tpath[:i]).is_dir():
                fpath = f"{Path(*tpath[:i])}"
                break

        return fpath

    @staticmethod
    def get_sep() -> str:
        """
            Gets the system directory separator.
        """
        return f"{Path(' ', ' ')}".strip()

    # --------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------

    def __enter__(self) -> str:
        """
            Performs the action on entry.

            :return: The name of the file where the molecule was generated.
        """
        # Create the parent directory.
        Path(self.path).mkdir(parents=True, exist_ok=True)

        # Creates a basic molecule.
        filename = f"{Path(self.path, 'temp_mol.yaml')}"

        with open(filename, mode="w", newline="\n") as file:
            file.writelines([
                "# Coordinates are in Angstrom\n",
                "# Mass is in AMU.\n",
                "# Radius is in Angstrom.\n",
                "# Atom type (atype) is not a mandatory field; set to '---' by "
                "default.\n",
                "# The diffusion tensor is always given with respect to the "
                "center of mass.",
                'molecule_name: "test_molecule"\n',
                "atoms:\n",
            ])

        return filename

    def __exit__(self, exc_type, exc_val, exc_tb):
        """

        """
        pass

    def __init__(self, path: str = None, dtensor: bool = False):
        """
            Creates a temporary file with a simple molecule.
        """
        # Get the path.
        tpath = path if path is not None and isinstance(path, str) else ""
        self.path = Path(tpath, "temp").resolve()

        # Fix the path.
        cntr = 0
        while self.path.is_dir():
            self.path = Path(tpath, f"temp{cntr}").resolve()

        # Determine which one is NOT the top directory.
        self.tpath = MoleculeManager.get_top_dir(f"{self.path}")

        # If the diffusion tensor must be set.
        self.dtensor = bool(dtensor) if dtensor is not None else False


class TestMolecule(unittest.TestCase):


    # --------------------------------------------------------------------------
    # Creation test.
    # --------------------------------------------------------------------------

    def test_creation(self):
        """
            Tests the creation of the molecule.
        """
        # Randomly choose a radius, mass and set of coordinates.
        location = os.path.dirname(__file__)

        # Set the working directory in the current location.
        with MoleculeManager() as mpath:
            pass



# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
