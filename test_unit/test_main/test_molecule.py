"""
    File that contains the unit test for setting up a molecule.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General
import os
import shutil
import unittest
import yaml


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

        # Auxiliary variables.
        indent = "    "
        info = (
            ("[0.0,0.0,1.0]", "3.0", "1.0"),
            ("[0.0,0.0,-1.0]", "3.0", "1.0")
        )

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

            # Append the atoms and their information.
            for i, (coordinate, mass, radius) in enumerate(info, start=1):
                file.writelines([
                    f"{indent * 1}atom_{i}:\n",
                    f"{indent * 2}coordinates: {coordinate}\n",
                    f"{indent * 2}mass: {mass}\n",
                    f"{indent * 2}radius: {radius}\n",
                    f'{indent * 2}atype: "---"\n',
                ])

        # Don't worry about this.
        if not self.dtensor:
            return filename

        with open(filename, mode="a", newline="\n") as file:
            file.writelines([
                "diffusion_tensor:\n",
                f"{indent * 1}- [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n",
                f"{indent * 1}- [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]\n",
                f"{indent * 1}- [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]\n",
                f"{indent * 1}- [0.0, 0.0, 0.0, 4.0, 0.0, 0.0]\n",
                f"{indent * 1}- [0.0, 0.0, 0.0, 0.0, 5.0, 0.0]\n",
                f"{indent * 1}- [0.0, 0.0, 0.0, 0.0, 0.0, 6.0]\n",
            ])

        return filename

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
            Exit the context manager.

            :param exc_type: The type of the exeception thrown.

            :param exc_val: the values of the execeptions thrown.

            :param exc_tb: The traceback of the exceptions thrown.
        """
        # Remove all the directories.
        shutil.rmtree(self.tpath)

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
            cntr += 1

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
        with MoleculeManager(dtensor=True) as mpath:
            pass



# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
