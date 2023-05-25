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

from numpy import array, ndarray, sqrt
from pathlib import Path

# User defined.
import code.managers.context_manager_swd as cm_swd
import code.main.molecule as molecule

# ##############################################################################
# Classes
# ##############################################################################


class MoleculeManager2D:
    """
        Context manager that sets up the environment where the 3D molecule file
        is created and destroyed to create a simple linear molecule.
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
        sep = MoleculeManager2D.get_sep()

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
            ("[+1.0,0.0]", "3.0", "1.0"),
            ("[-1.0,0.0]", "3.0", "1.0")
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
                "center of mass.\n",
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

        # Append to the file.
        with open(filename, mode="a", newline="\n") as file:
            # 6x6 diffusion tensor.
            file.writelines([
                "diffusion_tensor:\n",
                f"{indent * 1}- [1.0, 0.0, 0.0]\n",
                f"{indent * 1}- [0.0, 2.0, 0.0]\n",
                f"{indent * 1}- [0.0, 0.0, 3.0]\n",
            ])

            sq0 = sqrt(2)
            sq1 = -sqrt(2)

            # 3x3 diffusion tensor.
            file.writelines([
                "orientation:\n",
                f"{indent * 1}- [{sq0},{sq0}]\n",
                f"{indent * 1}- [{sq0},{sq1}]\n",
            ])

        return filename

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
            Exit the context manager.

            :param exc_type: The type of the exeception thrown.

            :param exc_val: the values of the execeptions thrown.

            :param exc_tb: The traceback of the exceptions thrown.
        """
        # Remove all the directories and files of the temporary files.
        shutil.rmtree(self.tpath)

    def __init__(self, path: str = None, dtensor: bool = False):
        """
            Creates a temporary file with a simple molecule.

            :param path: The path to the temporary directory.

            :param dtensor: A boolean flag that indicates if the diffusion
             tensor entry and the orientation entries should be created. True,
             if the diffusion tensor and orientation must be created; False,
             otherwise
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
        self.tpath = MoleculeManager2D.get_top_dir(f"{self.path}")

        # If the diffusion tensor must be set.
        self.dtensor = bool(dtensor) if dtensor is not None else False


class MoleculeManager3D:
    """
        Context manager that sets up the environment where the 3D molecule file
        is created and destroyed to create a simple linear molecule.
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
        sep = MoleculeManager3D.get_sep()

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
                "center of mass.\n",
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

        # Append to the file.
        with open(filename, mode="a", newline="\n") as file:
            # 6x6 diffusion tensor.
            file.writelines([
                "diffusion_tensor:\n",
                f"{indent * 1}- [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n",
                f"{indent * 1}- [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]\n",
                f"{indent * 1}- [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]\n",
                f"{indent * 1}- [0.0, 0.0, 0.0, 4.0, 0.0, 0.0]\n",
                f"{indent * 1}- [0.0, 0.0, 0.0, 0.0, 5.0, 0.0]\n",
                f"{indent * 1}- [0.0, 0.0, 0.0, 0.0, 0.0, 6.0]\n",
            ])

            sq0 = sqrt(2)
            sq1 = -sqrt(2)

            # 3x3 diffusion tensor.
            file.writelines([
                "orientation:\n",
                f"{indent * 1}- [{sq0},{sq0}, 0.0]\n",
                f"{indent * 1}- [{sq0},{sq1}, 0.0]\n",
                f"{indent * 1}- [{0.0},{0.0}, 1.0]\n"
            ])

        return filename

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
            Exit the context manager.

            :param exc_type: The type of the exeception thrown.

            :param exc_val: the values of the execeptions thrown.

            :param exc_tb: The traceback of the exceptions thrown.
        """
        # Remove all the directories and files of the temporary files.
        shutil.rmtree(self.tpath)

    def __init__(self, path: str = None, dtensor: bool = False):
        """
            Creates a temporary file with a simple molecule.

            :param path: The path to the temporary directory.

            :param dtensor: A boolean flag that indicates if the diffusion
             tensor entry and the orientation entries should be created. True,
             if the diffusion tensor and orientation must be created; False,
             otherwise
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
        self.tpath = MoleculeManager3D.get_top_dir(f"{self.path}")

        # If the diffusion tensor must be set.
        self.dtensor = bool(dtensor) if dtensor is not None else False


class TestMolecule(unittest.TestCase):

    # --------------------------------------------------------------------------
    # Creation test.
    # --------------------------------------------------------------------------

    def test_creation_2D(self):
        """
            Tests the creation of the molecule.
        """
        # Randomly choose a radius, mass and set of coordinates.
        location = os.path.dirname(__file__)

        # Set the working directory in the current location.
        with cm_swd.SetWD(location) as _:
            with MoleculeManager2D(dtensor=True) as mpath:
                # Load the molecule.
                mol = molecule.Molecule(mpath)

        # Check the name of the atoms.
        self.assertEqual(2, len(mol.anames))
        anames = mol.anames

        for ename, name in zip(("atom_1", "atom_2"), anames):
            self.assertEqual(ename, name)

        # Check that the atoms have been created.
        self.assertEqual(2, len(mol.atoms))
        atoms = mol.atoms

        sign = 1
        for i, atom in enumerate(atoms, start=1):
            # String quantities.
            self.assertEqual(f"atom_{i}", atom.aname)
            self.assertEqual("---", atom.atype)

            # Scalar quantities.
            self.assertEqual(1.0, atom.radius)
            self.assertEqual(3.0, atom.mass)

            # Array quantities.
            expected = array([1, 0], dtype=float)
            self.assertEqual(len(atom), len(atom.coordinates))
            for entry0, entry1 in zip(expected, atom.coordinates):
                self.assertEqual(entry0, entry1 * sign)

            # Reverse the sign
            sign *= -1

        # Check the center of geometry.
        self.assertEqual(2, len(mol.cog))

        cog = mol.cog
        self.assertEqual((2,), cog.shape)
        self.assertIsInstance(cog, ndarray)

        for ccog in cog:
            self.assertEqual(0.0, ccog)

        # Check the center of mass.
        self.assertEqual(2, len(mol.com))

        com = mol.com
        self.assertEqual((2,), com.shape)
        self.assertIsInstance(com, ndarray)

        for ccom in com:
            self.assertEqual(0.0, ccom)

        # Check the coordinates.
        self.assertEqual(2, len(mol.coordinates))
        coordinates = mol.coordinates
        self.assertIsInstance(coordinates, ndarray)
        self.assertEqual((2, 2), coordinates.shape)

        ecoordinates = array([[1, 0], [-1, 0]], dtype=float)
        for ecoordinate, coordinate in zip(ecoordinates, coordinates):
            for ecoord, coord in zip(ecoordinate, coordinate):
                self.assertEqual(ecoord, coord)

        # Check the diffusion tensor.
        dtensor = mol.diffusion_tensor
        self.assertEqual(dtensor.shape, (3, 3))

        # Check the dimensionality.
        self.assertEqual(mol.dimensions, len(atoms[0]))
        self.assertIsInstance(mol.dimensions, int)

        # Check the masses.
        self.assertEqual(2, len(mol.masses))

        masses = mol.masses
        self.assertIsInstance(masses, ndarray)
        self.assertEqual((2,), masses.shape)

        for mass in masses:
            self.assertEqual(3.0, mass)

        # Check the name of the molecule.
        mname = mol.name
        self.assertIsInstance(mname, str)
        self.assertEqual("test_molecule", mname)

        # Check the orientation.
        orientation = mol.orientation
        self.assertIsInstance(orientation, ndarray)
        self.assertEqual((2, 2), orientation.shape)
        eorientation = array([
            [sqrt(2), sqrt(2)],
            [sqrt(2), -sqrt(2)],
        ], dtype=float)

        for erorientation, rorientation in zip(eorientation, orientation):
            for erco, rcor in zip(erorientation, rorientation):
                self.assertEqual(erco, rcor)

        # Check the radii.
        self.assertEqual(2, len(mol.radii))

        radii = mol.radii
        self.assertIsInstance(radii, ndarray)
        self.assertEqual((2,), radii.shape)

        for radius in radii:
            self.assertEqual(1.0, radius)

        # ------------ No diffusion tensor or orientation provided ----------- #

        # Set the working directory in the current location.
        with cm_swd.SetWD(location) as _:
            with MoleculeManager2D(dtensor=False) as mpath:
                # Load the molecule.
                mol = molecule.Molecule(mpath)

        # Check the name of the atoms.
        self.assertEqual(2, len(mol.anames))
        anames = mol.anames

        for ename, name in zip(("atom_1", "atom_2"), anames):
            self.assertEqual(ename, name)

        # Check that the atoms have been created.
        self.assertEqual(2, len(mol.atoms))
        atoms = mol.atoms

        sign = 1
        for i, atom in enumerate(atoms, start=1):
            # String quantities.
            self.assertEqual(f"atom_{i}", atom.aname)
            self.assertEqual("---", atom.atype)

            # Scalar quantities.
            self.assertEqual(1.0, atom.radius)
            self.assertEqual(3.0, atom.mass)

            # Array quantities.
            expected = array([1, 0], dtype=float)
            self.assertEqual(len(atom), len(atom.coordinates))
            for entry0, entry1 in zip(expected, atom.coordinates):
                self.assertEqual(entry0 * sign, entry1)

            # Reverse the sign
            sign *= -1

        # Check the center of geometry.
        self.assertEqual(2, len(mol.cog))

        cog = mol.cog
        self.assertEqual((2,), cog.shape)
        self.assertIsInstance(cog, ndarray)

        for ccog in cog:
            self.assertEqual(0.0, ccog)

        # Check the center of mass.
        self.assertEqual(2, len(mol.com))

        com = mol.com
        self.assertEqual((2,), com.shape)
        self.assertIsInstance(com, ndarray)

        for ccom in com:
            self.assertEqual(0.0, ccom)

        # Check the coordinates.
        self.assertEqual(2, len(mol.coordinates))
        coordinates = mol.coordinates
        self.assertIsInstance(coordinates, ndarray)
        self.assertEqual((2, 2), coordinates.shape)

        ecoordinates = array([[1, 0], [-1, 0]], dtype=float)
        for ecoordinate, coordinate in zip(ecoordinates, coordinates):
            for ecoord, coord in zip(ecoordinate, coordinate):
                self.assertEqual(ecoord, coord)

        # Check the diffusion tensor.
        dtensor = mol.diffusion_tensor
        self.assertEqual(dtensor.shape, (3, 3))

        # Check the dimensionality.
        self.assertEqual(mol.dimensions, len(atoms[0]))
        self.assertIsInstance(mol.dimensions, int)

        # Check the masses.
        self.assertEqual(2, len(mol.masses))

        masses = mol.masses
        self.assertIsInstance(masses, ndarray)
        self.assertEqual((2,), masses.shape)

        for mass in masses:
            self.assertEqual(3.0, mass)

        # Check the name of the molecule.
        mname = mol.name
        self.assertIsInstance(mname, str)
        self.assertEqual("test_molecule", mname)

        # Check the orientation.
        orientation = mol.orientation
        self.assertIsInstance(orientation, ndarray)
        self.assertEqual((2, 2), orientation.shape)
        eorientation = array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=float)

        for erorientation, rorientation in zip(eorientation, orientation):
            for erco, rcor in zip(erorientation, rorientation):
                self.assertEqual(erco, rcor)

        # Check the radii.
        self.assertEqual(2, len(mol.radii))

        radii = mol.radii
        self.assertIsInstance(radii, ndarray)
        self.assertEqual((2,), radii.shape)

        for rad in radii:
            self.assertEqual(1.0, rad)

    def test_creation_3D(self):
        """
            Tests the creation of the molecule.
        """
        # Randomly choose a radius, mass and set of coordinates.
        location = os.path.dirname(__file__)

        # Set the working directory in the current location.
        with cm_swd.SetWD(location) as _:
            with MoleculeManager3D(dtensor=True) as mpath:
                # Load the molecule.
                mol = molecule.Molecule(mpath)

        # Check the name of the atoms.
        self.assertEqual(2, len(mol.anames))
        anames = mol.anames

        for ename, name in zip(("atom_1", "atom_2"), anames):
            self.assertEqual(ename, name)

        # Check that the atoms have been created.
        self.assertEqual(2, len(mol.atoms))
        atoms = mol.atoms

        sign = 1
        for i, atom in enumerate(atoms, start=1):
            # String quantities.
            self.assertEqual(f"atom_{i}", atom.aname)
            self.assertEqual("---", atom.atype)

            # Scalar quantities.
            self.assertEqual(1.0, atom.radius)
            self.assertEqual(3.0, atom.mass)

            # Array quantities.
            expected = array([0, 0, 1], dtype=float)
            self.assertEqual(len(atom), len(atom.coordinates))
            for entry0, entry1 in zip(expected, atom.coordinates):
                self.assertEqual(entry0, entry1 * sign)

            # Reverse the sign
            sign *= -1

        # Check the center of geometry.
        self.assertEqual(3, len(mol.cog))

        cog = mol.cog
        self.assertEqual((3,), cog.shape)
        self.assertIsInstance(cog, ndarray)

        for ccog in cog:
            self.assertEqual(0.0, ccog)

        # Check the center of mass.
        self.assertEqual(3, len(mol.com))

        com = mol.com
        self.assertEqual((3,), com.shape)
        self.assertIsInstance(com, ndarray)

        for ccom in com:
            self.assertEqual(0.0, ccom)

        # Check the coordinates.
        self.assertEqual(2, len(mol.coordinates))

        coordinates = mol.coordinates
        self.assertIsInstance(coordinates, ndarray)
        self.assertEqual((2, 3), coordinates.shape)

        ecoordinates = array([[0, 0, 1], [0, 0, -1]], dtype=float)
        for ecoordinate, coordinate in zip(ecoordinates, coordinates):
            for ecoord, coord in zip(ecoordinate, coordinate):
                self.assertEqual(ecoord, coord)

        # Check the diffusion tensor.
        dtensor = mol.diffusion_tensor
        self.assertIsInstance(dtensor, ndarray)
        self.assertEqual((6, 6), dtensor.shape)
        edtensor = array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 6.0]
        ], dtype=float)

        for erdtensor, rdtensor in zip(edtensor, dtensor):
            for ercdtensor, rcdtensor in zip(erdtensor, rdtensor):
                self.assertEqual(ercdtensor, rcdtensor)

        # Check the dimensionality.
        self.assertEqual(mol.dimensions, len(atoms[0]))
        self.assertIsInstance(mol.dimensions, int)

        # Check the masses.
        self.assertEqual(2, len(mol.masses))

        masses = mol.masses
        self.assertIsInstance(masses, ndarray)
        self.assertEqual((2,), masses.shape)

        for mass in masses:
            self.assertEqual(3.0, mass)

        # Check the name of the molecule.
        mname = mol.name
        self.assertIsInstance(mname, str)
        self.assertEqual("test_molecule", mname)

        # Check the orientation.
        orientation = mol.orientation
        self.assertIsInstance(orientation, ndarray)
        self.assertEqual((3, 3), orientation.shape)
        eorientation = array([
            [sqrt(2), sqrt(2), 0.0],
            [sqrt(2), -sqrt(2), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        for erorientation, rorientation in zip(eorientation, orientation):
            for erco, rcor in zip(erorientation, rorientation):
                self.assertEqual(erco, rcor)

        # Check the radii.
        self.assertEqual(2, len(mol.radii))

        radii = mol.radii
        self.assertIsInstance(radii, ndarray)
        self.assertEqual((2,), radii.shape)

        for radius in radii:
            self.assertEqual(1.0, radius)

        # ------------ No diffusion tensor or orientation provided ----------- #

        # Set the working directory in the current location.
        with cm_swd.SetWD(location) as _:
            with MoleculeManager3D(dtensor=False) as mpath:
                # Load the molecule.
                mol = molecule.Molecule(mpath)

        # Check the name of the atoms.
        self.assertEqual(2, len(mol.anames))
        anames = mol.anames

        for ename, name in zip(("atom_1", "atom_2"), anames):
            self.assertEqual(ename, name)

        # Check that the atoms have been created.
        self.assertEqual(2, len(mol.atoms))
        atoms = mol.atoms

        sign = 1
        for i, atom in enumerate(atoms, start=1):
            # String quantities.
            self.assertEqual(f"atom_{i}", atom.aname)
            self.assertEqual("---", atom.atype)

            # Scalar quantities.
            self.assertEqual(1.0, atom.radius)
            self.assertEqual(3.0, atom.mass)

            # Array quantities.
            expected = array([0, 0, 1], dtype=float)
            self.assertEqual(len(atom), len(atom.coordinates))
            for entry0, entry1 in zip(expected, atom.coordinates):
                self.assertEqual(entry0, entry1 * sign)

            # Reverse the sign
            sign *= -1

        # Check the center of geometry.
        self.assertEqual(3, len(mol.cog))

        cog = mol.cog
        self.assertEqual((3,), cog.shape)
        self.assertIsInstance(cog, ndarray)

        for ccog in cog:
            self.assertEqual(0.0, ccog)

        # Check the center of mass.
        self.assertEqual(3, len(mol.com))

        com = mol.com
        self.assertEqual((3,), com.shape)
        self.assertIsInstance(com, ndarray)

        for ccom in com:
            self.assertEqual(0.0, ccom)

        # Check the coordinates.
        self.assertEqual(2, len(mol.coordinates))
        coordinates = mol.coordinates
        self.assertIsInstance(coordinates, ndarray)
        self.assertEqual((2, 3), coordinates.shape)

        ecoordinates = array([[0, 0, 1], [0, 0, -1]], dtype=float)
        for ecoordinate, coordinate in zip(ecoordinates, coordinates):
            for ecoord, coord in zip(ecoordinate, coordinate):
                self.assertEqual(ecoord, coord)

        # Check the diffusion tensor.
        dtensor = mol.diffusion_tensor
        self.assertIsInstance(dtensor, ndarray)
        self.assertEqual((6, 6), dtensor.shape)

        # Check the dimensionality.
        self.assertEqual(mol.dimensions, len(atoms[0]))
        self.assertIsInstance(mol.dimensions, int)

        # Check the masses.
        self.assertEqual(2, len(mol.masses))

        masses = mol.masses
        self.assertIsInstance(masses, ndarray)
        self.assertEqual((2,), masses.shape)

        for mass in masses:
            self.assertEqual(3.0, mass)

        # Check the name of the molecule.
        mname = mol.name
        self.assertIsInstance(mname, str)
        self.assertEqual("test_molecule", mname)

        # Check the orientation.
        orientation = mol.orientation
        self.assertIsInstance(orientation, ndarray)
        self.assertEqual((3, 3), orientation.shape)
        eorientation = array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        for erorientation, rorientation in zip(eorientation, orientation):
            for erco, rcor in zip(erorientation, rorientation):
                self.assertEqual(erco, rcor)

        # Check the radii.
        self.assertEqual(2, len(mol.radii))

        radii = mol.radii
        self.assertIsInstance(radii, ndarray)
        self.assertEqual((2,), radii.shape)

        for rad in radii:
            self.assertEqual(1.0, rad)


# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == '__main__':
    """
        Runs the main program.
    """
    unittest.main()
