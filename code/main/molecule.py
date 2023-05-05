"""
    File that contains the Molecule class and its methods.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import yaml

from numpy import append as nappend, array, delete as ndelete, ndarray
from pathlib import Path
from typing import Any, Union

# User defined.
import code.main.atom as atom
import code.utilities.utilities_molecule as umolecule
import code.validation.validation_parameters as vparameters
import code.validation.validation_molecule as vmolecule

# ##############################################################################
# Classes
# ##############################################################################


class Molecule:
    """
        Class that represents a rigid molecule made of spheres. A file with the
        molecule information can be provided to load the molecule.
    """

    # ##########################################################################
    # Properties
    # ##########################################################################

    @property
    def anames(self) -> tuple:
        """
            Returns the names of all the atoms in the molecule.

            :return: The tuple that contains all of the names of the atoms in
             the molecule, in the order in which the atoms are stored.
        """
        return tuple(x.aname for x in self.atoms)

    # ------------------------------------------------------------------------ #

    @property
    def atoms(self) -> list:
        """
            Returns the list of atoms in the molecule.

            :return: The list of the atoms in the molecule.
        """
        return self.__atoms

    @atoms.setter
    def atoms(self, atoms: list) -> None:
        """
            Initially sets the list of atoms. Atoms to be added must be added
            using the add atom function.

            :param atoms: The initial list of atoms.
        """
        # Cannot change the atoms if they already exist.
        vparameters.is_not_in_dictionary("_Molecule__atoms", self.__dict__)

        self.__atoms = atoms

    # ------------------------------------------------------------------------ #

    @property
    def cod(self) -> Union[None, ndarray]:
        """
            Returns the numpy array that represents the center of diffusion of
            the molecule.

            :return: The numpy array that represents the center of diffusion of
             the molecule.
        """
        return self.__cod[0]

    @cod.setter
    def cod(self, cod: ndarray) -> None:
        """
            Sets the center of diffusion.

            :param cod: The numpy array that represents the center of diffusion
             of the molecule.
        """
        tcod = array(cod, dtype=float)
        self.__cod = nappend(self.__cod, [tcod], axis=0)
        self.__cod = ndelete(self.__cod, 0, axis=0)

    # ------------------------------------------------------------------------ #

    @property
    def cog(self) -> ndarray:
        """
            Returns the numpy array that represents the center of geometry of
            the molecule.

            :return: The numpy array that represents the center of geometry of
             the molecule.
        """
        return self.__com[0]

    @cog.setter
    def cog(self, cog: ndarray) -> None:
        """
            Sets the center of geometry.

            :param cog: The numpy array that represents the center of geometry
             of the molecule.
        """
        tcog = array(cog, dtype=float)
        self.__cog = nappend(self.__cog, [tcog], axis=0)
        self.__cog = ndelete(self.__cog, 0, axis=0)

    # ------------------------------------------------------------------------ #

    @property
    def com(self) -> ndarray:
        """
            Returns the numpy array that represents the center of mass of the
            molecule.

            :return: The numpy array that represents the center of mass of the
             molecule.
        """
        return self.__com[0]

    @com.setter
    def com(self, com: ndarray) -> None:
        """
            Sets the center of mass.

            :param com: The numpy array that represents the center of mass of
             the molecule.
        """
        tcom = array(com, dtype=float)
        self.__com = nappend(self.__com, [tcom], axis=0)
        self.__com = ndelete(self.__com, 0, axis=0)

    # ------------------------------------------------------------------------ #

    @property
    def coordinates(self) -> ndarray:
        """
            Returns the coordinates of all the atoms in the molecule.

            :return: The numpy array of the coordinates of the atoms in the
             molecule, in the order in which the atoms are stored.
        """
        return array([x.coordinates for x in self.atoms], dtype=float)

    # ------------------------------------------------------------------------ #

    @property
    def diffusion_tensor(self) -> ndarray:
        """
            The diffusion tensor, only exists for 3D molecules.

            :return: The diffusion tensor for the molecule.
        """
        return self.__diffusion_tensor

    # ------------------------------------------------------------------------ #

    @property
    def dimensions(self) -> int:
        """
            The number of coordinates used to described the position of a
            molecule in space.

            :return: The number of coordinates used to described the position of
             a molecule in space.
        """
        return len(self.atoms[0])

    # ------------------------------------------------------------------------ #

    @property
    def masses(self) -> ndarray:
        """
            Returns the masses of all the atoms in the molecule.

            :return: The numpy array of the masses of the atoms in the molecule,
             in the order in which the atoms are stored.
        """
        return array([x.mass for x in self.atoms], dtype=float)

    # ------------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        """
            Returns the string that represents the name of the molecule.

            :return: The string that represents the name of the molecule.
        """
        return self.__name

    @name.setter
    def name(self, name: Any) -> None:
        """
            Sets the name of the molecule to the given strings.

            :param name: The object or string whose string representation is the
             name of the molecule.
        """
        # The molecule must have a non-empty name.
        vparameters.is_string(name, strip=True, empty=True)

        self.__name = name.strip()

    # ------------------------------------------------------------------------ #

    @property
    def orientation(self) -> ndarray:
        """
            Returns the numpy array that represents the orientation of the
            molecule, with respect to the center of mass.

            :return: The numpy array that represents the orientation of the
             molecule, with respect to the center of mass.
        """
        return self.__orientation

    @orientation.setter
    def orientation(self, orientation: ndarray) -> None:
        """
            Sets the orientaiton of the molecule.

            :param orientation: The numpy array that represents the orientation
             of the molecule, with respect to the center of mass.
        """
        # The molecule must have a non-empty name.
        vparameters.is_shape_matrix(
            orientation, (self.dimensions, self.dimensions)
        )

        self.__orientation = orientation

    # ------------------------------------------------------------------------ #

    @property
    def radii(self) -> ndarray:
        """
            Returns the radius of all the atoms in the molecule.

            :return: The numpy array of the radius of the atoms in the molecule,
             in the order in which the atoms are stored.
        """
        return array([x.radius for x in self.atoms], dtype=float)

    # ##########################################################################
    # Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Atom Methods
    # --------------------------------------------------------------------------

    def atom_add(
        self, aname: str, radius: float, mass: float, coordinates: ndarray,
        atype: str = None
    ) -> None:
        """
            Adds an atom to the atom list.

            :param radius: The floating point number that represents the radius
             of the atom.

            :param mass: The floating point number that represents the mass of
             the atom.

            :param coordinates: The numpy array of floating point numbers that
             that represent the coordinates of the atom.

            :param aname: The name of the atom.

            :param atype: The type of the atom; this is an optional parameter.
        """

        # Validate both strings, if needed.
        vparameters.is_string(aname, empty=False)
        if atype is not None:
            vparameters.is_string(atype, empty=False)

        # Format the strings properly.
        aname = "---" if aname.strip() == "" else aname
        atype = "---" if atype is None or atype.strip() == "" else atype

        # Check the atom names are unique.
        vmolecule.is_unique_name(aname, self.anames)

        # Create and add the atom.
        matom = atom.Atom(
            radius, mass, coordinates, atype.strip(), aname.strip()
        )
        self.atoms.append(matom)

    def atom_remove(self, index: int) -> None:
        """
            Removes the atom at the given index.

            :param index: The index of the atom to be removed.
        """

        # Validate the index.
        if index < 0 or index >= len(self):
            raise ValueError(
                "The index of the given atom is out of range; i.e., must be a "
                f"number between 0 and {len(self)}. Requested index: {index}."
            )

        del self.atoms[index]

    # --------------------------------------------------------------------------
    # Clean Methods
    # --------------------------------------------------------------------------

    def clean_atoms(self) -> None:
        """
            Removes all the atoms and leaves an empty list.
        """
        # Remove ALL the atoms.
        while len(self) > 0:
            self.atom_remove(0)

    # --------------------------------------------------------------------------
    # Load an Save Methods
    # --------------------------------------------------------------------------

    def load(self) -> tuple:
        """
            Loads the molecule from the self.filename variable; if the diffusion
            tensor exists, it will return it, along with the orientation of the
            molecule.

            :return: The tuple with the diffusion tensor and the orientation,
             if they exist; otherwise, it will return a tuple with None.
        """

        # Remove all the atoms.
        self.clean_atoms()

        # Load the file.
        with open(self.filename, "r") as stream:
            info = yaml.safe_load(stream)

        # Get the molecule name.
        self.name = info["molecule_name"]

        # Load the atoms.
        for name, iatom in info["atoms"].items():
            # Extract the information.
            atype = f"{iatom['atype']}"
            crds = array(iatom["coordinates"], dtype=float)
            mass = float(iatom["mass"])
            radius = float(iatom["radius"])

            # Add the atom to the system.
            self.atom_add(name, radius, mass, crds, atype)

        # Dimensionality.
        length = len(self.coordinates[0])

        return umolecule.get_dtensor_and_orientation(info, length)

    def save(self, path: str) -> None:
        """
            Saves the configuration of the molecule to a yaml file, with the
            given name.

            :param path: The path where to save the yaml file; must have a yaml
             extension for this to work.
        """

        # Auxiliary variables.
        indent = "    "

        # Validate it's a yaml file.
        path = path.strip()
        vparameters.is_yaml(path)

        # Ge the information of the atoms:
        with open(path, mode="w", newline="\n") as fl:
            # Write the file header.
            fl.writelines([
                "# Coordinates are in Angstrom.\n",
                "# Mass is in AMU.\n",
                "# Radius is in Angstrom.\n",
                "# Atom type (atype) is not a mandatory field; set to '---' by "
                "default.\n",
                "# The diffusion tensor must ALWAYS be given with respect to "
                "the center of mass.\n",
                f"molecule_name: {self.name}\n",
                f"atoms:\n",
            ])

            # For all the atoms.
            for matom in self.atoms:
                # Get a LIST of the coordinates.
                coordinates = ','.join([f"{float(x)}" for x in matom.coordinates])

                # Add the atom properties.
                fl.write(f"{indent * 1}{matom.aname}:\n")
                fl.write(f"{indent * 2}coordinates: [{coordinates}]\n")
                fl.write(f"{indent * 2}mass: {matom.mass}\n")
                fl.write(f"{indent * 2}radius: {matom.radius}\n")
                fl.write(f"{indent * 2}atype: {matom.atype}\n")

            # Add the diffusion tensor.
            fl.write("diffusion_tensor:\n")
            for row in self.diffusion_tensor:
                fl.write(f"{indent * 1}- [{','.join(list(map(str, row)))}]\n")

    # --------------------------------------------------------------------------
    # Rotate Methods
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Translate Methods
    # --------------------------------------------------------------------------

    # ##########################################################################
    # Constructor
    # ##########################################################################

    def __init__(self, filename: str = None):
        """
            Constructs a new instance of the a molecule. If the name of the file
            is not provided, it will create a single-sphere molecule with a
            radius of 1.0 Angstom (\u212B) and mass of 1.0 Atomic Mass Units
            AMU.

            :param filename: The name of the file from where the molecule must
             be loaded.
        """
        # Set the file name.
        self.filename = f"{Path(filename).resolve()}"
        self.name = "<Unnamed molecule>"

        # Create the basic quantities.
        self.atoms = list()

        # Load the molecule.
        self.__diffusion_tensor, self.orientation = self.load()

        # Get the center of geometry and mass of the molecule.
        self.__cog = array([[0.0] * self.dimensions], dtype=float)
        self.cog = umolecule.get_cog(self.coordinates, self.radii)

        self.__com = array([[0.0] * self.dimensions], dtype=float)
        self.com = umolecule.get_com(self.coordinates, self.masses)

        # Set the center of diffusion of the molecule.
        self.__cod = array([[0.0] * self.dimensions], dtype=float)
        self.cod = self.com

        # Fix the diffusion tensor.
        if self.__diffusion_tensor is None and self.dimensions == 3:
            self.__diffusion_tensor = umolecule.get_dtensor(
                self.coordinates, self.masses, -self.com
            )

        # Get the center of diffusion.
        if self.diffusion_tensor is not None:
            self.cod = umolecule.get_cod(self.diffusion_tensor) + self.com

    # ##########################################################################
    # Dunder Methods
    # ##########################################################################

    def __len__(self):
        """
            Returns the number of atoms in the molecule, i.e., the length of
            the atoms array.

            :return: Number of atoms in the molecule.
        """
        return len(self.atoms)

    def __repr__(self):
        """
            Returns a string with a quick represenation of the molecule, i.e.,
            the current coordinate, radius and mass of each atom.
        """
        return ""

    def __str__(self):
        """
            Returns a more sophisticated string representation of the molecule
            to include a better looking table and more information such as the
            center of diffusion, center of geometry, center of mass and
            diffusion tensor; the latter with respect to the center of mass.
        """
        return ""


# ##############################################################################
#
# ##############################################################################

if __name__ == "__main__":

    import os

    # Path from where the molecules are loaded.
    mp0 = f"{Path(os.getcwd(), '..', '..', 'data', 'product.yaml').resolve()}"
    mp1 = f"{Path(os.getcwd(), '..', '..', 'data', 'reactant.yaml').resolve()}"
#
#     # Path to where the molecules are saved.
#     mp2 = f"{Path(os.getcwd(), '..', '..', 'data', 'product_1.yaml').resolve()}"
#     mp3 = f"{Path(os.getcwd(), '..', '..', 'data', 'reactant_1.yaml').resolve()}"
#
    # Load using the absolute path.
    # molecule0 = Molecule(mp0)
    molecule1 = Molecule(mp1)
