"""
    File that contains the Molecule class and its methods.
"""

# ##############################################################################
# Imports
# ##############################################################################


# General.
import copy
import numpy as np

from pathlib import Path

# User defined.
import code.main.atom as atom
import code.utilities.utilities_molecule as umolecule
import code.validation.validation_parameters as vparameters


# ##############################################################################
# Classes
# ##############################################################################


class Molecule:
    """
        Class that represents a rigid molecule made of spheres. A file with the
        molecule information can be provided to load the molecule.
    """

    # ##########################################################################
    # Constants
    # ##########################################################################

    __info = (
        "# Coordinates are in Angstrom.\n# Mass is in Dalton.\n# Radius is in "
        "Angstrom.\n# Atom type (atype) is not a mandatory field; set to '---' "
        "by default.\n# The diffusion tensor as ALWAYS given with respect to "
        "the center of mass.\n# The orientation is always given with respect "
        "to the center of mass.\n\n"
    )

    # ##########################################################################
    # Properties
    # ##########################################################################

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #

    # ##########################################################################
    # Methods
    # ##########################################################################

    # --------------------------------------------------------------------------
    # Atom Methods
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # 'get' Methods
    # --------------------------------------------------------------------------

    def get_atoms_string(self, yaml: bool = False) -> str:
        """
            Gets the string with the atoms information, set in a proper table;
            or in a yaml style formatted string.

            :param yaml: A boolean flag that indicates if the string format must
             be yaml style. If True, the string will be formatted in a yaml
             style; False, otherwise.

            :return: String with the atoms information.
        """
        # //////////////////////////////////////////////////////////////////////
        # Auxiliar Functions
        # //////////////////////////////////////////////////////////////////////

        def get_table() -> str:
            """
                Gets the string in table form.

                :return: String with the atoms information in table form.
            """
            # Auxiliary variables.
            header = [(
                "#", "Name", "Type", "Coordinates (x,y,z) - \u212B",
                "Mass - Dalton", "Radius - \u212B"
            )]
            atoms = header + [
                (i + 1, *x.get_information()) for i, x in enumerate(self.atoms)
            ]

            # Length of the strings.
            length = list(map(len, header[0]))

            # Get the lengths.
            for i, tatom in enumerate(atoms):
                for j, value in enumerate(tatom):
                    length[j] = max(length[j], len(str(value)))

            # Get the string.
            string = ""
            for i, tatom in enumerate(atoms):
                align = "^" if i == 0 else "<"
                for j, value in enumerate(tatom):
                    string += f"{value:{align}{length[j]}}   "
                string += "\n"

            return string

        def get_yaml() -> str:
            """
                Gets the string in yaml form.

                :return: String with the atoms information in yaml form.
            """
            string = ""
            atoms = [
                x.get_information() for i, x in enumerate(self.atoms)
            ]

            # Get the string.
            for tatom in atoms:
                string += f"{tatom[0]}:\n"
                string += (
                    f"  coordinates: {tatom[2]}\n".replace("(", "[")
                ).replace(")", "]")
                string += f"  mass: {tatom[3]}\n"
                string += f"  radius: {tatom[4]}\n"
                string += f"  atype: {tatom[1]}\n"

            return string

        # //////////////////////////////////////////////////////////////////////
        # Implementation
        # //////////////////////////////////////////////////////////////////////

        # Return the yaml formatted string.
        if yaml:
            return get_yaml()

        return get_table()

    # --------------------------------------------------------------------------
    # 'load' Methods
    # --------------------------------------------------------------------------

    def load(self) -> None:
        """
            Loads the molecule from the file.
        """
        # Get the parameters from the yaml file.
        parameters = umolecule.get_parameters(self.filename)

        # Name of the molecule.
        self.name = parameters["molecule_name"]

        # Get the molecule parameters.
        self.dtensor = np.array(parameters["diffusion_tensor"], dtype=float)
        self.orientation = np.array(parameters["orientation"], dtype=float)
        atoms = parameters["atoms"]

        # Load the atoms.
        for name, information in atoms.items():
            coords = np.array(information["coordinates"], dtype=float)
            atype = information["atype"]
            mass = information["mass"]
            radius = information["radius"]
            self.atoms.append(atom.Atom(name, atype, mass, radius, coords))
        
        # Validate the matrices.
        vparameters.is_matrix(self.dtensor, (6, 6), "diffusion tensor")
        vparameters.is_matrix(self.orientation, (3, 3), "orientation")

    def load_axes(self) -> None:
        """
            Loads the longest and shortest axes of the molecule.
        """
        # Get the longest and shortest axes.
        axes = umolecule.get_axes(self.atoms, step=1.0e-3)

        # Get the longest axis.
        self.longest_axis_lenght = axes[0][0]
        self.longest_axis = axes[0][1]

        # Get the shortest axis.
        self.shortest_axis_lenght = axes[1][0]
        self.shortest_axis = axes[1][1]

    def load_centers(self) -> None:
        """
            Loads the different centers of the molecule.
        """
        self.cod = umolecule.get_cod(self.dtensor)
        self.cog = umolecule.get_cog(self.atoms)
        self.com = umolecule.get_com(self.atoms)

    # --------------------------------------------------------------------------
    # Rotate Methods
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Translate Methods
    # --------------------------------------------------------------------------

    # ##########################################################################
    # Constructor
    # ##########################################################################

    def __init__(self, filename: str, working: str):
        """
            Constructs a new instance of the a molecule. If the name of the file
            is not provided, it will create a single-sphere molecule with a
            radius of 1.0 Angstom (\u212B) and mass of 1.0 Atomic Mass Units
            AMU.

            :param filename: The name of the file from where the molecule must
             be loaded.
            
            :param working: The name of the directory where the simulation
             output is being saved.
        """
        # Initialize the molecule properties.
        self.atoms = []
        self.dtensor = None
        self.name = None
        self.orientation = None

        # Molecule centers, cod = center of diffusion, cog = center of geometry
        # and com = center of mass.
        self.cod = None
        self.cog = None
        self.com = None

        self.longest_axis = None
        self.longest_axis_lenght = None

        self.shortest_axis = None
        self.shortest_axis_lenght = None

        # Set the file name and get the working directory.
        self.filename = f"{Path(filename).resolve()}"
        self.working = f"{Path(working).resolve()}"

        # Load the molecule.
        self.load()
        self.load_centers()

        # Load the long and short axes.
        self.load_axes()

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
        # Set the basic variables.
        string = copy.deepcopy(Molecule.__info).strip() + "\n"
        string += f"molecule_name: \"{self.name}\"\n"

        # Orientation.
        string += f"orientation:\n"
        for ori in self.orientation:
            ori = ', '.join([f"{o:+.7e}" for o in ori])
            string += f"  - [{ori}]\n"

        # Diffusion tensor.
        string += f"diffusion_tensor:\n"
        for ent in self.dtensor:
            ent = ', '.join([f"{o:+.7e}" for o in ent])
            string += f"  - [{ent}]\n"

        # Atoms.
        string += f"atoms:\n  "
        string += self.get_atoms_string(yaml=True).replace("\n", "\n  ")

        return string

    def __str__(self):
        """
            Returns a more sophisticated string representation of the molecule
            to include a better looking table and more information such as the
            center of diffusion, center of geometry, center of mass and
            diffusion tensor; the latter with respect to the center of mass.
        """
        # Atom variables.
        variables = [[
            "#", "Name", "Type", "Coordinates (x,y,z) - \u212B",
            "Mass - Dalton", "Radius - \u212B"
        ]]

        # Molecule name.
        string = copy.deepcopy(Molecule.__info).strip() + "\n"
        string += f"molecule_name: \"{self.name}\"\n"
        string += f"location: \"{self.filename}\"\n"

        # Atoms.
        string += f"atoms: \"{self.name}\"\n"
        string += self.get_atoms_string(yaml=False)

        # Orientation.
        string += f"orientation:\n"
        for ori, crd in zip(self.orientation, ("x'", "y'", "z'")):
            ori = ', '.join([f"{o:+.7e}" for o in ori])
            string += f"    {crd}: [{ori}]\n"\

        # Diffusion tensor.
        string += f"diffusion_tensor:\n"
        for ent in self.dtensor:
            ent = '  '.join([f"{o:+.7e}" for o in ent])
            string += f"    |{ent}|\n"

        return string
