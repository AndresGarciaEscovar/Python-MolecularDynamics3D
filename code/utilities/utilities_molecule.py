"""
    File that contains several utility functions to calculate properties of
    molecules.
"""

# ##############################################################################
# Imports
# ##############################################################################


# General.
import numpy as np
import yaml

# User defined.
import code.main.atom as atom
import code.validation.validation_parameters as vparameters


# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Get Functions
# ------------------------------------------------------------------------------


def get_cod(diffusion_tensor: np.ndarray) -> np.ndarray:
    """
        This function is used to get the center of diffusion from the diffusion
        tensor.

        :param diffusion_tensor: The diffusion tensor.

        :return: The center of diffusion, given the diffusion tensor.
    """
    print("get_cod needs to be implemented.")
    return np.zeros(3, dtype=float)


def get_cog(atoms: list) -> np.ndarray:
    """
        This function is used to get the center of geometry from the atom
        coordinates.

        :param atoms: The list of atoms.

        :return: The center of geometry of the molecule.
    """
    # Extract the needed properties.
    radii = np.array([atom.radius for atom in atoms], dtype=float)
    coords = np.array([atom.coordinates for atom in atoms], dtype=float)

    # Calculate the center of geometry.
    pcoords = np.array([x + y for x, y in zip(coords, radii)], dtype=float)
    ncoords = np.array([x - y for x, y in zip(coords, radii)], dtype=float)

    # The number of dimensions.
    mrange = len(coords[0])

    # Get the maximum and minimum coordinates for each axis.
    cmax = [pcoords[:,i].max(initial=None) for i in range(mrange)]
    cmin = [ncoords[:,i].min(initial=None) for i in range(mrange)]
    
    # Convert the lists to numpy arrays.
    cmax = np.array(cmax, dtype=float)
    cmin = np.array(cmin, dtype=float)

    return (cmax + cmin) * 0.5


def get_com(atoms: atom.Atom) -> np.ndarray:
    """
        This function is used to get the center of mass from the atom 
        coordinates.

        :param atoms: The list of atoms.

        :return: The center of mass of the molecule.
    """
    # Extract the needed properties.
    masses = np.array([atom.mass for atom in atoms], dtype=float)
    coords = np.array([atom.coordinates for atom in atoms], dtype=float)

    # Calculate the center of geometry.
    com = np.array([x * y for x, y in zip(coords, masses)], dtype=float)
    com = sum(com)

    return com / masses.sum(initial=None)


def get_parameters(path: str) -> dict:
    """
        This function is used to get the simulation and molecule parameters.

        :return: A dictionary with the simulation and molecule parameters.
    """
    # Verify that the file exists and it is a YAML file.
    vparameters.is_yaml(path)

    # Get the dictionary of parameters from the yaml file.
    with open(path, "r") as file:
        parameters = yaml.safe_load(file)
    
    return parameters
