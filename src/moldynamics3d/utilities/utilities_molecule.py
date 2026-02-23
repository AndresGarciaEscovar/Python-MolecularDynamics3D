"""
    File that contains several utility functions to calculate properties of
    molecules.
"""

# ##############################################################################
# Imports
# ##############################################################################


# General.
import copy
import numpy as np
import yaml

# User defined.
import code.validation.validation_parameters as vparameters


# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Get Functions
# ------------------------------------------------------------------------------


def get_axes(atoms: list, step: float = 1.0e-3) -> tuple:
    """
        Gets the longest and shortest axis of the molecule.

        :param atoms: The list of atoms.

        :param step: The step to use to calculate the angles.

        :return: A tuple with the longest and shortest axis of the molecule.
    """
    # Get the coordinates of the atoms.
    coordinates = np.array([atom.coordinates for atom in atoms], dtype=float)
    radii = np.array([atom.radius for atom in atoms], dtype=float)

    # Define the longest and shortest axis.
    azimuthal = np.arange(0.0, step + np.pi * 0.5, step=step)
    polar = np.arange(0.0, step + np.pi, step=step)

    # Get the directive cosines.
    cazimuthal = np.cos(azimuthal)
    cpolar = np.cos(polar)

    # Get the directive sines.
    sazimuthal = np.sin(azimuthal)
    spolar = np.sin(polar)

    length_azimuth = len(azimuthal)
    length_polar = len(polar)

    # No need to store these.
    del azimuthal
    del polar

    # Max final.
    max_final = -np.inf
    max_vector = None

    # Min final.
    min_final = np.inf
    min_vector = None

    # Iterate over the angles.
    for i in range(length_polar):
        for j in range(length_azimuth):
            # Get the unit vector.
            vector = np.array([
                cpolar[i] * sazimuthal[j],
                spolar[i] * sazimuthal[j],
                cazimuthal[j]
            ])

            # Get the projections.
            projections = np.dot(coordinates, vector)

            # Get  the maximum and minimum points.
            max_point = (projections + radii).max(initial=None)
            min_point = (projections - radii).min(initial=None)

            # Difference between the maximum and minimum points, along the axis.
            difference = (max_point - min_point)

            # Longest axis.
            if difference > max_final:
                max_final = difference
                max_vector = copy.deepcopy(vector)

            # Shortest axis.
            if difference < min_final:
                min_final = difference
                min_vector = copy.deepcopy(vector)

    return (max_final, max_vector), (min_final, min_vector)


def get_cod(diffusion_tensor: np.ndarray) -> np.ndarray:
    """
        This function is used to get the center of diffusion from the diffusion
        tensor.

        :param diffusion_tensor: The diffusion tensor.

        :return: The center of diffusion, given the diffusion tensor.
    """
    # Get the tensors.
    rr = diffusion_tensor[3:6, 3:6]
    tr = diffusion_tensor[3:6, 0:3]

    # Matrix with rotation-rotation
    matrix = np.linalg.inv(np.array(
        [
            [rr[1, 1] + rr[2, 2], -rr[0, 1], -rr[0, 2]],
            [-rr[1, 0], rr[0, 0] + rr[2, 2], -rr[1, 2]],
            [-rr[2, 0], -rr[2, 1], rr[0, 0] + rr[1, 1]],
        ],
        dtype=float
    ))

    # Vector from translation-rotation coupling.
    vector = np.array(
        [tr[1, 2] - tr[2, 1], tr[2, 0] - tr[0, 2], tr[0, 1] - tr[1, 0]],
        dtype=float
    )

    return matrix @ vector


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
    cmax = [pcoords[:, i].max(initial=None) for i in range(mrange)]
    cmin = [ncoords[:, i].min(initial=None) for i in range(mrange)]
    
    # Convert the lists to numpy arrays.
    cmax = np.array(cmax, dtype=float)
    cmin = np.array(cmin, dtype=float)

    return (cmax + cmin) * 0.5


def get_com(atoms: list) -> np.ndarray:
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
