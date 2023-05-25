"""
    File that contains the validation functions for different features of
    molecules.
"""


# ##############################################################################
# Imports
# ##############################################################################

# General.
from numpy import ndarray

# User defined.
import code.validation.validation_parameters as vparameters

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Is Functions
# ------------------------------------------------------------------------------


def is_diffusion_tensor(dimensions: int, diffusion_tensor: ndarray) -> None:
    """
        Checks that the diffusion tensor has valid dimensions, given the number
        of dimensions.

        :param dimensions: The dimensionality of the space (e.g., 2D -> 2,
         3D -> 3, etc.).

        :param diffusion_tensor: The numpy array that represents the diffusion
         tensor.

        :raise ValueError: If the diffusion tensor doesn't have the correct
         dimensions.
    """
    # Check the dimensions is a positive number.
    if not isinstance(dimensions, int) or dimensions <= 0:
        value = f"{dimensions}" if isinstance(dimensions, int) else "---"
        raise ValueError(
            f"The number of dimensions must be a positive integer number. Type "
            f"{type(dimensions)}, value: {value}."
        )

    # Intended dimensions of the diffusion tensor.
    length = 2 * dimensions if dimensions > 2 else dimensions + 1
    length = 1 if dimensions == 1 else length

    # Check the shape of the diffusion tensor.
    try:
        vparameters.is_shape_matrix(diffusion_tensor, (length, length))

    except ValueError as e:
        raise ValueError(
            f"The diffusion tensor doesn't have the correct dimensions; must "
            f"be {(length, length)} for a {dimensions}D space. Matrix shape "
            f"validation {e}."
        )


def is_unique_name(name: str, names: tuple) -> None:
    """
        Checks that the given name to an atom in the molecule is unique.

        :param name: The intended name of the atom.

        :param names: The current names of the atoms in the molecule.

        :raise ValueError: If the name is in the given tuple.
    """
    if name in names:
        raise ValueError(
            f"There is already a '{name}' string in the tuple: '{names}', in "
            f"this molecule: {name}. Please make sure all the atoms have "
            f"UNIQUE names."
        )
