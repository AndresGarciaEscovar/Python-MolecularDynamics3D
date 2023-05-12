"""
    File that contains utility functions for strings related to the molecules.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
from numpy import ndarray

# User defined.
import code.validation.validation_parameters as vparamaters

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Get Functions.
# ------------------------------------------------------------------------------


def get_string_vector(array: ndarray, precision: int = 7) -> str:
    """
        Gets the string representation of the given numpy array, to the given
        precision.

        :param array: The numpy array to be turned into string representation.

        :param precision: The precision with which the floating point numbers
         must be represented; seven significant figures, by default.

        :return: The string representation of the given array.
    """
    return "(" + ", ".join(f"{e:+.{precision}e}" for e in array) + ")"


def get_string_matrix(matrix: ndarray, precision: int = 7) -> str:
    """
        Gets the string representation of the given matrix, to the given
        precision.

        :param matrix: The matrix to be turned into string representation.

        :param precision: The precision with which the floating point numbers
         must be represented; seven significant figures, by default.

        :return: The string representation of all the matrix rows.
    """

    # String where the results will be stored.
    string = []

    # For each coordinate.
    for i, row in enumerate(matrix):
        string.append(get_string_vector(row, precision))

    return "\n".join(string)


def get_string_molecule(
        coordinates: ndarray, radii: ndarray, masses: ndarray,
        precision: int = 7
) -> str:
    """
        Gets the string representation of the given molecule.

        :param coordinates: The coordinates of each atom in the molecule.

        :param radii: The radius of each atom in the molecule.

        :param masses: The mass of each atom in the molecule.

        :param precision: The precision with which the floating point numbers
         must be represented; seven significant figures, by default.

        :return: The string representation of all the molecule.
    """

    # Auxiliary variables.
    string_list = [[
            "#", "Coordinates (x,y,z) (Angstrom)", "Radius (Angstrom)",
            "Mass (AMU)"
    ]]

    # Width of each column.
    width = list(map(len, string_list[0]))

    # For each quantity.
    for i, (crd, r, m) in enumerate(zip(coordinates, radii, masses)):
        # The entry tuple.
        entry = [
            f"{i + 1}.", get_string_vector(crd, precision),
            f"{r:.{precision}e}", f"{m:.{precision}e}"
        ]

        # Get the maximum column width.
        width = list(map(lambda x, y: max(x, len(y)), width, entry))

        # Append the entry.
        string_list.append(entry)

    # Format each string.
    for i, stg in enumerate(string_list):
        # Initial string is centered.
        if i == 0:
            string_list[i] = [f"{stn:^{width[j]}}" for j, stn in enumerate(stg)]
            string_list[i] = "   ".join(string_list[i])
            continue

        # Other strings are not centered.
        string_list[i] = [f"{stn:<{width[j]}}" for j, stn in enumerate(stg)]
        string_list[i] = "   ".join(string_list[i])

    return "\n".join(string_list)


# ------------------------------------------------------------------------------
# Print Function.
# ------------------------------------------------------------------------------

def print_matrix(matrix: ndarray) -> None:
    """
        Prints the given 2D matrix.

        :param matrix: The 2D matrix to be printed.
    """
    # Check it's a two dimensional matrix.
    vparamaters.is_shape_matrix(matrix, (len(matrix[0]), len(matrix)))

    # Print each row.
    for row in matrix:
        print("|" + " ".join(f"{x:+.7e}" for x in row) + "|")


def print_vector(vector: ndarray) -> None:
    """
        Prints the given 1D vector.

        :param vector: The 1D vector to be printed.
    """
    # Check it's a two dimensional matrix.
    vparamaters.is_shape_matrix(vector, (len(vector),))

    # Print each row.
    print(get_string_vector(vector, precision=7))
