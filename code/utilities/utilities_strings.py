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


def get_string_vector(
    array: ndarray, precision: int = 7, characters: tuple = None
) -> str:
    """
        Gets the string representation of the given numpy array, to the given
        precision.

        :param array: The numpy array to be turned into string representation.

        :param precision: The precision with which the floating point numbers
         must be represented; seven significant figures, by default.

        :param characters: The opening and closing characters.

        :return: The string representation of the given array.
    """
    # Set the opening and closing characters.
    chars = ("(", ")") if characters is None else characters

    if not isinstance(chars, tuple):
        raise ValueError(
            "The object that contains the characters to enclose the vector "
            f"must be a 2-tuple. It's currently a {type(chars)}."
        )

    if len(chars) != 2:
        raise ValueError(
            "The number of enclosing characters to print a vector must be two "
            f"(2). The current number is {len(chars)}."
        )

    return (
        f"{chars[0]}" + ", ".join(f"{e:+.{precision}e}" for e in array) +
        f"{chars[1]}"
    )


def get_string_matrix(
    matrix: ndarray, precision: int = 7, characters: tuple = None
) -> str:
    """
        Gets the string representation of the given matrix, to the given
        precision.

        :param matrix: The matrix to be turned into string representation.

        :param precision: The precision with which the floating point numbers
         must be represented; seven significant figures, by default.

        :param characters: The opening and closing characters.

        :return: The string representation of all the matrix rows.
    """

    # String where the results will be stored.
    string = []

    # Choose the right enclosing parentheses.
    chars = ("|", "|") if characters is None else characters

    # For each coordinate.
    for i, row in enumerate(matrix):
        string.append(get_string_vector(row, precision, characters=chars))

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


def print_matrix(matrix: ndarray, characters: tuple = None) -> None:
    """
        Prints the given 2D matrix.

        :param matrix: The 2D matrix to be printed.

        :param characters: The opening and closing characters.
    """
    # Check it's a two dimensional matrix.
    vparamaters.is_shape_matrix(matrix, (len(matrix), len(matrix[0])))

    # Choose the right enclosing parentheses.
    chars = ("|", "|") if characters is None else characters

    print(get_string_matrix(matrix, precision=7, characters=chars))


def print_vector(vector: ndarray, characters: tuple = None) -> None:
    """
        Prints the given 1D vector.

        :param vector: The 1D vector to be printed.

        :param characters: The opening and closing characters.
    """
    # Check it's a two dimensional matrix.
    vparamaters.is_shape_matrix(vector, (len(vector),))

    # Print each row.
    print(get_string_vector(vector, precision=7, characters=characters))
