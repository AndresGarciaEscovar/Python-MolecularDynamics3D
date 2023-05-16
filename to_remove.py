"""
    General Tests
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import warnings

from numpy import append, array, dot, empty, pi

# User defined.
import code.utilities.utilities_math as umath

# ##############################################################################
# Functions
# ##############################################################################


def test_is_instance() -> None:
    """
        Tests that it's instance works without tuples.
    """
    print_banner("test_is_instance")

    print(
        "Testing it's instance without tuple, i.e, 'isinstance(" ", str)': ",
        end="")
    print(isinstance(" ", str))
    print()


def test_numpy_array_append() -> None:
    """
        Tests that when an array is to an empty numpy array it is added as an
        array and when a non-empty array is added to a numpy array, it will
        generate an error if the dimensions don't match.
    """
    print_banner("test_numpy_array_append")

    print(
        "Generating an empty numpy array:\n\tcoordinates = np.empty((0, 2), "
        "dtype=float)."
    )
    coordinates = empty((0, 2), dtype=float)

    print(
        f"Print to see how the array looks:\n\tprint(coordinates) -> "
        f"{coordinates}"
    )

    print(
        "Try adding a 2D list:"
        "\n\ttcoordinates = [int(1), int(2)]"
        "\n\tcoordinates = np.append(coordinates, [tcoordinates], axis=0)."
    )
    tcoordinates = [int(1), int(2)]
    coordinates = append(coordinates, [tcoordinates], axis=0)

    print(
        f"Print to see how the array looks:\n\tprint(coordinates) -> "
        f"{coordinates}"
    )

    print(
        "Try adding another 2D list:"
        "\n\ttcoordinates = [int(2), int(3)]"
        "\n\tcoordinates = np.append(coordinates, [tcoordinates], axis=0)."
    )

    tcoordinates = [int(2), int(3)]
    coordinates = append(coordinates, [tcoordinates], axis=0)

    print(
        f"Print to see how the array looks:\n\tprint(coordinates) ->\n "
        f"{coordinates}"
    )

    print(
        "Try adding a 3D list:"
        "\n\ttcoordinates = [int(4), int(5), int(6)]"
        "\n\tcoordinates = np.append(coordinates, [tcoordinates], axis=0)."
    )

    tcoordinates = [int(4), int(5), int(6)]
    try:
        append(coordinates, [tcoordinates], axis=0)

    except ValueError as e:
        print(f"We get the message: '{e}'")

    print()


def test_numpy_array_projection() -> None:
    """
        Tests projections on zero vectors.
    """
    print_banner("test_numpy_array_projection")

    print(
        "Define a non-zero vector:\n\tvector_0 = np.array((0, 2), "
        "dtype=float)."
    )
    vector_0 = array((0, 2), dtype=float)

    print(
        "Define a zero vector:\n\tvector_1 = np.array((0, 0), "
        "dtype=float)."
    )
    vector_1 = array((0, 0), dtype=float)

    print(
        f"Try to obtain the projection:\n\tprojection = np.dot(vector_0, "
        f"vector_1) * vector_1 / np.dot(vector_1, vector_1)"
    )

    try:
        snorm = dot(vector_1, vector_1)
        dot(vector_0, vector_1) * vector_1 / snorm
    except RuntimeWarning:
        print("Cannot make the projection; there is a division by zero.")
    print()


def test_numpy_array_remove() -> None:
    """
        Tests that when an array is to an empty numpy array it is added as an
        array and when a non-empty array is added to a numpy array, it will
        generate an error if the dimensions don't match.
    """
    print_banner("test_numpy_array_remove")

    print(
        "Create a numpy array with a single element:\n\tcoordinates = "
        "np.array(((0, 2),(0, 1)), dtype=float)."
    )
    coordinates = array(((0, 2), (0, 1)), dtype=float)

    print(
        f"Print to see how the array looks:\n\tprint(coordinates) ->\n"
        f"{coordinates}\ncannot delete elements using the delete command: del "
        f"coordinate[index] -> FORBIDDEN!\n"
    )


def test_print_none() -> None:
    """
        Tests how the None parameter prints.
    """
    print_banner("test_print_none")

    print(f"The string for None is: {None}", end="\n\n")


def test_temporary() -> None:
    """
        Tests that it's instance works without tuples.
    """
    print_banner("test_temporary")

    vector = array([0, 1, 0], dtype=float)
    around = array([1, 0, 0], dtype=float)
    about = array([0, 0, 0], dtype=float)

    amount = pi * 0.5

    print(umath.rotate_vector(vector, around, amount, about))


# ##############################################################################
# Miscellaneous Function
# ##############################################################################


def print_banner(message: str) -> None:
    """
        Prints the banner with the given message.
    """
    # Banner line.
    line = "#" * 80 + "\n"

    # Print the banner.
    print(f"{line}# {message}\n{line}")


# ##############################################################################
# Main Function
# ##############################################################################


def main() -> None:
    """
        Runs the main whole program.
    """
    # Activate warnings as erorrs.
    warnings.filterwarnings("error")

    # Print the None pointer.
    test_is_instance()
    test_numpy_array_append()
    test_numpy_array_projection()
    test_numpy_array_remove()
    test_print_none()

    # General test that can be removed.
    test_temporary()


# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == "__main__":
    """
        Runs the main function.
    """
    main()
