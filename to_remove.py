"""
    General Tests
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import numpy as np

# ##############################################################################
# Functions
# ##############################################################################


def test_is_instance() -> None:
    """
        Tests that it's instance works without tuples.
    """
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
    print(
        "Generating an empty numpy array:\n\tcoordinates = np.empty((0, 2), "
        "dtype=float)."
    )
    coordinates = np.empty((0, 2), dtype=float)

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
    coordinates = np.append(coordinates, [tcoordinates], axis=0)

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
    coordinates = np.append(coordinates, [tcoordinates], axis=0)

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
        np.append(coordinates, [tcoordinates], axis=0)

    except ValueError as e:
        print(f"We get the message: '{e}'")

    print()


def test_numpy_array_remove() -> None:
    """
        Tests that when an array is to an empty numpy array it is added as an
        array and when a non-empty array is added to a numpy array, it will
        generate an error if the dimensions don't match.
    """
    print(
        "Create a numpy array with a single element:\n\tcoordinates = "
        "np.array(((0, 2),(0, 1)), dtype=float)."
    )
    coordinates = np.array(((0, 2), (0, 1)), dtype=float)

    print(
        f"Print to see how the array looks:\n\tprint(coordinates) ->\n"
        f"{coordinates}\ncannot delete elements using the delete command: del "
        f"coordinate[index] -> FORBIDDEN!\n"
    )


def test_print_none() -> None:
    """
        Tests how the None parameter prints.
    """
    print(f"The string for None is: {None}")


# ##############################################################################
# Main Function
# ##############################################################################


def main() -> None:
    """
        Runs the main whole program.
    """

    # Print the None pointer.
    test_is_instance()
    test_numpy_array_append()
    test_numpy_array_remove()
    test_print_none()

# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == "__main__":
    """
        Runs the main function.
    """
    main()
