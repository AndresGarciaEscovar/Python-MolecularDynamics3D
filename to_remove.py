"""
    General Tests
"""

# ##############################################################################
# Imports
# ##############################################################################


# ##############################################################################
# Functions
# ##############################################################################


def test_is_instance() -> None:
    """
        Tests that it's instance works without tuples.
    """
    print("Testing it's instance: ", end="")
    print(isinstance(" ", str))


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
    test_print_none()

# ##############################################################################
# Main Program
# ##############################################################################


if __name__ == "__main__":
    """
        Runs the main function.
    """
    main()
