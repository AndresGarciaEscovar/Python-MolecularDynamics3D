"""
    File that contains the validation functions for different features of
    parameters.
"""


# ##############################################################################
# Imports
# ##############################################################################

# General.
import numpy as np
from typing import Any

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Is Functions
# ------------------------------------------------------------------------------


def is_float(number: Any) -> None:
    """
        Validates that the given string representation of the string is an empty
        string.

        :param number: The number to be validated.

        :raise ValueError: If the given number is not a floating point number
         or an integer.
    """
    # Check the object is a valid float.
    if not isinstance(number, (float, np.dtype("float64").type, int)):
        raise TypeError(f"The given object is not a real number.")


def is_matrix(array: np.ndarray, shape: tuple, message: str = None) -> None:
    """
        Validates that the given numpy array has the proper shape.

        :param array: The array whose shape is going to be validated.

        :param shape: The expected shape of the array.

        :param message: The message to be displayed if the shape is not correct.
         Default is None.
    """
    # Check the shape of the array.
    if array.shape != shape:
        # Format the message.
        message = " " if message is None else f", {message}, "

        raise ValueError(
            f"The numpy array{message}is not of the proper shape. The current"
            f"shape is {array.shape} and it should be {shape}."
        )


def is_negative(number: Any, zero: bool = False) -> None:
    """
        Validates that the given number is negative.

        :param number: The number to be validated.

        :param zero: If True, the number is allowed to be zero; False otherwise.
         Default is False.

        :raise ValueError: If the given number is not negative.
    """
    # Check the object is a valid float.
    is_float(number)

    # Check the number is positive.
    if (number > 0.0 if zero else number >= 0.0):
        raise ValueError(f"The given number is not negative.")


def is_positive(number: Any, zero: bool = False) -> None:
    """
        Validates that the given number is positive.

        :param number: The number to be validated.

        :param zero: If True, the number is allowed to be zero; False otherwise.
         Default is False.

        :raise ValueError: If the given number is not positive.
    """
    # Check the object is a valid float.
    is_float(number)

    # Check the number is positive.
    if (number < 0.0 if zero else number <= 0.0):
        raise ValueError(f"The given number is not positive.")


def is_string_empty(string: Any, empty: bool =True) -> None:
    """
        Validates that the given string representation of the string is not an
        empty string.

        :param string: The string to be validated.

        :param not_empty: If True, the string is not allowed to be empty; False
         otherwise. Default is False.
    """
    # Check the object is a valid string.
    if not isinstance(string, str):
        raise TypeError(f"The given object is not a string.")

    # Check the string is not empty.
    if empty and string.strip() == "":
        raise ValueError(f"The given string is empty. It must not be empty.")


def is_yaml(path: str) -> None:
    """
        Validates that the given string corresponds to the path of a yaml file.

        :param path: The string to be validated.

        :raise TypeError: If the given string does not represent a yaml file.
    """
    # Validate it's a string.
    is_string_empty(path)

    # Check it is a yaml file.
    if not (path.endswith(".yaml") or path.endswith(".yml")):
        raise TypeError(
            f"The given file location: {path}, doesn't correspond to a "
            f"yaml file. The file must have a '.yaml' or a '.yml' extension."
        )
