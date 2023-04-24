"""
    File that contains the validation functions for different features of
    parameters.
"""


# ##############################################################################
# Imports
# ##############################################################################

# General.
from numpy import dtype, float64, ndarray
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
    if not isinstance(number, (float, dtype("float64").type, int)):
        raise TypeError(f"The given object is not a real number.")


def is_ndarray(array: ndarray, atype: type, length: int) -> None:
    """
        Validates all the features of the ndarray.

        :param array: The object to be validated as the numpy array.

        :param atype: The expected type of the array.

        :param length: The expected length of the array.
    """

    # Validate the three features.
    is_ndarray_type(array)
    is_ndarray_dtype(array, atype)
    is_ndarray_length(array, length)


def is_ndarray_dtype(array: ndarray, atype: type) -> None:
    """
        Validates that the entries of the given array are all of the same type.

        :param array: The array whose type is going to be validated.

        :param atype: The expected type of the array.

        :raise TypeError: If the elements in the ndarray do not match the
         required type.
    """
    if array.dtype.type is not atype:
        raise TypeError(
            f"The numpy array is not of the proper type. It's entries are of "
            f"type {array.dtype}, but must be of type {atype}."
        )


def is_ndarray_length(array: ndarray, length: int) -> None:
    """
        Validates that the given numpy array has the proper length.

        :param array: The array whose length is going to be validated.

        :param length: The expected length of the array.

        :raise TypeError: If the length of the numpy array doesn't match the
         expected length.
    """

    # Customize the name if needed.
    if len(array) != length:
        raise ValueError(
            f"The numpy array is not of the proper length. The current number "
            f"of entries is {len(array)} and it should have {length}."
        )


def is_ndarray_type(array: Any) -> None:
    """
        Validates that the array is an ndarray.

        :param array: The object to be validated as the numpy array.

        :raise TypeError: If array object is not an ndarray.
    """
    if not isinstance(array, ndarray):
        raise TypeError(
            f"The given object is not a numpy array; it should be a numpy "
            f"array."
        )


def is_negative(number: Any, include: bool = False) -> None:
    """
        Validates if the given number is a negative floating point number, i.e.,
        a float, and it's greater than zero.

        :param number: The number to be tested.

        :param include: If zero must be included.

        :raise ValueError: If the number is not a negative number.
    """
    # Check it's a float.
    is_float(number)

    if not include and not number < 0.0:
        raise ValueError(
            f"The given floating point number is a positive number; it should "
            f"be a negative number and non-zero."
        )

    # Check if the number is not negative or zero.
    if include and not number <= 0.0:
        raise ValueError(
            f"The given floating point number is a positive number; it should "
            f"be a negative number, or zero."
        )


def is_not_in_dictionary(attribute: Any, dictionary: dict) -> None:
    """
        Checks that the given parameter exists in the dictionary, otherwise it
        raises an error.

        :param attribute: The attribute name to be checked.

        :param dictionary: The dictionary that potentiall contains the
         parameter.

        :raise AttributeError: If the attribute is not in the given dictionary.
    """
    # Check the key is NOT in the dictionary.
    if attribute in dictionary.keys():
        raise AttributeError(
            f"The attibute '{attribute}' already exists and can only be "
            f"initialized once."
        )


def is_not_none(tobject: Any) -> None:
    """
        Validates that the given parameter is not noea string.

        :param tobject: The object whose type is to be examined.

        :raise TypeError: If the given object is None.
    """
    # Check the object is not None.
    if tobject is None:
        raise TypeError(
            f"The given object is None; it must take a non-None value."
        )


def is_positive(number: Any, include: bool = False) -> None:
    """
        Validates if the given number is a floating point number, i.e., a float.
        and it's greater than zero.

        :param number: The number to be tested.

        :param include: If zero must be included.

        :raise ValueError: If the number is not a positive number.
    """
    # Check it's a float.
    is_float(number)

    # Check if it's a negative number.
    if not include and not number > 0.0:
        raise ValueError(
            f"The given floating point number is a positive number; it should "
            f"be a positive number and non-zero."
        )

    # Check if the number is not postive or zero.
    if include and not number >= 0.0:
        raise ValueError(
            f"The given floating point number is a negative number; it should "
            f"be a positive number, or zero."
        )


def is_shape_matrix(matrix: Any, dimensions: tuple) -> None:
    """
        Validates that the given matrix has the proper shape and it's a
        numerical matrix.

        :param matrix: The matrix to be validated.

        :param dimensions: The expected dimensions of the numpy array.

        :raise ValueError: If the given string is not an empty string.

        :raise TypeError: If the given object is not a string.
    """

    # Check it's a numpy array and has the proper type.
    is_ndarray_type(matrix)
    is_ndarray_dtype(matrix, float64)

    # Check the right dimensions.
    if matrix.shape != dimensions:
        raise ValueError(
            f"The given matrix has the wrong shape. Expected shape: "
            f"{dimensions}, current shape: {matrix.shape}."
        )


def is_string(string: Any, strip: bool = False, empty: bool = False) -> None:
    """
        Validates that the given string representation of the string is an empty
        string.

        :param string: The string to be validated.

        :param strip: A boolean flag indicating if the string must be stripped
         before making the comparison. True  if the string must be stripped
         before making the comparison; False, otherwise. True by default.

        :param empty: A boolean flag that indicates if the string is to be
         checked to be empty. True, if the string cannot be empty; False,
         otherwise.

        :raise ValueError: If the given string is not an empty string.

        :raise TypeError: If the given object is not a string.
    """
    # Check the object is a string.
    if not isinstance(string, str):
        raise TypeError(f"The given object is not a string.")

    # Check the string is not empty.
    tstring = string.strip() if strip else string
    if empty and tstring == "":
        raise ValueError("The string should not be an empty string.")


def is_yaml(path: str) -> None:
    """
        Validates that the given string corresponds to the path of a yaml file.

        :param path: The string to be validated.

        :raise TypeError: If the given string does not represent a yaml file.
    """
    # Validate it's a string.
    is_string(path)

    # Check it is a yaml file.
    if not (path.endswith(".yaml") or path.endswith(".yml")):
        raise TypeError(
            f"The given file location: {path}, doesn't correspond to a "
            f"yaml file. The file must have a '.yaml' or a '.yml' extension."
        )
