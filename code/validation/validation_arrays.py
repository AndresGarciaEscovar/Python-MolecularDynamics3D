"""
    File that contains the validation functions for different features of
    arrays.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
from numpy import ndarray
from typing import Any

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Miscellaneous Functions
# ------------------------------------------------------------------------------


def array_ndarray(
    array: ndarray, atype: type, length: int, name: str = None
) -> None:
    """
        Validates all the features of the ndarray.

        :param array: The object to be validated as the numpy array.

        :param atype: The expected type of the array.

        :param length: The expected length of the array.

        :param name: The name of the array being checked (optional).
    """

    # Validate the three features.
    array_ndarray_is(array, name)
    array_ndarray_length(array, length, name)
    array_ndarray_type(array, atype, name)


def array_ndarray_is(array: Any, name: str = None) -> None:
    """
        Validates that the array is an ndarray.

        :param array: The object to be validated as the numpy array.

        :param name: The name of the array being checked (optional).

        :raise TypeError: If array object is not an ndarray.
    """
    if not isinstance(array, (ndarray,)):
        name = " " if name is None else f" , {name}, "
        raise TypeError(f"The given object{name}is not a numpy array!")


def array_ndarray_length(array: ndarray, length: int, name: str = None) -> None:
    """
        Validates that the given numpy array has the proper length.

        :param array: The array whose length is going to be validated.

        :param length: The expected length of the array.

        :param name: The name of the array being checked (optional).

        :raise TypeError: If the length of the numpy array doesn't match the
         expected length.
    """

    # Customize the name if needed.
    if len(array) != length:
        name = " " if name is None else f" , {name}, "
        raise TypeError(
            f"The numpy array{name}is not of the proper length. The current "
            f"number of entries is {len(array)} and it should have {length}."
        )


def array_ndarray_type(array: ndarray, atype: type, name: str = None) -> None:
    """
        Validates that the entries of the given array are all of the same type.

        :param array: The array whose type is going to be validated.

        :param atype: The expected type of the array.

        :param name: The name of the array being checked (optional).

        :raise TypeError: If the elements in the ndarray do not match the
         required type.
    """

    # Customize the name if needed.
    if not isinstance(array.dtype, (atype,)):
        name = " " if name is None else f" , {name}, "
        raise TypeError(
            f"The numpy array{name}is not of the proper type. It's entries are "
            f"of type {array.dtype}, but must be of type {atype}."
        )
