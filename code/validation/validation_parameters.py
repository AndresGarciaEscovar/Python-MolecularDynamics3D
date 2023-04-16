"""
    File that contains the validation functions for different features of
    parameters.
"""


# ##############################################################################
# Imports
# ##############################################################################

# General.
from typing import Any

# ##############################################################################
# Functions
# ##############################################################################


# ------------------------------------------------------------------------------
# Miscellaneous Functions
# ------------------------------------------------------------------------------


def exists_in_dict(attribute: str, dictionary: dict) -> None:
    """
        Checks that the given parameter exists in the dictionary, otherwise it
        raises an error.

        :param attribute: The attribute name to be checked.

        :param dictionary: The dictionary that potentiall contains the
         parameter.

        :raise AttributeError: If the attribute is not in the given dictionary.
    """
    if attribute in dictionary:
        raise AttributeError(
            f"The attibute '{attribute}' already exists and can only be "
            f"initialized once."
        )


def none_not(robj: Any, name: str = None) -> None:
    """
        Validates that the given parameter is not noea string.

        :param robj: The object whose type is to be examined.

        :param name: The name of the object, optional.

        :raise TypeError: If the given object is None.
    """
    if robj is None:
        name = "" if name is None else f", {name},"
        raise TypeError (
            f"The given object{name} is None; it must take a non-None value."
        )


def srepr_empty(robj: Any, strip: bool = True, name: str = None) -> None:
    """
        Validates that the given string representation of the string is an empty
        string.

        :param robj: The object whose string representation is to be examined.

        :param strip: A boolean flag indicating if the string must be stripped
         before making the comparison. True  if the string must be stripped
         before making the comparison; False, otherwise. True by default.

        :param name: The name of the object, optional.

        :raise ValueError: If the given string representation of the object is
         not an empty tring.
    """

    # Check that the object is not none.
    none_not(robj, name)

    # Strip, or not.
    tstring = f"{robj}".strip() if strip else f"{robj}"

    # Must throw and exception.
    if tstring != "":
        name = "" if name is None else f", {name},"
        raise ValueError(
            f"The given object{name} string representation is not empty, when "
            f"if should be."
        )


def srepr_empty_not(robj: Any, strip: bool = True, name: str = None) -> None:
    """
        Validates that the given string representation of the string is not an
        empty string.

        :param robj: The object whose string representation is to be examined.

        :param strip: A boolean flag indicating if the string must be stripped
         before making the comparison. True  if the string must be stripped
         before making the comparison; False, otherwise. True by default.

        :param name: The name of the object, optional.

        :raise ValueError: If the given string representation of the object is
         not an empty tring.
    """

    # Check that the object is not none.
    none_not(robj, name)

    # Strip, or not.
    tstring = f"{robj}".strip() if strip else f"{robj}"

    # Must throw and exception.
    if tstring == "":
        name = "" if name is None else f", {name},"
        raise ValueError(
            f"The given object{name} string representation is empty, when it "
            f"shouldn't be."
        )
