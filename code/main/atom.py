# ##############################################################################
# Imports
# ##############################################################################

# General.
import copy
import warnings

import mendeleev
import numpy as np

from sqlalchemy.orm.exc import NoResultFound
from typing import Any
from warnings import warn

# ##############################################################################
# Classes
# ##############################################################################


class Atom:
    """
        Class that represents a spherical atom.

        Parameters:
        ___________
        self.coordinates: numpy.ndarray
         A 1D numpy array of n-entries that represents the position of the
         sphere in n-dimensional space.

        self.mass: float
         A positive floating point number that represents the mass of the atom.

        self.name: str
         A string that represents the name of the atom. It can be changed at
         any time.

        self.radius: float
         A positive floating point number that represents the radius of the
         spherical atom.
    """

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # Public Interface
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # //////////////////////////////////////////////////////////////////////////
    # Properties
    # //////////////////////////////////////////////////////////////////////////

    @property
    def aname(self) -> str:
        """
            Returns the string representation of the name of the atom.

            :return: Returns a copy of the name of the atom.
        """
        return copy.deepcopy(self.__aname)

    @aname.setter
    def aname(self, aname: Any) -> None:
        """
            Sets the name of the atom, that is the string representation of the
            given object.
        """

        # Must not be None or an empty string.
        if aname is None or f"{aname}".strip() == "":
            raise ValueError(
                f"To set the name of the atom, the string representation of "
                f"the object must not be an empty string, or the string must "
                f"not be 'None'"
                f"{'; requested atom name is None' if aname is None else ''}."
            )

        # Set the name.
        self.__aname = f"{aname}"

    # ------------------------------------------------------------------------ #

    @property
    def atype(self) -> str:
        """
            Returns the string representation of the type of the atom.

            :return: Returns a copy of the type of the atom.
        """
        return copy.deepcopy(self.__atype)

    @atype.setter
    def atype(self, atype: Any) -> None:
        """
            Sets the type of the atom, that is the string representation of the
            given object.
        """

        # Must not be None or an empty string.
        if atype is None or f"{atype}".strip() == "":
            raise ValueError(
                f"To set the type of the atom, the string representation of "
                f"the object must not be an empty string, or the string must "
                f"not be 'None'"
                f"{'; requested atom name is None' if atype is None else ''}."
            )

        # Set the name.
        self.__atype = f"{atype}"

    # ------------------------------------------------------------------------ #

    @property
    def coordinates(self) -> np.ndarray:
        """
            Returns a copy of the coordinates of the atom.

            :return: Returns a copy of the coordinates of the atom.
        """
        return copy.deepcopy(self.__coordinates)

    @coordinates.setter
    def coordinates(self, coordinates: np.ndarray) -> None:
        """
            Sets the coordinates of the atom.
        """

        # Validate the coordinates.
        if not isinstance(coordinates, np.ndarray):
            raise TypeError(
                f"The coordinates must be a numpy array of 3 entries. Type: "
                f"{type(coordinates)}."
            )

        elif len(coordinates) != 3:
            raise ValueError(
                f"The coordinates must be a numpy array of 3 entries. Length: "
                f"{len(coordinates)}."
            )

        elif not all(map(lambda x: isinstance(x, float), coordinates)):
            raise TypeError(
                f"The coordinates must be a numpy array of 3 entries of type "
                f"numpy float. Entry types: {[type(x) for x in coordinates]}."
            )

        # Set the coordinates.
        self.__coordinates = coordinates

    # ------------------------------------------------------------------------ #

    @property
    def dimensions(self) -> int:
        """
            Returns the number of coordinates needed to completely describe the
            position of an atom in space.

            :return: The number of coordinates needed to completely describe the
             position of an atom in space.
        """
        return len(self.coordinates)

    # ------------------------------------------------------------------------ #

    @property
    def mass(self) -> float:
        """
            Returns the floating point number that represents the mass of the
            atom.

            :return: Returns the floating point number that represents the mass
             of the atom.
        """
        return self.__mass

    @mass.setter
    def mass(self, mass: float) -> None:
        """
            Sets the mass of the atom.

            :param mass: The floating point number that represents the mass of
             the atom.
        """

        # Validate the mass.
        if not isinstance(mass, float) or mass <= 0.0:
            raise ValueError(
                f"The requested mass must be a floating point number greater "
                f"than zero. Type: {type(mass)}, value: {mass}."
            )

        self.__mass = mass

    # ------------------------------------------------------------------------ #

    @property
    def radius(self) -> float:
        """
            Returns the floating point number that represents the radius of the
            atom.

            :return: Returns the floating point number that represents the
             radius of the atom.
        """
        return self.__radius

    @radius.setter
    def radius(self, radius: float) -> None:
        """
            Sets the radius of the atom.

            :param radius: The floating point number that represents the
             radius of the atom.
        """

        # Validate the radius
        if not isinstance(radius, float) or radius <= 0.0:
            raise ValueError(
                f"The requested radius must be a floating point number greater "
                f"than zero. Type: {type(radius)}, value: {radius}."
            )

        self.__radius = radius

    # //////////////////////////////////////////////////////////////////////////
    # Methods
    # //////////////////////////////////////////////////////////////////////////

    # --------------------------------------------------------------------------
    # Set Methods
    # --------------------------------------------------------------------------

    def set_from_elements(self) -> None:
        """
            Attempts to set the mass and radius parameters from the parameters
            in the periodic table by using the atom type. If the name is not in
            the periodic table it leaves the values untouched.

            :raises ElementWarning: If the user attempts to setup an atom from
             the periodic table that does not exist.
        """

        # Try to get the element.
        try:
            element = mendeleev.element(self.atype)
        except NoResultFound:
            warn(
                f"Setting up an atom's mass and radius according to the "
                f"periodic table does not work since the atom type, "
                f"{self.atype} does not exist. The request will be ignored."
            )
            return

        # If the mass and/or radius do not exist.
        if element.vdw_radius is None or element.mass is None:
            warn(
                f"Setting up an atom's mass and radius according to the "
                f"periodic table does not work since the atom type, "
                f"{self.atype} does not have a valid mass ({element.mass}) "
                f"or van der Waal radius ({element.vdw_radius}). The request "
                f"will be ignored."
            )
            return

        # Setup the values.
        self.radius = element.mass
        self.radius = element.vdw_radius / 100.0

    # //////////////////////////////////////////////////////////////////////////
    # Constructor
    # //////////////////////////////////////////////////////////////////////////

    def __init__(
        self, radius: float, mass: float, coordinates: np.ndarray,
        atype: str = "H", aname: str = "<>"
    ):
        """
            Constructs a new instance of an atom. Once created, the mass and the
            radius of the atom cannot be changed.

            :param radius: A positive floating point number that represents the
             radius of the spherical atom.

            :param mass: A positive floating point number that represents the
             mass of the atom.

            :param coordinates: A 1D numpy array of n-entries that represents
             the position of the sphere in n-dimensional space.

            :param atype: The type of the atom, can be set at any point.

            :param aname: The type of the atom, can be set at any point.
        """

        # Set the atom.
        self.aname = f"{aname}"
        self.atype = f"{atype}"

        # Set the other parameters.
        self.coordinates = coordinates
        self.mass = mass
        self.radius = radius

    # //////////////////////////////////////////////////////////////////////////
    # Dunder Methods
    # //////////////////////////////////////////////////////////////////////////

    def __len__(self):
        """
            Returns the number of coordinates needed to completely describe the
            position of an atom in space.

            :return: The number of coordinates needed to completely describe the
             position of an atom in space.
        """
        return self.dimensions

    def __repr__(self) -> str:
        """
            Returns a string with a quick representation of the atom, i.e.,
            the name of the atom, the coordinates, the radius and mass; in that
            specific order.

            :return: A string with a quick representation of the atom.
        """

        # Get the coordinates string.
        crds = self.coordinates
        crds = [tuple(['+' if x >= 0 else '-', abs(x)]) for x in crds]
        crds = "(" + ",".join([f"{x[0]}{x[1]:.7e}" for x in crds]) + ")"

        # Mass and radius string.
        mass = f"{self.mass:.7e}"
        radius = f"{self.radius:.7e}"

        # Set the values of the string.
        string = [self.aname, crds, radius, mass]

        return "    ".join(string)

    def __str__(self) -> str:
        """
            Returns a string with a detailed representation of the atom.

            :return: A string with a detailed representation of the atom.
        """

        # Get the coordinates string.
        crds = self.coordinates
        crds = [tuple(['+' if x >= 0 else '-', abs(x)]) for x in crds]
        crds = "(" + ", ".join([f"{x[0]}{x[1]:.7e}" for x in crds]) + ")"

        # Mass and radius string.
        mass = f"{self.mass:.7e}"
        radius = f"{self.radius:.7e}"

        # Set the values of the string.
        string = [
            f"Name: {self.aname}",
            f"Coordinates: {crds} \u212B",
            f"Radius: {radius} \u212B",
            f"Mass: {mass} AMU"
        ]

        return "\n".join(string)
