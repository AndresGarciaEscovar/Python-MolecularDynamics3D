"""
    File that contains the atoms class.
"""

# ##############################################################################
# Imports
# ##############################################################################

# General.
import copy
import mendeleev
import numpy as np

from sqlalchemy.orm.exc import NoResultFound
from warnings import warn

# User defined.
import code.utilities.utilities_strings as ustrings
import code.validation.validation_parameters as vparameters

# ##############################################################################
# Classes
# ##############################################################################


class Atom:
    """
        Class that represents a spherical atom.

        Parameters:
        ___________
        self.aname: str
         A string that represents the name of the atom. It can be changed at
         any time.

        self.atype: str
         A string that represents the type of the atom. It can be changed at
         any time.

        self.coordinates: numpy.ndarray
         A 1D numpy array of n-entries that represents the position of the
         sphere in n-dimensional space.

        self.dimensions: int
         The number of dimensions, i.e., the length of the coordinates array.

        self.mass: float
         A positive floating point number that represents the mass of the atom.

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
        return self.__aname

    @aname.setter
    def aname(self, aname: str) -> None:
        """
            Sets the name of the atom, that is the string representation of the
            given object.

            :param aname: The new name of the atom.
        """
        # Validate it's a non-empty string.
        vparameters.is_string_empty(aname, empty=True)

        # Must be a string.
        self.__aname = aname.strip()

    # ------------------------------------------------------------------------ #

    @property
    def atype(self) -> str:
        """
            Returns the string representation of the type of the atom.

            :return: Returns a copy of the type of the atom.
        """
        return self.__atype

    @atype.setter
    def atype(self, atype: str) -> None:
        """
            Sets the type of the atom, that is the string representation of the
            given object.

            :param atype: The new type of the atom.
        """
        # Validate it's a string.
        vparameters.is_string_empty(atype, empty=True)

        self.__atype = atype.strip()

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

            :param coordinates: The new coordinates of the atom. Must be a
             1-dimensional numpy array of floats.
        """
        # Create the coordinates.
        self.__coordinates = np.array(coordinates, dtype=float)

        # Validate the coordinates.
        vparameters.is_length(self.__coordinates, 3)

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
        vparameters.is_positive(mass, zero=False)
        vparameters.is_negative(self.__mass, zero=True)

        self.__mass = float(mass)

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
        # Validate the radius.
        vparameters.is_positive(radius, zero=False)
        vparameters.is_negative(self.__radius, zero=True)

        self.__radius = float(radius)

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
        self.__mass = element.mass
        self.__radius = element.vdw_radius / 100.0

    # //////////////////////////////////////////////////////////////////////////
    # Constructor
    # //////////////////////////////////////////////////////////////////////////

    def __init__(
        self, aname: str, atype: str, radius: float, mass: float,
        coordinates: np.ndarray,
        
    ):
        """
            Constructs a new instance of an atom. Once created, the mass and the
            radius of the atom cannot be changed.

            :param aname: The name of the atom.

            :param atype: The type of the atom.
            
            :param radius: A positive floating point number that represents the
             radius of the spherical atom.

            :param mass: A positive floating point number that represents the
             mass of the atom.

            :param coordinates: A 1D numpy array of n-entries that represents
             the position of the sphere in n-dimensional space.
        """
        # Set the atom names and type.
        self.aname = aname
        self.atype = atype

        # Set the other parameters.
        self.__mass = 0.0
        self.mass = mass

        self.__radius = 0.0
        self.radius = radius

        # Easy way of setting up the coordinates.
        self.__coordinates = None
        self.coordinates = coordinates

        # Set the mass and radius from the periodic table.
        self.set_from_elements()

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
            the name of the atom, the type of the atom, the coordinates, the
            radius and mass; in that specific order.

            :return: A string with a quick representation of the atom.
        """
        # Auxiliary variables.
        get_string_vector = ustrings.get_string_vector

        return "    ".join([
            self.aname,
            self.atype,
            f"{get_string_vector(self.coordinates)} \u212B",
            f"{self.radius:.7e} \u212B",
            f"{self.mass:.7e} AMU"
        ])

    def __str__(self) -> str:
        """
            Returns a string with a detailed representation of the atom.

            :return: A string with a detailed representation of the atom.
        """
        # Auxiliary variables.
        get_string_vector = ustrings.get_string_vector

        return "\n".join([
            f"Name: {self.aname}",
            f"Type: {self.atype}",
            f"Coordinates: {get_string_vector(self.coordinates)} \u212B",
            f"Radius: {self.radius:.7e} \u212B",
            f"Mass: {self.mass:.7e} AMU"
        ])
