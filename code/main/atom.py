# ##############################################################################
# Imports
# ##############################################################################

# General.
import mendeleev

from numpy import append as nappend, array, delete as ndelete, ndarray, zeros
from sqlalchemy.orm.exc import NoResultFound
from warnings import warn

# User defined.
import code.utilities.utilities_strings as sutils
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
        # Validate it's a string.
        vparameters.is_string(aname, strip=True, empty=True)

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
        vparameters.is_string(atype, strip=True, empty=True)

        self.__atype = atype.strip()

    # ------------------------------------------------------------------------ #

    @property
    def coordinates(self) -> ndarray:
        """
            Returns a copy of the coordinates of the atom.

            :return: Returns a copy of the coordinates of the atom.
        """
        return self.__coordinates[0]

    @coordinates.setter
    def coordinates(self, coordinates: ndarray) -> None:
        """
            Sets the coordinates of the atom.

            :param coordinates: The new coordinates of the atom. Must be a
             1-dimensional numpy array of floats.
        """

        # Gurantees same dimensions and type.
        coords = array(coordinates, dtype=float)
        self.__coordinates = nappend(self.__coordinates, [coords], axis=0)
        self.__coordinates = ndelete(self.__coordinates, 0, axis=0)

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
        vparameters.is_positive(mass, include=False)

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
        vparameters.is_positive(radius, include=False)

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
        self.mass = element.mass
        self.radius = element.vdw_radius / 100.0

    # //////////////////////////////////////////////////////////////////////////
    # Constructor
    # //////////////////////////////////////////////////////////////////////////

    def __init__(
        self, radius: float, mass: float, coordinates: ndarray,
        atype: str = None, aname: str = None
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

        # Set the atom names and type.
        self.aname = "---" if aname is None else aname
        self.atype = "---" if atype is None else atype

        # Set the other parameters.
        self.mass = mass
        self.radius = radius

        # Easy way of setting up the coordinates.
        self.__coordinates = zeros((1, len(coordinates)), dtype=float)
        self.coordinates = coordinates

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
        return "    ".join([
            self.aname,
            self.atype,
            f"{sutils.get_string_vector(self.coordinates)} \u212B",
            f"{self.radius:.7e} \u212B",
            f"{self.mass:.7e} AMU"
        ])

    def __str__(self) -> str:
        """
            Returns a string with a detailed representation of the atom.

            :return: A string with a detailed representation of the atom.
        """
        return "\n".join([
            f"Name: {self.aname}",
            f"Type: {self.atype}",
            f"Coordinates: {sutils.get_string_vector(self.coordinates)} \u212B",
            f"Radius: {self.radius:.7e} \u212B",
            f"Mass: {self.mass:.7e} AMU"
        ])
