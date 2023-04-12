# Atom

## Description

This is the fundamental unit of the molecule. An atom is the basic unit that a
molecule contains. It is a sphere with a radius, a mass, a coordinate that 
determines its location in 3-dimensional space, the atom type and the atom name.

## Attributes

An atom is a solid non-deformable sphere that has a radius, a mass, the atom 
type, the atom name and a coordinate that determines its location in
3-dimensional space.

   - `self.radius`: A floating point number that represents the physical 
     **radius** of the sphere in angstrom; `1.0` by default. This is a mandatory
     parameter.
   - `self.mass`: A floating point number that represents the physical **mass**
     of the sphere in atomic mass units (AMU); `1.0` by default. This is a 
     mandatory parameter.
   - `self.coordinate`: A collection of 3 floating point numbers that describes
     the position of the center of the sphere in 3-dimensional space.
   - `self.atype`: A string that describes the type of the atom.
   - `self.aname`: A string that gives a custom name to the atom.

## Methods/Functions