****************************************************************************************************
2D HEAT DIFFUSION EXPLICIT FEM SOLVER
****************************************************************************************************

The purpose of the code is to solve the heat diffusion equation on an unstructured mesh using
the finite element method and explicit time stepping.

## Compilation

To compile the code you have to:

mkdir build
cd build
cmake ..
make

## Runnning

To run the code you have to:

cd test
../build/2d_Unsteady (.settings.blah.in)

## Visualization

To visualize the result you have to:

module load GRAPHICS paraview
module switch paraview/5.7.0-RC3 paraview/5.6.0
paraview

open Cylinder.pvtu
