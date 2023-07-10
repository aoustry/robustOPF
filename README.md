# robustOPF
Solving the Adjustable Robust AC OPF problem with an adaptive discretization algorithm

# Programming language and dependencies

This code is developed in Julia. The required packages are
- JLD
- ExaPF
- LinearAlgebra
- Distributions
- MathOptInterface

# Instance data

The data for the test cases should be dowloaded at github.com/power-grid-lib/pglib-opf. 

In the main_ar_acopf.jl file, the path to this dowloaded folder should be contained in the variable ``pglibfolder''.


