# tsslope-pump 


## Description
Python ML models trained using [PyTorch](https://pytorch.org/) and/or its extensions, such as [GPyTorch](https://gpytorch.ai/), are incporporated as constraints in with algebraic optimization problems specified using [JuMP](https://jump.dev/) in Julia. `tsslope-pump-py` contains the code that infers the ML model in Python and computes first- and second-order derivatives of the ML output with respect to a subset of the optimization variables (inputs to the ML model). `tsslope_lib_jl` contains the Julia code needed for (i) software interoperability between Python and Julia and (ii) integrating the ML model as constraints in an existing JuMP optimization model. 


Directories:
  - tsslope-pump-py: library that contains functions to load the ML model and some parameters
  - tsslope-pump-jl: contains functions that implement the TS constraints in julia
  - driver: contains test code


## Installation

Runthe installation script:
  
    ./install.sh 

## Testing

After installation, set all environment variables that are needed in the file "test/load_env_values.jl"    
Then run:
   
    julia driver/driver.jl

## Usage

In order to use this package, after installation, you must add the following to the preamble of the Julia code
```
include(string(path_to_tsslope, "/init.jl"))
```
where `path_to_tsslope` specifies path to this directory (tsslope/tsslope-pump).

For an example of how to load an existing JuMP optimization model and adding the TS ML-based constraints to it, see `test/test.jl`
