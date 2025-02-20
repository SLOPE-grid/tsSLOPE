# tsSLOPE


## Description

This repository provides interfaces and glue code to ensure software integration between machine learning models (MLs) in Python, algebraic modelling frameworks in Julia, and HPC optimization solvers in C++. This specific approach is required by ongoing research for solving optimal power flow problems with transient stability (TS) constraints informed by ML surrogates that is taking place under the [SLOPE-Grid](https://slope-grid.github.io/) project.

### Python MLs within JuMP optimization models (tsslope-pump)
We integrate ML models, which are trained using [PyTorch](https://pytorch.org/) and/or its extensions, such as [GPyTorch](https://gpytorch.ai/), with algebraic optimization problems specified using [JuMP](https://jump.dev/) in Julia. `tsslope_lib` contains the code that infers the ML model in Python and computes first- and second-order derivatives of the ML output with respect to a subset of the optimization variables (inputs to the ML model). `tsslope_lib_jl` contains the Julia code needed for (i) software interoperability between Python and Julia and (ii) integrating the ML model as constraints in an existing JuMP optimization model. 

### ML training using simulation data (tsslope-siml)
We provide Python drivers that generate training points for ML by using TS simulators. These drivers are based on existing software libraries, such as [libEnsemble](https://github.com/Libensemble/libensemble) for handling a large number of ensembles of simulations (that is, TS simulations) on massively parallel HPC platforms. The simulations can be performed by current TS libraries, such as PowerWorld or ANDES. Finally, we provide Python scripts that train a ML model to learn  TS measures, such as TS indexes (which are integrated in optimization models using  `tsslope-pump` described above). This work is in progress and code will be made available later.

### Optimization interfaces (tsslope-opt)
The optimization problems obtained  tsslope-pump is used to incorporate TS ML surrogates in JuMP optimal power flow models are of significant sizes and requires memory-distributed parallel computing. 

Directories:
  - tsslope_lib: library that contains functions to load the ML model and some parameters
  - tsslope_lib_jl: contains functions that implement the tsi contraints in julia
  - test: contains test code

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://lc.llnl.gov/gitlab/santiago/tsslope.git
git branch -M main
git push -uf origin main
```

## Installation

Runthe installation script:
  
    ./install.sh 

## Testing

After installation, set all environment variables that are needed in the file "test/load_env_values.jl"    
Then run:
   
    julia test/test.jl

## Usage

In order to use this package, after installation, you must add the following to the julia code before the use of any constrainr:

include(string(path_to_tsslope, "/init.jl"))
  
where: 
 - path_to_tsslope: path to the repo TsSLOPE

For an example of how to load the mode and adding the TSI constraints, see test/test.jl

## Acknowledgments

This code has been supported by DOE ASCR under the SciDAC-OE partnership project [SLOPE-Grid](https://slope-grid.github.io/).
