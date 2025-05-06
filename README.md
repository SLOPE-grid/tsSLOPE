# tsSLOPE


## Description

This repository provides interfaces and glue code to ensure software integration between machine learning models (MLs) in Python, algebraic modelling frameworks in Julia, and HPC optimization solvers in C++. This specific approach is required by ongoing research for solving optimal power flow problems with transient stability (TS) constraints informed by ML surrogates that is taking place under the [SLOPE-Grid](https://slope-grid.github.io/) project.

### tsslope-pump: Python MLs within JuMP optimization models 
tsslope-pump integrates Python ML models, which are trained using [PyTorch](https://pytorch.org/) and/or its extensions, such as [GPyTorch](https://gpytorch.ai/), with algebraic optimization problems specified using [JuMP](https://jump.dev/) in Julia. 

### tsslope-siml: ML training using simulation data
We provide Python drivers that generate training points for ML by using TS simulators. These drivers are based on existing software libraries, such as [libEnsemble](https://github.com/Libensemble/libensemble) for handling a large number of ensembles of simulations (that is, TS simulations) on massively parallel HPC platforms. The simulations can be performed by current TS libraries, such as PowerWorld or ANDES. Finally, we provide Python scripts that train a ML model to learn  TS measures, such as TS indexes (which are integrated in optimization models using  `tsslope-pump` described above). This work is in progress and code will be made available later.

## Installation and usage

Please see to README files in `tsslope-pump` and `tsslope-siml` directories for how to install, test, and use the two libraries above. 

## Issues

Users are highly encouraged to report issues by opening new github issues. 

## Contributors

tsSLOPE is developed by Claudio Santiago (santiago10@llnl.gov), Nai-Yuan Chiang (chiang7@llnl.gov), and Cosmin G. Petra (petra1@llnl.gov). 

## Acknowledgments

This code has been supported by DOE ASCR under the SciDAC-OE partnership project [SLOPE-Grid](https://slope-grid.github.io/).

## Copyright

tsSLOPE is free software distributed under the terms of the BSD 3-clause license. Users can modify and/or redistribute tsSLOPE under the terms of the BSD 3-clause license. See [COPYRIGHT](COPYRIGHT), [LICENSE](LICENSE), and [NOTICE](NOTICE) files for complete copyright and license information. All new contributions must be made under the terms of the BSD 3-clause license. 

LLNL-CODE-2005624


