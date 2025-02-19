# tsSLOPE


## Description

This project implements the interface that reads a machine learning (ML) model trained in Python to be used in Julia to inform a JuMP optimization model. 

Directories:
  - tsslope_lib: library that contains functions to load the ML model and some parameters
  - tsslope_lib_jl: contains functions that implement the tsi contraints in julia
  - test: c ontaints test code

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
