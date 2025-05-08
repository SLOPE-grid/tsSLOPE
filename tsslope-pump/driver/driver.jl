
include("load_config.jl")

using Pkg;
Pkg.activate((path_to_exajugo))
#push!(LOAD_PATH, string(path_to_exajugo, "/modules"))
push!(LOAD_PATH, string("C:\\Users\\chiang7\\Project\\2025\\scidac\\exajugo", "\\modules"))

include("exajugo_call.jl") 

using PyCall
pushfirst!(pyimport("sys")."path", path_to_tsslope)
tsslope_lib = pyimport("tsslope-pump-py")

jl_lib = string(path_to_tsslope,"/tsslope-pump-jl") # linux
include(string(jl_lib,"/tsi_constraints.jl")) # linux

jl_lib = string(path_to_tsslope,"\\tsslope-pump-jl") # windowes
include(string(jl_lib,"\\tsi_constraints.jl")) # windowes

GPmodel, data, TSI = tsslope_lib.load_model(model_path, data_record);

TSACOPF(case_path, case_sol_path, pf_limit_file, GPmodel);


