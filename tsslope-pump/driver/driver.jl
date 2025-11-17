
include("load_config.jl")

using Pkg;
Pkg.activate((path_to_exajugo))
push!(LOAD_PATH, string(path_to_exajugo, "/modules"))

include("exajugo_call.jl") 

using PyCall
pushfirst!(pyimport("sys")."path", path_to_tsslope)
tsslope_lib = pyimport("tsslope-pump-py")

jl_lib = string(path_to_tsslope,"/tsslope-pump-jl")
include(string(jl_lib,"/tsi_constraints.jl"))

# need a better way of doing this
model_type = "CNN"

display("Code is running")

CNNmodel, data, TSI = tsslope_lib.load_model(CNN_model_path, LLNL_data_record, model_type)

display("Model is loaded")

TSACOPF(case_path, case_sol_path, pf_limit_file, CNNmodel);

display("Code finished running")

# model_type = "DSPP"
# GPmodel, data, TSI = tsslope_lib.load_model(model_path, data_record, model_type)
# TSACOPF(case_path, case_sol_path, pf_limit_file, GPmodel);


