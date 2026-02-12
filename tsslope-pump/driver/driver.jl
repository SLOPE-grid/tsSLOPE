
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
model_type = "CNF_min"

if model_type == "CNN"
    surrogate, data, TSI = tsslope_lib.load_model(CNN_model_path, LLNL_data_record, model_type)
elseif model_type == "CNN_Grad_UQ"
    surrogate, data, TSI = tsslope_lib.load_model(CNN_model_path, LLNL_data_record, model_type)
elseif model_type == "CNF_min"
    model_type = "CNF"
    surrogate, data, TSI = tsslope_lib.load_model(CNF_model_min_path, LLNL_data_record, model_type)
elseif model_type == "CNF_final"
    model_type = "CNF"
    surrogate, data, TSI = tsslope_lib.load_model(CNF_model_final_path, LLNL_data_record, model_type)
elseif model_type == "DSPP"
    surrogate, data, TSI = tsslope_lib.load_model(model_path, LLNL_data_record, model_type)
else
    surrogate = Dict(
        "model_type" => nothing,
    )
    println("ACOPF will run without a surrogate.")
end

TSACOPF(case_path, case_sol_path, pf_limit_file, surrogate);


