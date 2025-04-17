
include("load_env_values_ny.jl")
include("exajugo_call.jl") 

include(string(path_to_tsslope, "/init.jl"))

GPmodel, data, TSI = tsslope_lib.load_model(model_path, data_record)

TSACOPF(case_path, case_sol_path, pf_limit_file, GPmodel);




