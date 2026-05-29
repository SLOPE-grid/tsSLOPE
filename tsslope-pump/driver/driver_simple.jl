include("/p/lustre1/hiop/project/scidac/tsSLOPE/tsslope-pump/driver/load_config.jl")

using Pkg;
# Use the new env_julia project instead of exajugo
Pkg.activate("/p/lustre1/hiop/project/scidac/env_julia")
push!(LOAD_PATH, string(path_to_exajugo, "/modules"))

include("/p/lustre1/hiop/project/scidac/tsSLOPE/tsslope-pump/driver/exajugo_call.jl") 

using PyCall
pushfirst!(pyimport("sys")."path", path_to_tsslope)
tsslope_lib = pyimport("tsslope-pump-py")

jl_lib = string(path_to_tsslope,"/tsslope-pump-jl")
include(string(jl_lib,"/tsi_constraints.jl"))

test_problem = ACTIVSg500_case_path

println("Reading instance from "*test_problem*" ... ")
psd = SCACOPFdata(test_problem)

max_iter = 300

num_iter, total_time, base_cost, Surr_Feasibility_margin, ter_status, norm_grad, pg =
        TSACOPF(
            test_problem,
            case_sol_path,
            pf_limit_file,
            nothing,
            psd,
            nothing,
            max_iter = max_iter,
        )
