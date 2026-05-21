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

test_problem = ACTIVSg500_case_path
#test_problem = Texas_case_path
#test_problem = Texas_old_case_path

if test_problem == ACTIVSg500_case_path
    gen_type = nothing
else
    gen_type = gen_type_7k_file
end

println("Reading instance from "*test_problem*" ... ")
psd = SCACOPFdata(test_problem)

active_gen_only = true

# model_type = "CNF"
# model_type = "DKL_ReLU"
#model_type = "DKL_Soft"
# model_type = "CNN_Soft"
model_type = "None"

max_iter = 300

if model_type == "CNF"
    tau = 0.0
else
    tau = 0.5
end

t0 = time()

Hess_approx = true

# if Hess_approx is false the following parameters are not used
gamma = 0.
r = 6

diag_pattern = true

gamma_update = false

# approx_type = "Sparse"
approx_type = "Limited"
# approx_type = "Full"

surrogate_path = nothing
model_type = "None"
approx_type = nothing
if test_problem == Texas_case_path || test_problem == Texas_old_case_path
    if model_type == "DKL_ReLU"
        surrogate_path = DKL_ReLU_model_path
    elseif model_type == "DKL_Soft"
        surrogate_path = DKL_Soft_model_path
    elseif model_type == "CNN_Soft"
        surrogate_path = CNN_Soft_7k_model_path
    end  
    data_path = DKL_train_xy_path
elseif test_problem == ACTIVSg500_case_path
    if model_type == "CNF"
        surrogate_path = CNF_model_final_path
    elseif model_type == "CNN_Soft"
        surrogate_path = CNN_Soft_UQ_1_model_path
    end
    data_path = LLNL_data_record
end   

if Hess_approx == false
    approx_type = nothing
end

println("The data drive: $data_path \n")

println("The surrogate: $model_type and the path: $surrogate_path")

surrogate = tsslope_lib.load_model(surrogate_path, data_path, model_type, active_gen_only = active_gen_only)

num_iter, total_time, base_cost, Surr_Feasibility_margin, ter_status, norm_grad, pg =
        TSACOPF(
            test_problem,
            case_sol_path,
            pf_limit_file,
            surrogate,
            psd,
            tau,
            max_iter = max_iter,
            Hess_approx = Hess_approx,
            gamma = gamma,
            approx_type = approx_type,
            r = r,
            gamma_update = gamma_update,
            gen_type = gen_type, 
            diag_pattern = diag_pattern
        )
