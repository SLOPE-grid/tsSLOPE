include("load_config.jl")

using Pkg;
Pkg.activate((path_to_exajugo))
push!(LOAD_PATH, string(path_to_exajugo, "/modules"))

include("exajugo_call.jl") 

using PyCall
pushfirst!(pyimport("sys")."path", path_to_tsslope)
tsslope_lib = pyimport("tsslope-pump-py")

print("Reading instance from "*case_path*" ... ")
psd = SCACOPFdata(case_path)

jl_lib = string(path_to_tsslope,"/tsslope-pump-jl")
include(string(jl_lib,"/tsi_constraints.jl"))

test_problem = case_path
# test_problem = Texas_case_path

if test_problem == Texas_case_path
    gen_type = gen_type_7k_file
else
    gen_type = nothing
end

print("Reading instance from "*test_problem*" ... ")
psd = SCACOPFdata(test_problem)

save_spect_info = false

save_Hess = false

active_gen_only = true

model_type = "CNF"
# model_type = "CNN_Soft"
# model_type = "None"

max_iter = 200

if model_type == "CNF"
    tau = 0.0
else
    tau = 0.5
end

t0 = time()

Hess_approx = false

gamma = 0.
r = 6

num_iters = Vector{Int}()
total_times = Vector{Float64}()

Mels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# Mels = [2]

gamma_update = false

approx_type = "Sparse"
approx_type = "Limited"
# approx_type = "Full"

if Hess_approx == false
    surrogate_path = nothing
elseif model_type == "CNF"
    surrogate_path = CNF_model_final_path
elseif model_type == "CNN_Soft"
    surrogate_path = CNN_Soft_UQ_1_model_path
else
    surrogate_path = nothing
end

surrogate, data, TSI = tsslope_lib.load_model(CNF_model_final_path, LLNL_data_record, model_type, active_gen_only = active_gen_only)


if approx_type == "Sparse"
    _, _, _, _, _, _, _, _, SR1_hist =
        TSACOPF(
            test_problem,
            case_sol_path,
            pf_limit_file,
            surrogate,
            psd,
            tau,
            spect_info = save_spect_info,
            save_Hess = save_Hess,
            max_iter = max_iter,
            Hess_approx = Hess_approx,
            gamma = gamma,
            approx_type = "Limited",
            r = r,
            gamma_update = gamma_update
        )

    for Mel in Mels
        print("Mel = $Mel \n")
        num_iter, total_time, _, _, _, _, _, _ =
            TSACOPF(
                test_problem,
                case_sol_path,
                pf_limit_file,
                surrogate,
                psd,
                tau,
                spect_info = save_spect_info,
                save_Hess = save_Hess,
                max_iter = max_iter,
                Hess_approx = Hess_approx,
                gamma = gamma,
                approx_type = approx_type,
                r = r,
                gamma_update = gamma_update, 
                SR1_hist = SR1_hist,
                Mel = Mel
            )
        push!(num_iters, num_iter)
        push!(total_times, total_time)
    end

else
    num_iter, total_time, base_cost,
    Surr_Feasibility_margin, ter_status, norm_grad, pg =
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
            gamma_update = gamma_update
        )
    push!(num_iters, num_iter)
    push!(total_times, total_time)
end


if approx_type == "Sparse"
    for i = 1:length(num_iters)
        println("Mel = $(Mels[i]), num_iter = $(num_iters[i]), total_time = $(total_times[i])")
    end
else
    for i = 1:length(num_iters)
        println("num_iter = $(num_iters[i]), total_time = $(total_times[i])")
    end
end
println("Total time elapsed $(time() - t0)")

