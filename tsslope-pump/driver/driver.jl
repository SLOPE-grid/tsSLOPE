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

using DataFrames, CSV
using Random

function perturb_percent(x::AbstractVector; eps=0.01, rng=Random.GLOBAL_RNG)
    return x .* (1 .+ eps .* (2 .* rand(rng, length(x)) .- 1))
end


print("Reading instance from "*case_path*" ... ")
psd = SCACOPFdata(case_path)

tau = 0.5

max_iterations = 50

numb_runs = 100

# The perterbation of the pl, and ql
perc = 0.15

# the number of iteration for the TS-ACOPF is only iter_slack percentage from the ACOPF number of iterations
iter_slack = .1

active_gen_only = true

D = Dict{String, Vector{String}}()
D["None"] = [""]
# D["CNN_GELU"] = [
#                 # CNN_GELU_all_1_model_path, CNN_GELU_all_2_model_path, CNN_GELU_all_3_model_path, CNN_GELU_all_4_model_path, CNN_GELU_all_5_model_path,
#                 # CNN_GELU_AC_1_model_path, CNN_GELU_AC_2_model_path, CNN_GELU_AC_3_model_path, CNN_GELU_AC_4_model_path, CNN_GELU_AC_5_model_path,
#                 # CNN_GELU_UQ_1_model_path, CNN_GELU_UQ_2_model_path, CNN_GELU_UQ_3_model_path, CNN_GELU_UQ_4_model_path, CNN_GELU_UQ_5_model_path
#                 # CNN_GELU_UQ_1_model_path, CNN_GELU_UQ_4_model_path
#                 ]
D["CNN_Sig"]  = [
                # CNN_Sig_all_1_model_path , CNN_Sig_all_2_model_path , CNN_Sig_all_3_model_path , CNN_Sig_all_4_model_path , CNN_Sig_all_5_model_path,
                CNN_Sig_all_3_model_path , CNN_Sig_all_4_model_path ,
                CNN_Sig_AC_1_model_path , CNN_Sig_AC_2_model_path , CNN_Sig_AC_3_model_path , CNN_Sig_AC_4_model_path , CNN_Sig_AC_5_model_path,
                # CNN_Sig_UQ_1_model_path , CNN_Sig_UQ_2_model_path , CNN_Sig_UQ_3_model_path , CNN_Sig_UQ_4_model_path , CNN_Sig_UQ_5_model_path
                ]
# D["CNN_SiLU"] = [
#                 # CNN_SiLU_all_1_model_path, CNN_SiLU_all_2_model_path, CNN_SiLU_all_3_model_path, CNN_SiLU_all_4_model_path, CNN_SiLU_all_5_model_path,
#                 # CNN_SiLU_AC_1_model_path, CNN_SiLU_AC_2_model_path, CNN_SiLU_AC_3_model_path, CNN_SiLU_AC_4_model_path, CNN_SiLU_AC_5_model_path,
#                 # CNN_SiLU_UQ_1_model_path, CNN_SiLU_UQ_2_model_path, CNN_SiLU_UQ_3_model_path, CNN_SiLU_UQ_4_model_path, CNN_SiLU_UQ_5_model_path
#                 ]
D["CNN_Soft"] = [
                # CNN_Soft_all_1_model_path, CNN_Soft_all_2_model_path, CNN_Soft_all_3_model_path, CNN_Soft_all_4_model_path, CNN_Soft_all_5_model_path,
                CNN_Soft_all_2_model_path, 
                CNN_Soft_AC_1_model_path, CNN_Soft_AC_2_model_path, CNN_Soft_AC_3_model_path, CNN_Soft_AC_4_model_path, CNN_Soft_AC_5_model_path,
                # CNN_Soft_UQ_1_model_path, CNN_Soft_UQ_2_model_path, CNN_Soft_UQ_3_model_path, CNN_Soft_UQ_4_model_path, CNN_Soft_UQ_5_model_path
                CNN_Soft_UQ_4_model_path
                ] 
D["CNN_Tanh"] = [
                # CNN_Tanh_all_1_model_path, CNN_Tanh_all_2_model_path, CNN_Tanh_all_3_model_path, CNN_Tanh_all_4_model_path, CNN_Tanh_all_5_model_path,
                # CNN_Tanh_AC_1_model_path, CNN_Tanh_AC_2_model_path, CNN_Tanh_AC_3_model_path, CNN_Tanh_AC_4_model_path, CNN_Tanh_AC_5_model_path,
                CNN_Tanh_AC_2_model_path,
                # CNN_Tanh_UQ_1_model_path, CNN_Tanh_UQ_2_model_path, CNN_Tanh_UQ_3_model_path, CNN_Tanh_UQ_4_model_path, CNN_Tanh_UQ_5_model_path
                ] 

# D["CNN_GELU"] = [CNN_GELU_1_model_path, CNN_GELU_2_model_path, CNN_GELU_3_model_path, CNN_GELU_4_model_path, CNN_GELU_5_model_path]
# D["CNN_Sig"]  = [CNN_Sig_1_model_path , CNN_Sig_2_model_path , CNN_Sig_3_model_path , CNN_Sig_4_model_path , CNN_Sig_5_model_path ]
# D["CNN_SiLU"] = [CNN_SiLU_1_model_path, CNN_SiLU_2_model_path, CNN_SiLU_3_model_path, CNN_SiLU_4_model_path, CNN_SiLU_5_model_path, CNN_model_path]
# D["CNN_Soft"] = [CNN_Soft_1_model_path, CNN_Soft_2_model_path, CNN_Soft_3_model_path, CNN_Soft_4_model_path, CNN_Soft_5_model_path] 
# D["CNN_Tanh"] = [CNN_Tanh_1_model_path, CNN_Tanh_2_model_path, CNN_Tanh_3_model_path, CNN_Tanh_4_model_path, CNN_Tanh_5_model_path] 

results_perb = Dict{Int, Dict{String, Dict{Int, NamedTuple}}}()
t0 = time()
for j =1:numb_runs
    psd_temp = psd

    if j == 1
        perb = false
    else
        perb = true
        psd_temp.N.Pd = perturb_percent(psd.N.Pd, eps=perc)
        psd_temp.N.Qd = perturb_percent(psd.N.Qd, eps=perc)
    end


    max_iter = max_iterations

    results = Dict{String, Dict{Int, NamedTuple}}()
    for (model_type, model_paths) in D
        # temporary dict for this model
        tmp = Dict{Int, NamedTuple}()

        for (i, path) in enumerate(model_paths)
            println("Model: $model_type,  Perturbation $j,  Run $i")

            if model_type == "None"
                surrogate = Dict(
                    "model_type" => nothing,
                    )
            else
                surrogate, data, TSI = 
                    tsslope_lib.load_model(path, LLNL_data_record, model_type, active_gen_only = active_gen_only,)
            end           

            num_iter, total_time, base_cost,
            Surr_Feasibility_margin, termination_status, norm_grad, pg, hess_analy_temp =
                TSACOPF(
                    case_path,
                    case_sol_path,
                    pf_limit_file,
                    surrogate,
                    psd_temp,
                    tau,
                    spect_info = false,
                    save_Hess = false, 
                    max_iter = max_iter,
                )

            if model_type == "None"
                max_iter = max(ceil(Int, (1 + iter_slack) * num_iter), 50)
            end
                
            println("num_iter: $num_iter,\n total_time: $total_time,\n base_cost: $base_cost,\n termination_status: $termination_status,\n Surr_Feasibility_margin: $Surr_Feasibility_margin")
            # store results for iteration i
            tmp[i] = (
                num_iter = num_iter,
                total_time = total_time,
                base_cost = base_cost,
                Surr_Feasibility_margin = Surr_Feasibility_margin,
                norm_grad = norm_grad,
                termination_status = termination_status,
                perb = perb,
                model_path = path
            )
        end
        # store temp dict under the same string key
        results[model_type] = tmp
    end
    results_perb[j] = results
end
et = time() - t0
println("Total time elapsed $et")

cvs_filename = "results_all_model_new_data_refine_perb_$(perc)_slack_$(iter_slack)_runs_$(numb_runs).csv"

results_dir = "./perb_results"

if !ispath(results_dir)
    mkpath(results_dir)
end

rows = NamedTuple[]

for (j, results) in results_perb
    for (model_name, iter_dict) in results
        for (i, r) in iter_dict
            push!(rows, (
                model = model_name,
                model_number = i,
                run_numb = j,
                num_iter = r.num_iter,
                total_time = r.total_time,
                base_cost = r.base_cost,
                norm_grad = r.norm_grad,
                Surr_Feasibility_margin = r.Surr_Feasibility_margin,
                perb = r.perb,
                termination_status = r.termination_status,
                model_path = r.model_path,
            ))
        end
    end
end

df = DataFrame(rows)
CSV.write(results_dir * "/" * cvs_filename, df)


