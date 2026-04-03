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

using DataFrames, CSV, HDF5
using Random

function perturb_percent(x::AbstractVector; eps=0.01, rng=Random.GLOBAL_RNG)
    return x .* (1 .+ eps .* (2 .* rand(rng, length(x)) .- 1))
end

jobid = parse(Int, get(ENV, "SLURM_JOB_ID", "0"))
folder_st = parse(Int, get(ENV, "folder_start", "0"))

print("Reading instance from "*case_path*" ... ")
psd = SCACOPFdata(case_path)

tau = 0.5

max_iterations = 200

numb_runs = 100

# The perterbation of the pl, and ql
perc = 0.15  # or 0.67082 for Emil's CNF

# the number of iteration for the TS-ACOPF is only iter_slack percentage from the ACOPF number of iterations
iter_slack = .1

active_gen_only = true

D = Dict{String, Vector{String}}()
D["None"] = [""]

D["CNN_Sig"]  = [
                CNN_Sig_all_3_model_path , CNN_Sig_all_4_model_path ,
                CNN_Sig_AC_1_model_path 
                ]
                
D["CNN_Soft"] = [
                CNN_Soft_all_2_model_path,
                CNN_Soft_AC_1_model_path, 
                CNN_Soft_UQ_1_model_path, CNN_Soft_UQ_4_model_path
                ]                
                
D["CNN_Tanh"] = [
                CNN_Tanh_AC_2_model_path,
                ] 

Dname = Dict{String, Vector{String}}()
Dname["None"] = ["None"]

Dname["CNN_Sig"]  = [
                "CNN_Sig_all_3" , "CNN_Sig_all_4" ,
                "CNN_Sig_AC_1" , 
                ]
                
Dname["CNN_Soft"] = [
                "CNN_Soft_all_2", 
                "CNN_Soft_AC_1", 
                "CNN_Soft_UQ_1", "CNN_Soft_UQ_4"
                ]

Dname["CNN_Tanh"] = [
                "CNN_Tanh_AC_2",
                ] 

results_perb = Dict{Int, Dict{String, Dict{Int, NamedTuple}}}()
t0 = time()
rng = MersenneTwister(jobid)
for j =1:numb_runs
    psd_temp = deepcopy(psd)

    if j == 1
        perb = false
    else
        perb = true
        psd_temp.loads[!, :PL] = perturb_percent(psd.loads[!, :PL], eps=perc, rng=rng)
        psd_temp.loads[!, :QL] = perturb_percent(psd.loads[!, :QL], eps=perc, rng=rng)
        
        if size(psd_temp.loads, 1) > 0
            BusLoad = indexin(psd_temp.loads[!,:I], psd_temp.N[!,:Bus])
            for l = 1:size(psd_temp.loads, 1)
                if psd_temp.loads[l,:STATUS] == 1
                    psd_temp.N[BusLoad[l],:Pd] = 0.0
                    psd_temp.N[BusLoad[l],:Qd] = 0.0
                end
            end
            for l = 1:size(psd_temp.loads, 1)
                if BusLoad[l] == nothing
                    error("bus ", psd_temp.loads[l,:I], " of load ", l, " not found.")
                end
                if psd_temp.loads[l,:STATUS] == 1
                    psd_temp.N[BusLoad[l],:Pd] += psd_temp.loads[l,:PL]/psd_temp.MVAbase
                    psd_temp.N[BusLoad[l],:Qd] += psd_temp.loads[l,:QL]/psd_temp.MVAbase
                end
            end
        end
    end

    max_iter = max_iterations

```
    results = Dict{String, Dict{Int, NamedTuple}}()
```
    for (model_type, model_paths) in D
        # temporary dict for this model
        tmp = Dict{Int, NamedTuple}()

        for (i, path) in enumerate(model_paths)
            model_name = Dname[model_type][i]
            println("Model: $model_type, Model name: $model_name, Perturbation $j,  Run $i")

            if model_type == "None"
                surrogate = Dict(
                    "model_type" => nothing,
                    )
            else
                surrogate, data, TSI = 
                    tsslope_lib.load_model(path, LLNL_data_record, model_type, active_gen_only = active_gen_only,)
            end           

            solution_dir = joinpath(case_sol_path, "$(model_name)", "folder_$(j+folder_st)")
            if !ispath(solution_dir)
                mkpath(solution_dir)
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
                
            println("num_iter: $num_iter,\n total_time: $total_time,\n base_cost: $base_cost,\n termination_status: $termination_status,\n Surr_Feasibility_margin: $Surr_Feasibility_margin")
```
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
```
        end
```
        # store temp dict under the same string key
        results[model_type] = tmp
```
    end
    results_perb[j] = results
end
et = time() - t0
println("Total time elapsed $et")

```
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
```
