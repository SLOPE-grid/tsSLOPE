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

using DataFrames, CSV
using Random

function perturb_percent_uniform(x::AbstractVector; eps=0.01, rng=Random.GLOBAL_RNG)
    return x .* (1 .+ eps .* (2 .* rand(rng, length(x)) .- 1))
end

function perturb_percent_normal(x::AbstractVector; eps=0.01, rng=Random.GLOBAL_RNG)
    sigma = sqrt(eps)
    return x .* (1 .+ sigma .* randn(rng, length(x)))
end

jobid = parse(Int, get(ENV, "SLURM_JOB_ID", "0"))
println("JOBID = $jobid")
folder_st = parse(Int, get(ENV, "folder_start", "0"))

# test_problem = Texas_case_path
test_problem = Texas_old_case_path

gen_type = gen_type_7k_file

println("Reading instance from "*test_problem*" ... ")
psd = SCACOPFdata(test_problem)

apply_warm_start = true

active_gen_only = true

Hess_approx = true

max_iter = 500
numb_runs = 2           # the number of simulations to be performed
tau = 0.5
println("tau: $tau")

# The perterbation of the pl, and ql
perturb_mode = "normal"    # uniform or normal
perc = 0.15 # percentage in uniform random sample, or varience in normal random sample
keep_power_factor = true

if perturb_mode == "normal"
    perturb_percent = perturb_percent_normal
else
    perturb_percent = perturb_percent_uniform
end

gamma = 0.
r = 6

gamma_update = false

approx_type = "Sparse"
# approx_type = "Limited"
# approx_type = "Full"

data_path = DKL_train_xy_path

D = Dict{String, Vector{String}}()
D["None"] = [""]
D["CNN_Soft"] = [
                CNN_Soft_7k_1_model_path, CNN_Soft_7k_2_model_path
                ] 

warm_start = Dict{String, Any}(
    "warm_start" => Dict{String, Any}(
                        "primal" => Dict{Any, Any}(),
                        "dual"   => Dict{Any, Any}(),
                    ),
)
results_perb = Dict{Int, Dict{String, Dict{Int, NamedTuple}}}()

t0 = time()
# Combine jobid and folder_st to create unique seed for each parallel task
seed = jobid * 1000 + folder_st
println("Random seed = $seed (jobid=$jobid, folder_start=$folder_st)")
rng = MersenneTwister(seed)
for j = 1:numb_runs
    psd_temp = deepcopy(psd)

    if j == 1
        perb = false
    else
        perb = true
        base_p = psd.loads[!, :PL]
        base_q = psd.loads[!, :QL]

        # Perturb active power demand (PL)
        psd_temp.loads[!, :PL] = perturb_percent(base_p; eps=perc, rng=rng)

        # Perturb reactive power demand (QL)
        if keep_power_factor
            p_scaled = psd_temp.loads[!, :PL]
            q_scaled = copy(Float64.(base_q))

            # Buses with nonzero active power: preserve Q/P ratio
            mask_p_nonzero = abs.(base_p) .> 1e-8
            if any(mask_p_nonzero)
                ratio = zeros(Float64, length(base_p))
                ratio[mask_p_nonzero] .= base_q[mask_p_nonzero] ./ base_p[mask_p_nonzero]
                q_scaled[mask_p_nonzero] .= ratio[mask_p_nonzero] .* p_scaled[mask_p_nonzero]
            end

            # Effective PL perturbation for logging / reuse
            p_noise = zeros(Float64, length(base_p))
            p_noise[mask_p_nonzero] .= p_scaled[mask_p_nonzero] ./ base_p[mask_p_nonzero] .- 1.0

            # Purely reactive buses: apply the same relative perturbation to Q
            mask_p_zero_q_nonzero = .!mask_p_nonzero .& (abs.(base_q) .> 1e-8)
            if any(mask_p_zero_q_nonzero)
                q_scaled[mask_p_zero_q_nonzero] .=
                    base_q[mask_p_zero_q_nonzero] .* (1.0 .+ p_noise[mask_p_zero_q_nonzero])
            end

            psd_temp.loads[!, :QL] = q_scaled
        else
            # Independently perturb reactive power demand
            psd_temp.loads[!, :QL] = perturb_percent(base_q; eps=perc, rng=rng)
        end

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

    results = Dict{String, Dict{Int, NamedTuple}}()

    for (model_type, model_paths) in D
        println("Model: $model_type, Perturbation $j")
        tmp = Dict{Int, NamedTuple}()

        for (i, path) in enumerate(model_paths)
            println("Model: $model_type, Perturbation $j, Run $i")

            solution_dir = joinpath(case_sol_path, "$(model_type)", "folder_$(j+folder_st)")
            if !ispath(solution_dir)
                mkpath(solution_dir)
            end

            if model_type == "None"
                surrogate = Dict("model_type" => nothing)

                num_iter, total_time, base_cost,
                Surr_Feasibility_margin, ter_status, norm_grad, _, warm_start_temp =
                    TSACOPF(
                        test_problem,
                        solution_dir,
                        pf_limit_file,
                        nothing,
                        psd_temp,
                        tau;
                        max_iter = 300,
                        gen_type = gen_type,
                        save_warm_start = true,
                    )

                warm_start["warm_start"] = warm_start_temp

            else
                surrogate = tsslope_lib.load_model(
                    path,
                    data_path,
                    model_type;
                    active_gen_only = active_gen_only,
                )

                num_iter, total_time, base_cost,
                Surr_Feasibility_margin, ter_status, norm_grad, _ =
                    TSACOPF(
                        test_problem,
                        solution_dir,
                        pf_limit_file,
                        surrogate,
                        psd_temp,
                        tau;
                        max_iter = max_iter,
                        Hess_approx = Hess_approx,
                        gamma = gamma,
                        approx_type = approx_type,
                        r = r,
                        gamma_update = gamma_update,
                        gen_type = gen_type,
                        diag_pattern = false,
                        warm_start = warm_start["warm_start"],
                    )
            end

            println(
                "num_iter: $num_iter,\n",
                "total_time: $total_time,\n",
                "base_cost: $base_cost,\n",
                "termination_status: $ter_status,\n",
                "Surr_Feasibility_margin: $Surr_Feasibility_margin",
            )

            tmp[i] = (
                num_iter = num_iter,
                total_time = total_time,
                base_cost = base_cost,
                Surr_Feasibility_margin = Surr_Feasibility_margin,
                norm_grad = norm_grad,
                termination_status = ter_status,
                perb = perb,
                model_path = path,
            )
        end

        results[model_type] = tmp
    end

    results_perb[j] = results
end


cvs_filename = "results_all_model_new_data_refine_perb_$(perc)_runs_$(numb_runs).csv"

results_dir = "./test_7k_results"

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