include("/p/lustre1/hiop/project/scidac/tsSLOPE/tsslope-pump/driver/load_config.jl")

using Pkg;
# Use the new env_julia project instead of exajugo
Pkg.activate("/p/lustre1/hiop/project/scidac/env_julia")
push!(LOAD_PATH, string(path_to_exajugo, "/modules"))

# use the Emil helper
include("/p/lustre1/hiop/project/scidac/tsSLOPE/tsslope-pump/driver/exajugo_emil.jl")

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
    # adjust to std from var
    sigma = eps
    # add load scaler
    return x .* 1.05 .* (1 .+ sigma .* randn(rng, length(x)))
end

jobid = parse(Int, get(ENV, "SLURM_JOB_ID", "0"))
println("JOBID = $jobid")
folder_st = parse(Int, get(ENV, "folder_start", "0"))

test_problem = ACTIVSg500_case_path

gen_type = nothing

println("Reading instance from "*test_problem*" ... ")
psd = SCACOPFdata(test_problem)

active_gen_only = true

# use the full CNF iteration budget
max_iter = 200
numb_runs = 1         # the number of simulations to be performed

model_type = "CNF"

if model_type == "CNF"
    tau = 0.0   # change to 0.0
else
    tau = 0.5
end
println("tau: $tau")

# The perterbation of the pl, and ql
perturb_mode = "normal"    # uniform or normal
perc = 0.15 # percentage in uniform random sample, or std in normal random sample
keep_power_factor = true

if perturb_mode == "normal"
    perturb_percent = perturb_percent_normal
else
    perturb_percent = perturb_percent_uniform
end

gamma = 0.
r = 6

apply_warm_start = true

display_Sur_Eval_stats = false

gamma_update = false

Hess_approx = true # if false it uses the exact Hessian, otherwise it uses an approximation

approx_type = "Full"
# approx_type = "Limited"
approx_type = "Sparse"

data_path = LLNL_data_record

# Use the new CNF checkpoint from generate_samples
#CNF_checkpoint_path = "/p/lustre1/hiop/project/scidac/generate_samples/emil_src/CDF_Flow/cdf_flow_checkpoint.pt"
#CNF_checkpoint_path = CNF_model_final_path
#exp_007
# CNF_checkpoint_path = "/p/lustre1/hiop/project/scidac/tsSLOPE/tsslope-ml/cnf_exp_007/cdf_flow_checkpoint.pt"
#current
CNF_checkpoint_path = "/p/lustre1/hiop/project/scidac/tsSLOPE/tsslope-ml/cnf_current/cdf_flow_checkpoint.pt"


D = Dict{String, Vector{String}}()
D["CNF"] = [CNF_checkpoint_path]

results_perb = Dict{Int, Dict{String, Dict{Int, NamedTuple}}}()

t0 = time()
# fix the load draw seed
seed = 172
Random.seed!(seed)
println("Random seed = $seed (jobid=$jobid, folder_start=$folder_st)")
rng = MersenneTwister(seed)

# Maximum number of resampling attempts per run
max_resample_attempts = 2

# Track failures for summary
failure_log = []

for j = 1:numb_runs
    run_success = false
    attempt = 0

    # Initialize variables outside try block so they're accessible in catch
    psd_temp = nothing
    base_p = nothing
    base_q = nothing

    while !run_success && attempt < max_resample_attempts
        attempt += 1
        if attempt > 1
            println("Resampling run $j, attempt $attempt/$max_resample_attempts")
        end

        # try
            psd_temp = deepcopy(psd)

            perb = true
            base_p = psd.loads[!, :PL]
            base_q = psd.loads[!, :QL]

            # Perturb active power demand (PL)
            # share P/Q load multipliers
            factors = perturb_percent(ones(length(base_p)); eps=perc, rng=rng)
            all(x -> isfinite(x) && x > 0,factors) || error("Invalid load draw; no clipping or resampling")
            psd_temp.loads[!, :PL] = base_p .* factors

            # Perturb reactive power demand (QL)
            if keep_power_factor
                psd_temp.loads[!, :QL] = base_q .* factors
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

            results = Dict{String, Dict{Int, NamedTuple}}()

            for (model_type, model_paths) in D
                println("Model: $model_type, Perturbation $j")
                tmp = Dict{Int, NamedTuple}()

                for (i, path) in enumerate(model_paths)
                    println("Model: $model_type, Perturbation $j, Run $i")

                    solution_dir = joinpath(case_sol_path, "$(model_type)_$(i)", "folder_$(j+folder_st)")
                    if !ispath(solution_dir)
                        mkpath(solution_dir)
                    end

                    surrogate = tsslope_lib.load_model(
                        path,
                        data_path,
                        model_type;
                        active_gen_only = active_gen_only,
                    )

		    # test zero probability - ask Emil
		    # surrogate["alpha"] = 0.999


                    # if apply_warm_start && approx_type == "Sparse"
                    if apply_warm_start 
                        # reset the base solver seed
                        Random.seed!(123)
                        _, _, _, _, _, _, _, warm_start =
                            TSACOPF(
                                test_problem,
                                solution_dir,
                                pf_limit_file,
                                nothing,
                                psd_temp,
                                tau,
                                max_iter = 300,
                                gen_type = gen_type, 
                                save_warm_start = true
                            )

                        # reset the CNF solver seed
                        Random.seed!(123)
                        num_iter, total_time, base_cost, Surr_Feasibility_margin, ter_status, norm_grad, _ =
                            TSACOPF(
                                test_problem,
                                solution_dir,
                                pf_limit_file,
                                surrogate,
                                psd_temp,
                                tau,
                                max_iter = max_iter,
                                Hess_approx = Hess_approx,
                                gamma = gamma,
                                approx_type = approx_type,
                                r = r,
                                gamma_update = gamma_update,
                                gen_type = gen_type, 
                                diag_pattern = false,
                                warm_start = warm_start,
                                display_Sur_Eval_stats = display_Sur_Eval_stats
                            )


                    else
                        num_iter, total_time, base_cost, Surr_Feasibility_margin, ter_status, norm_grad, _ =
                            TSACOPF(
                                test_problem,
                                solution_dir,
                                pf_limit_file,
                                surrogate,
                                psd_temp,
                                tau,
                                max_iter = max_iter,
                                Hess_approx = Hess_approx,
                                gamma = gamma,
                                approx_type = approx_type,
                                r = r,
                                gamma_update = gamma_update,
                                gen_type = gen_type, 
                                display_Sur_Eval_stats = display_Sur_Eval_stats
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
            run_success = true  # If we reach here, the run succeeded

        # catch e
        #     error_msg = sprint(showerror, e, catch_backtrace())
        #     println("ERROR in run $j, attempt $attempt: ", e)

        #     # Save perturbed loads for debugging
        #     failure_dir = "./validation/CNF_results/failed_perturbations"
        #     if !ispath(failure_dir)
        #         mkpath(failure_dir)
        #     end

        #     failure_file = joinpath(failure_dir, "failed_run_$(j+folder_st)_attempt_$(attempt).csv")

        #     # Only save if we have valid data
        #     if !isnothing(psd_temp) && !isnothing(base_p) && !isnothing(base_q)
        #         try
        #             # Save the perturbed PL and QL that caused the failure
        #             df_failed = DataFrame(
        #                 bus_id = psd_temp.loads[!, :I],
        #                 PL_perturbed = psd_temp.loads[!, :PL],
        #                 QL_perturbed = psd_temp.loads[!, :QL],
        #                 PL_base = base_p,
        #                 QL_base = base_q,
        #                 PL_ratio = psd_temp.loads[!, :PL] ./ base_p,
        #                 QL_ratio = psd_temp.loads[!, :QL] ./ base_q,
        #             )
        #             CSV.write(failure_file, df_failed)
        #             println("Saved failed perturbation to: $failure_file")

        #             # Log this failure
        #             push!(failure_log, (
        #                 run_number = j + folder_st,
        #                 attempt = attempt,
        #                 error_type = string(typeof(e)),
        #                 error_msg = first(split(error_msg, '\n'), 1)[1],  # First line of error
        #                 file_saved = failure_file,
        #                 timestamp = time() - t0,
        #             ))
        #         catch save_error
        #             println("Warning: Could not save failed perturbation: $save_error")
        #         end
        #     else
        #         println("Skipping perturbation save (error occurred before data was generated)")
        #         # Still log the failure without file
        #         push!(failure_log, (
        #             run_number = j + folder_st,
        #             attempt = attempt,
        #             error_type = string(typeof(e)),
        #             error_msg = first(split(error_msg, '\n'), 1)[1],
        #             file_saved = "not_saved",
        #             timestamp = time() - t0,
        #         ))
        #     end

        #     if attempt >= max_resample_attempts
        #         println("Failed after $max_resample_attempts attempts. Skipping run $j.")
        #         # Record failed run
        #         results_perb[j] = Dict{String, Dict{Int, NamedTuple}}()
        #     else
        #         println("Retrying with new random sample...")
        #     end
        # end
    end
end


cvs_filename = "results_CNF_perb_$(perc)_runs_$(numb_runs)_folder_$(folder_st).csv"

results_dir = "./validation/CNF_results"

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

# Save failure summary if there were any failures
if length(failure_log) > 0
    println("\n========================================")
    println("FAILURE SUMMARY: $(length(failure_log)) failures occurred")
    println("========================================")
    df_failures = DataFrame(failure_log)
    failure_summary_file = results_dir * "/failure_summary_folder_$(folder_st).csv"
    CSV.write(failure_summary_file, df_failures)
    println("Failure summary saved to: $failure_summary_file")
    println("Individual perturbation files saved in: ./validation/CNF_results/failed_perturbations/")
else
    println("\n========================================")
    println("All runs completed successfully with no failures!")
    println("========================================")
end

et = time() - t0
println("Total time elapsed $et")
