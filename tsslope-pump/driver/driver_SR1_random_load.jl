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

using DataFrames
using Random

function perturb_percent_uniform(x::AbstractVector; eps=0.01, rng=Random.GLOBAL_RNG)
    return x .* (1 .+ eps .* (2 .* rand(rng, length(x)) .- 1))
end

function perturb_percent_normal(x::AbstractVector; eps=0.01, rng=Random.GLOBAL_RNG)
    sigma = sqrt(eps)
    return x .* (1 .+ sigma .* randn(rng, length(x)))
end

jobid = parse(Int, get(ENV, "SLURM_JOB_ID", "0"))
folder_st = parse(Int, get(ENV, "folder_start", "0"))

print("Reading instance from "*case_path*" ... ")
psd = SCACOPFdata(case_path)

save_spect_info = false
save_Hess = false
active_gen_only = true

model_type = "CNF"
# model_type = "CNN_Soft"
# model_type = "None"

max_iter = 300

if model_type == "CNF"
    tau = 1.0-1e-8
else
    tau = 0.5
end

Hess_approx = false
gamma = -100.
r = 6
approx_type = "Sparse"

numb_runs = 25

# The perterbation of the pl, and ql
perturb_mode = "normal"    # uniform or normal
perc = 0.15 # percentage in uniform random sample, or varience in normal random sample
keep_power_factor = true

if perturb_mode == "normal"
    perturb_percent = perturb_percent_normal
else
    perturb_percent = perturb_percent_uniform
end
    
t0 = time()
rng = MersenneTwister(jobid)
for j =1:numb_runs
    psd_temp = deepcopy(psd)

    # generate random load
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

    println("Model: $model_type, Perturbation $j")

    if model_type == "CNF"
        surrogate, data, TSI = tsslope_lib.load_model(CNF_model_final_path, LLNL_data_record, model_type, active_gen_only = active_gen_only)
    elseif model_type == "CNN_Soft"
        surrogate, data, TSI = tsslope_lib.load_model(CNN_Soft_UQ_1_model_path, LLNL_data_record, model_type, active_gen_only = active_gen_only)
    else
        surrogate, data, TSI = tsslope_lib.load_model(CNF_model_final_path, LLNL_data_record, model_type, active_gen_only = active_gen_only)
    end

    solution_dir = joinpath(case_sol_path, "$(model_type)", "folder_$(j+folder_st)")
    if !ispath(solution_dir)
        mkpath(solution_dir)
    end

    num_iter, total_time, base_cost,
    Surr_Feasibility_margin, termination_status, norm_grad, pg, hess_analy =
        TSACOPF(
            case_path,
            solution_dir,
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
            r = r
        )
 
    println("num_iter: $num_iter,\n total_time: $total_time,\n base_cost: $base_cost,\n termination_status: $termination_status,\n Surr_Feasibility_margin: $Surr_Feasibility_margin")

end

et = time() - t0
println("Total time elapsed $et")

