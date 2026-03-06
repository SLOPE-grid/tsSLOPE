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


print("Reading instance from "*case_path*" ... ")
psd = SCACOPFdata(case_path)

save_spect_info = false

save_Hess = false

active_gen_only = true

model_type = "CNF"
# model_type = "CNN_Soft"
# model_type = "None"

max_iter = 300

tau = 0.0

t0 = time()

Hess_approx = true

gamma = -100.
r = 6
# approx_type = "Sparse"
# approx_type = "Limited"
approx_type = "Full"

if model_type == "None"
    surrogate = Dict(
        "model_type" => nothing,
        )
else
    # surrogate, data, TSI = tsslope_lib.load_model(CNF_model_final_path, LLNL_data_record, model_type, active_gen_only = active_gen_only)
    surrogate, data, TSI = tsslope_lib.load_model(CNN_Soft_UQ_1_model_path, LLNL_data_record, model_type, active_gen_only = active_gen_only)
end

num_iter, total_time, base_cost,
Surr_Feasibility_margin, termination_status, norm_grad, pg, hess_analy =
    TSACOPF(
        case_path,
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
        r = r
    )


results = (
    num_iter = num_iter,
    total_time = total_time,
    base_cost = base_cost,
    norm_grad = norm_grad,
    Surr_Feasibility_margin = Surr_Feasibility_margin,
    termination_status = termination_status,
)

println("Total time elapsed $(time() - t0)")

# cvs_filename = "results_SR1.csv"
# h5_filename = "results_SR1.h5"

# results_dir = "./SR1_results"

# if !ispath(results_dir)
#     mkpath(results_dir)
# end

# rows = NamedTuple[]

# for (model_name, perb_dict) in results
#     for (j, r) in perb_dict
#         push!(rows, (
#             model = model_name,
#             perturbation_id = j,
#             num_iter = r.num_iter,
#             total_time = r.total_time,
#             base_cost = r.base_cost,
#             norm_grad = r.norm_grad,
#             Surr_Feasibility_margin = r.Surr_Feasibility_margin,
#             termination_status = r.termination_status,
#         ))
#     end
# end

# df = DataFrame(rows)
# CSV.write(results_dir * "/" * cvs_filename, df)

# filename = abspath(results_dir * "/" * h5_filename)
# tmpfile  = filename * ".tmp"

# h5open(tmpfile, "w") do f
#     for (act, run_dict) in hess_analy
#         g_act = create_group(f, string(act))

#         for (j, iter_dict) in run_dict
#             run_name = "run_$(lpad(string(j), 3, '0'))"
#             g_run = create_group(g_act, run_name)

#             for (iter, results) in iter_dict
#                 iter_name = "iter_$(lpad(string(iter), 4, '0'))"
#                 g_iter = create_group(g_act, iter_name)

#                 if save_Hess
#                     g_iter["H"]                          = results["H"]
#                 end
                
#                 g_iter["eigenvalues"]                    = results["eigenvalues"]  
#                 g_iter["hessian_density"]                = results["hessian_density"] 
#                 g_iter["nonzero_ev_indx"]                = results["nonzero_ev_indx"]
#                 g_iter["nonzero_evecs"]                  = results["nonzero_evecs"]
#                 g_iter["grad"]                           = results["grad"]
#                 g_iter["pg"]                             = results["pg"]
#             end
#         end
#     end
# end

# # Atomic replace
# mv(tmpfile, filename; force=true)
