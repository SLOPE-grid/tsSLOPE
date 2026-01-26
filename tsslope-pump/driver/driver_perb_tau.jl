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

taus = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95]

D = Dict{String, Vector{String}}()
D["None"] = [""]
D["CNN_GELU"] = [CNN_GELU_4_model_path]
D["CNN_SiLU"] = [CNN_SiLU_2_model_path]
D["CNN_SiLU_OG"] = [CNN_model_path]
D["CNN_Soft"] = [CNN_Soft_5_model_path] 
D["CNN_Tanh"] = [CNN_Tanh_1_model_path] 

activations = ["None", "CNN_GELU", "CNN_SiLU", "CNN_SiLU_OG", "CNN_Soft", "CNN_Tanh"]

results = Dict(
    act => Dict{Int, NamedTuple}()
    for act in activations
)

t0 = time()

for j = 1:length(taus)
    tau = taus[j]

    for (model_type, model_paths) in D
        path = model_paths[1]   # exactly one model per activation

        println("Model: $model_type, tau $tau")

        model_type_tmp = model_type
        model_type_load = model_type == "CNN_SiLU_OG" ? "CNN_SiLU" : model_type

        if model_type == "None"
            surrogate = Dict(
                "model_type" => nothing,
                )
        else
            surrogate, data, TSI =
                tsslope_lib.load_model(path, LLNL_data_record, model_type_load)
        end

        num_iter, total_time, base_cost,
        Surr_Feasibility_margin, termination_status, norm_grad =
            TSACOPF(
                case_path,
                case_sol_path,
                pf_limit_file,
                surrogate,
                psd,
                tau
            )

        conv = num_iter < 200 ? 1 : 0

        # store the finally norm grad
        tmp = (
            tau = tau,
            num_iter = num_iter,
            total_time = total_time,
            base_cost = base_cost,
            norm_grad = norm_grad,
            Surr_Feasibility_margin = Surr_Feasibility_margin,
            termination_status = termination_status,
            conv = conv,
        )

        results[model_type_tmp][j] = tmp
    end
end

println("Total time elapsed $(time() - t0)")


rows = NamedTuple[]

for (model_name, perb_dict) in results
    for (j, r) in perb_dict
        push!(rows, (
            model = model_name,
            tau = r.tau,
            num_iter = r.num_iter,
            total_time = r.total_time,
            base_cost = r.base_cost,
            norm_grad = r.norm_grad,
            Surr_Feasibility_margin = r.Surr_Feasibility_margin,
            termination_status = r.termination_status,
            conv = r.conv,
        ))
    end
end

df = DataFrame(rows)
CSV.write("results_tau.csv", df)



