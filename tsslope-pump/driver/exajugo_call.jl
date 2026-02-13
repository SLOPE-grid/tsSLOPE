
using Ipopt, JuMP, Printf
using SCACOPFSubproblems
const MOI = JuMP.MOI

using MAT
using LinearAlgebra
using SparseArrays

jl_lib = string(path_to_tsslope,"/tsslope-pump-jl")
include(string(jl_lib,"/load_case.jl"))

const CACHE = Dict{UInt64, Any}()

function TSACOPF(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, tau; spect_info::Bool = false, save_Hess::Bool = false, max_iter::Int = 200, ev_nonzero_tol::Float64 = 1e-10)
	print("done.\nCreating index lists for TSI constraint ...")
	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
		
	print("done.\nSolving basecase using sparse OPF ...")
	opt = optimizer_with_attributes(Ipopt.Optimizer,
		                            # "linear_solver" => "ma57",
		                            "sb" => "yes",
                                    "max_iter" =>  max_iter,)
                  
    # get primal starting point
    x0 = get_primal_starting_point(psd)

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    hess_analy =  Dict{Int, Dict{String,Any}}()

    # Used to store spectral data
    iter = 1

    if Surrogate["model_type"] != nothing
        N_gen = st_args["numb_active_gen"]
        function tsif(args...)
            pg_vec = collect(args[1:N_gen])
            qg_vec = collect(args[N_gen+1:2*N_gen])
    
            return TSIConstraint(psd, Surrogate, st_args, pg_vec, qg_vec)
        end
    
        function tsig(g::AbstractVector, args...)
            pg_vec = collect(args[1:N_gen])
            qg_vec = collect(args[N_gen+1:2*N_gen])
            grad = TSIConstraintPrime(psd, Surrogate, st_args, pg_vec, qg_vec)
            g[1:2*N_gen] .= grad            
        end
    
        function tsih(h::AbstractMatrix, args...)
    
            pg_vec = collect(args[1:N_gen])
            qg_vec = collect(args[N_gen+1:2*N_gen])
            hsh = hash(pg_vec)
    
            if haskey(CACHE, hsh)
                h =  CACHE[hsh]
                return
            else
                hess = TSIConstraintPrimePrime(psd, Surrogate, st_args, pg_vec, qg_vec)

                if spect_info
                    hess_analy_temp = h_analysis( hess;
                                sparsity_tol = 1e-3,
                                verbose  = false,
                                save_Hess = save_Hess,
                                ev_nonzero_tol = 1e-10,)

                    hess_analy[iter] = hess_analy_temp
                end

                iter += 1

                for i = 1:2*N_gen
                    for j = 1:i
                        h[i, j] = hess[i,j]
                    end
                end
                CACHE[hsh] = h
            end     
        end

        register(m, :tsicon, 2*N_gen, tsif, tsig, tsih)

        if Surrogate["model_type"] == "CNF"
            @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= 0.0 )
        else
            @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= tau )
        end
        
        if !ispath(solution_dir)
            mkpath(solution_dir)
        end
    end

    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir="output")
    
    total_time = MOI.get(m, MOI.SolveTimeSec())

    termination_status = MOI.get(m, MOI.TerminationStatus())
    num_iter = MOI.get(m, MOI.BarrierIterations())

    if Surrogate["model_type"] == nothing
        Surr_Feasibility_margin = NaN
        norm_grad = NaN
    else
        Surr_Feasibility_margin = TSIConstraint(psd, Surrogate, st_args, solution.p_g, solution.q_g) - tau
        grad = TSIConstraintPrime(psd, Surrogate, st_args, solution.p_g, solution.q_g)
        norm_grad = dot(grad, grad)
        if spect_info
            hess = TSIConstraintPrimePrime(psd, Surrogate, st_args, solution.p_g, solution.q_g)
            hess_analy_temp = h_analysis( hess;
                                sparsity_tol = 1e-3,
                                verbose  = false,
                                save_Hess = save_Hess,
                                ev_nonzero_tol = 1e-10,)

            hess_analy[iter] = hess_analy_temp
        end
    end
        
	print("done. Objective value: \$", round(solution.base_cost, digits=1),
		".\nWriting solution to "*solution_dir*" ... \n")

    return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g, hess_analy

end
