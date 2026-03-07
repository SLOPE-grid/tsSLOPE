
using Ipopt, JuMP, Printf
using SCACOPFSubproblems
const MOI = JuMP.MOI

using MAT
using LinearAlgebra
using SparseArrays

jl_lib = string(path_to_tsslope,"/tsslope-pump-jl")
include(string(jl_lib,"/load_case.jl"))

const CACHE = Dict{UInt64, Any}()

function TSACOPF(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, tau; spect_info::Bool = false, save_Hess::Bool = false, max_iter::Int = 200, ev_nonzero_tol::Float64 = 1e-10, Hess_approx::Bool = false, gamma::Float64 = 0., approx_type::String = "Sparse", r::Int = 6)

	opt = optimizer_with_attributes(Ipopt.Optimizer,
		                            "sb" => "yes",
		                            # "linear_solver" => "ma57",
                                    "max_iter" =>  max_iter,
                                    "print_timing_statistics" => "yes",
                                    )

    # Case when there is no surrogate 
    if Surrogate["model_type"] == nothing
        return TSACOPF_No_Surrogate(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt)

    # Case when the true surrogate Hessian is used
    elseif Hess_approx == false
        return TSACOPF_True_Surrogate_Hessian(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt, tau, spect_info = spect_info, save_Hess = save_Hess, max_iter = max_iter, ev_nonzero_tol = ev_nonzero_tol)

    # Case when full memory SR1 is used for the Hessian
    elseif approx_type == "Full"
        return TSACOPF_Full_Memory_SR1(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt, tau; max_iter = max_iter, gamma = gamma, approx_type = approx_type)

    # Case when limited memory SR1 is used for the Hessian
    elseif approx_type == "Limited"
        return TSACOPF_Limited_Memory_SR1(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt, tau; max_iter = max_iter, gamma = gamma, approx_type = approx_type, r = r)

    # Case when sparse limited memory SR1 is used for the Hessian
    else
        return TSACOPF_sparse_Limited_Memory_SR1(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt, tau; max_iter = max_iter, gamma = gamma, approx_type = approx_type, r = r)
    end
end

function TSACOPF_No_Surrogate(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, opt)
	print("done.\nCreating index lists for TSI constraint ...")
	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
		
	print("done.\nSolving basecase using sparse OPF ...")

    # get primal starting point
    x0 = get_primal_starting_point(psd)

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir="output")

    hess_analy =  Dict{Int, Dict{String,Any}}()
    
    total_time = MOI.get(m, MOI.SolveTimeSec())

    termination_status = MOI.get(m, MOI.TerminationStatus())
    num_iter = MOI.get(m, MOI.BarrierIterations())

    Surr_Feasibility_margin = NaN
    norm_grad = NaN
        
	print("done. Objective value: \$", round(solution.base_cost, digits=1),
		".\nWriting solution to "*solution_dir*" ... \n")

    return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g, hess_analy

end

function TSACOPF_True_Surrogate_Hessian(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, opt, tau; spect_info::Bool = false, save_Hess::Bool = false, max_iter::Int = 200, ev_nonzero_tol::Float64 = 1e-10)
	print("done.\nCreating index lists for TSI constraint ...")
	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
		
	print("done.\nSolving basecase using sparse OPF ...")
                  
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

                    hess_analy_temp["grad"] = grad
                    hess_analy_temp["pg"]   = pg_vec
                    hess_analy_temp["pl"]   = st_args["PL"]
                    hess_analy_temp["ql"]   = st_args["QL"]

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

        @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= tau )
        
        if !ispath(solution_dir)
            mkpath(solution_dir)
        end
    end

    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir="output")
    
    total_time = MOI.get(m, MOI.SolveTimeSec())

    termination_status = MOI.get(m, MOI.TerminationStatus())
    num_iter = MOI.get(m, MOI.BarrierIterations())

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

        hess_analy_temp["grad"] = grad
        hess_analy_temp["pg"]   = solution.p_g
        hess_analy_temp["pl"]   = st_args["PL"]
        hess_analy_temp["ql"]   = st_args["QL"]

        hess_analy[iter] = hess_analy_temp
    end
        
	print("done. Objective value: \$", round(solution.base_cost, digits=1),
		".\nWriting solution to "*solution_dir*" ... \n")

    return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g, hess_analy
end

function TSACOPF_Full_Memory_SR1(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, opt, tau; max_iter = max_iter, gamma = gamma, approx_type = approx_type)
	print("done.\nCreating index lists for TSI constraint ...")
	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
		
	print("done.\nSolving basecase using sparse OPF ...")
                  
    # get primal starting point
    x0 = get_primal_starting_point(psd)

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    iter = 1

    hess_analy =  Dict{Int, Dict{String,Any}}()

    N_gen = st_args["numb_active_gen"]

    # For SR1 Hessian approximation
    B0 = gamma * Matrix{Float64}(I, 2*N_gen, 2*N_gen)
    x = Vector{Vector{Float64}}()
    g = Vector{Vector{Float64}}()
    S = Vector{Vector{Float64}}()
    Y = Vector{Vector{Float64}}()
    B = Vector{Matrix{Float64}}()
    grad_temp = Vector{Vector{Float64}}()
    push!(B, B0)

    function tsif(args...)
        pg_vec = collect(args[1:N_gen])
        qg_vec = collect(args[N_gen+1:2*N_gen])

        return TSIConstraint(psd, Surrogate, st_args, pg_vec, qg_vec)
    end

    function tsig(g::AbstractVector, args...)
        pg_vec = collect(args[1:N_gen])
        qg_vec = collect(args[N_gen+1:2*N_gen])
        grad = TSIConstraintPrime(psd, Surrogate, st_args, pg_vec, qg_vec)
        push!(grad_temp, grad)
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
            grad = grad_temp[end]
            popfirst!(grad_temp)
            x_temp = zeros(2*N_gen)
            if iter == 1
                x_temp[1:N_gen] .= pg_vec 
                if Surrogate["model_type"] == "CNF"
                    x_temp[N_gen+1:2*N_gen] .= qg_vec 
                end
                push!(x, x_temp)
                push!(g, grad)
                hess = B0
            else
                x_temp[1:N_gen] .= pg_vec 
                if Surrogate["model_type"] == "CNF"
                    x_temp[N_gen+1:2*N_gen] .= qg_vec 
                end
                push!(x, x_temp)
                push!(g, grad)
                push!(S, x[2] - x[1])
                push!(Y, g[2] - g[1])
                popfirst!(x)
                popfirst!(g)

                hess = TSIConstraintHessApprox(st_args, B[1], S[end], Y[end], approx_type)

                push!(B, hess)
                popfirst!(B)
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

    @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= tau )
    
    if !ispath(solution_dir)
        mkpath(solution_dir)
    end

    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir="output")
    
    total_time = MOI.get(m, MOI.SolveTimeSec())

    termination_status = MOI.get(m, MOI.TerminationStatus())
    num_iter = MOI.get(m, MOI.BarrierIterations())

    Surr_Feasibility_margin = TSIConstraint(psd, Surrogate, st_args, solution.p_g, solution.q_g) - tau
    grad = TSIConstraintPrime(psd, Surrogate, st_args, solution.p_g, solution.q_g)
    norm_grad = dot(grad, grad)

        
	print("done. Objective value: \$", round(solution.base_cost, digits=1),
		".\nWriting solution to "*solution_dir*" ... \n")

    return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g, hess_analy

end

function TSACOPF_Limited_Memory_SR1(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, opt, tau; max_iter::Int = 200, gamma::Float64 = 0., approx_type::String = "Sparse", r::Int = 6)
	print("done.\nCreating index lists for TSI constraint ...")
	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
		                  
    # get primal starting point
    x0 = get_primal_starting_point(psd)

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    hess_analy =  Dict{Int, Dict{String,Any}}()

    # Used to store spectral data
    iter = 1

    # limited memory parameter
    LMp = r   

    N_gen = st_args["numb_active_gen"]

    # For SR1 Hessian approximation
    B0 = gamma * Matrix{Float64}(I, 2*N_gen, 2*N_gen)
    x = Vector{Vector{Float64}}()
    g = Vector{Vector{Float64}}()
    S = Vector{Vector{Float64}}()
    Y = Vector{Vector{Float64}}()
    B = Vector{Matrix{Float64}}()
    grad_temp = Vector{Vector{Float64}}()
    push!(B, B0)

    function tsif(args...)
        pg_vec = collect(args[1:N_gen])
        qg_vec = collect(args[N_gen+1:2*N_gen])

        return TSIConstraint(psd, Surrogate, st_args, pg_vec, qg_vec)
    end

    function tsig(g::AbstractVector, args...)
        pg_vec = collect(args[1:N_gen])
        qg_vec = collect(args[N_gen+1:2*N_gen])
        grad = TSIConstraintPrime(psd, Surrogate, st_args, pg_vec, qg_vec)
        push!(grad_temp, grad)
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
            grad = grad_temp[end]
            popfirst!(grad_temp)
            x_temp = zeros(2*N_gen)
            if iter == 1
                x_temp[1:N_gen] .= pg_vec 
                if Surrogate["model_type"] == "CNF"
                    x_temp[N_gen+1:2*N_gen] .= qg_vec 
                end
                push!(x, x_temp)
                push!(g, grad)
                hess = B0
            else
                x_temp[1:N_gen] .= pg_vec 
                if Surrogate["model_type"] == "CNF"
                    x_temp[N_gen+1:2*N_gen] .= qg_vec 
                end
                push!(x, x_temp)
                push!(g, grad)
                push!(S, x[2] - x[1])
                push!(Y, g[2] - g[1])
                popfirst!(x)
                popfirst!(g)

                hess = TSIConstraintHessApprox(st_args, B0, S, Y, approx_type)

                if length(S) > LMp
                    popfirst!(S)
                    popfirst!(Y)
                end
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

    @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= tau )
    
    if !ispath(solution_dir)
        mkpath(solution_dir)
    end

    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir="output")
    
    total_time = MOI.get(m, MOI.SolveTimeSec())

    termination_status = MOI.get(m, MOI.TerminationStatus())
    num_iter = MOI.get(m, MOI.BarrierIterations())

    Surr_Feasibility_margin = TSIConstraint(psd, Surrogate, st_args, solution.p_g, solution.q_g) - tau
    grad = TSIConstraintPrime(psd, Surrogate, st_args, solution.p_g, solution.q_g)
    norm_grad = dot(grad, grad)
        
	print("done. Objective value: \$", round(solution.base_cost, digits=1),
		".\nWriting solution to "*solution_dir*" ... \n")

    return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g, hess_analy

end

function TSACOPF_sparse_Limited_Memory_SR1(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, opt, tau; max_iter::Int = 200, gamma::Float64 = 0., approx_type::String = "Sparse", r::Int = 6)
	print("done.\nCreating index lists for TSI constraint ...")
	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
		
	print("done.\nSolving basecase using sparse OPF ...")
                  
    # get primal starting point
    x0 = get_primal_starting_point(psd)

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    hess_analy =  Dict{Int, Dict{String,Any}}()

    # Used to store spectral data
    iter = 1

    # limited memory parameter
    LMp = r   

    N_gen = st_args["numb_active_gen"]

    # For SR1 Hessian approximation
    B0 = gamma * Matrix{Float64}(I, 2*N_gen, 2*N_gen)
    x = Vector{Vector{Float64}}()
    g = Vector{Vector{Float64}}()
    S = Vector{Vector{Float64}}()
    Y = Vector{Vector{Float64}}()
    B = Vector{Matrix{Float64}}()
    grad_temp = Vector{Vector{Float64}}()
    push!(B, B0)

    function tsif(args...)
        pg_vec = collect(args[1:N_gen])
        qg_vec = collect(args[N_gen+1:2*N_gen])
        return TSIConstraint(psd, Surrogate, st_args, pg_vec, qg_vec)
    end

    function tsig(g::AbstractVector, args...)
        pg_vec = collect(args[1:N_gen])
        qg_vec = collect(args[N_gen+1:2*N_gen])
        grad = TSIConstraintPrime(psd, Surrogate, st_args, pg_vec, qg_vec)
        push!(grad_temp, grad)
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
            grad = grad_temp[end]
            popfirst!(grad_temp)
            x_temp = zeros(2*N_gen)
            if iter == 1
                x_temp[1:N_gen] .= pg_vec 
                if Surrogate["model_type"] == "CNF"
                    x_temp[N_gen+1:2*N_gen] .= qg_vec 
                end
                push!(x, x_temp)
                push!(g, grad)
                hess = B0
            else
                x_temp[1:N_gen] .= pg_vec 
                if Surrogate["model_type"] == "CNF"
                    x_temp[N_gen+1:2*N_gen] .= qg_vec 
                end
                push!(x, x_temp)
                push!(g, grad)
                push!(S, x[2] - x[1])
                push!(Y, g[2] - g[1])
                popfirst!(x)
                popfirst!(g)

                hess = TSIConstraintHessApprox(st_args, B0, S, Y, approx_type)

                if length(S) > LMp
                    popfirst!(S)
                    popfirst!(Y)
                end

                push!(B, hess)
                popfirst!(B)
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

    @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= tau )
    
    if !ispath(solution_dir)
        mkpath(solution_dir)
    end

    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir="output")
    
    total_time = MOI.get(m, MOI.SolveTimeSec())

    termination_status = MOI.get(m, MOI.TerminationStatus())
    num_iter = MOI.get(m, MOI.BarrierIterations())
 
    Surr_Feasibility_margin = TSIConstraint(psd, Surrogate, st_args, solution.p_g, solution.q_g) - tau
    grad = TSIConstraintPrime(psd, Surrogate, st_args, solution.p_g, solution.q_g)
    norm_grad = dot(grad, grad)
        
	print("done. Objective value: \$", round(solution.base_cost, digits=1),
		".\nWriting solution to "*solution_dir*" ... \n")

    return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g, hess_analy

end


# function TSACOPF(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, tau; spect_info::Bool = false, save_Hess::Bool = false, max_iter::Int = 200, ev_nonzero_tol::Float64 = 1e-10, Hess_approx::Bool = false, gamma::Float64 = 0., approx_type::String = "Sparse", r::Int = 6)
# 	print("done.\nCreating index lists for TSI constraint ...")
# 	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
		
# 	print("done.\nSolving basecase using sparse OPF ...")
# 	opt = optimizer_with_attributes(Ipopt.Optimizer,
# 		                            # "linear_solver" => "ma57",
# 		                            "sb" => "yes",
#                                     "max_iter" =>  max_iter,
#                                     "print_timing_statistics" => "yes",
#                                     )
                  
#     # get primal starting point
#     x0 = get_primal_starting_point(psd)

# 	# create model
#     m, model_data = create_basecase_model(psd, opt, x0)

#     hess_analy =  Dict{Int, Dict{String,Any}}()

#     # Used to store spectral data
#     iter = 1

#     # limited memory parameter
#     LMp = r   

#     if Surrogate["model_type"] != nothing
#         N_gen = st_args["numb_active_gen"]

#         # For SR1 Hessian approximation
#         if Hess_approx
#             B0 = gamma * Matrix{Float64}(I, 2*N_gen, 2*N_gen)
#             x = Vector{Vector{Float64}}()
#             g = Vector{Vector{Float64}}()
#             S = Vector{Vector{Float64}}()
#             Y = Vector{Vector{Float64}}()
#             grad_temp = Vector{Vector{Float64}}()
#             B = Vector{Matrix{Float64}}()
#             push!(B, B0)
#         end

#         function tsif(args...)
#             pg_vec = collect(args[1:N_gen])
#             qg_vec = collect(args[N_gen+1:2*N_gen])
    
#             # sleep(1.0)

#             return TSIConstraint(psd, Surrogate, st_args, pg_vec, qg_vec)
#         end
    
#         function tsig(g::AbstractVector, args...)
#             pg_vec = collect(args[1:N_gen])
#             qg_vec = collect(args[N_gen+1:2*N_gen])
#             grad = TSIConstraintPrime(psd, Surrogate, st_args, pg_vec, qg_vec)
#             push!(grad_temp, grad)
#             g[1:2*N_gen] .= grad  
#         end
    
#         function tsih(h::AbstractMatrix, args...)
    
#             pg_vec = collect(args[1:N_gen])
#             qg_vec = collect(args[N_gen+1:2*N_gen])
#             hsh = hash(pg_vec)
    
#             if haskey(CACHE, hsh)
#                 h =  CACHE[hsh]
#                 return
#             else
#                 if Hess_approx
#                     grad = grad_temp[end]
#                     popfirst!(grad_temp)

#                     x_temp = zeros(2*N_gen)
#                     if iter == 1
#                         x_temp[1:N_gen] .= pg_vec 
#                         if Surrogate["model_type"] == "CNF"
#                             x_temp[N_gen+1:2*N_gen] .= qg_vec 
#                         end
#                         push!(x, x_temp)
#                         push!(g, grad)
#                         hess = B0
#                     else
#                         x_temp[1:N_gen] .= pg_vec 
#                         if Surrogate["model_type"] == "CNF"
#                             x_temp[N_gen+1:2*N_gen] .= qg_vec 
#                         end
#                         push!(x, x_temp)
#                         push!(g, grad)
#                         push!(S, x[2] - x[1])
#                         push!(Y, g[2] - g[1])
#                         popfirst!(x)
#                         popfirst!(g)

#                         if approx_type == "Full"
#                             hess = TSIConstraintHessApprox(st_args, B[1], S[end], Y[end], approx_type)
#                         else
#                             hess = TSIConstraintHessApprox(st_args, B0, S, Y, approx_type)
#                         end

#                         if length(S) > LMp
#                             popfirst!(S)
#                             popfirst!(Y)
#                         end

#                         push!(B, hess)
#                         popfirst!(B)
#                     end

#                 else
#                     hess = TSIConstraintPrimePrime(psd, Surrogate, st_args, pg_vec, qg_vec)
#                 end
#                 if spect_info
#                     hess_analy_temp = h_analysis( hess;
#                                 sparsity_tol = 1e-3,
#                                 verbose  = false,
#                                 save_Hess = save_Hess,
#                                 ev_nonzero_tol = 1e-10,)

#                     hess_analy_temp["grad"] = grad
#                     hess_analy_temp["pg"]   = pg_vec
#                     hess_analy_temp["pl"]   = st_args["PL"]
#                     hess_analy_temp["ql"]   = st_args["QL"]

#                     hess_analy[iter] = hess_analy_temp
#                 end

#                 iter += 1

#                 for i = 1:2*N_gen
#                     for j = 1:i
#                         h[i, j] = hess[i,j]
#                     end
#                 end
#                 CACHE[hsh] = h
#             end     
#         end

#         register(m, :tsicon, 2*N_gen, tsif, tsig, tsih)

#         @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= tau )
        
#         if !ispath(solution_dir)
#             mkpath(solution_dir)
#         end
#     end

#     solution, m = solve_basecase_from_model(m, psd, model_data, output_dir="output")
    
#     total_time = MOI.get(m, MOI.SolveTimeSec())

#     termination_status = MOI.get(m, MOI.TerminationStatus())
#     num_iter = MOI.get(m, MOI.BarrierIterations())

#     if Surrogate["model_type"] == nothing
#         Surr_Feasibility_margin = NaN
#         norm_grad = NaN
#     else
#         Surr_Feasibility_margin = TSIConstraint(psd, Surrogate, st_args, solution.p_g, solution.q_g) - tau
#         grad = TSIConstraintPrime(psd, Surrogate, st_args, solution.p_g, solution.q_g)
#         norm_grad = dot(grad, grad)
#         if spect_info
#             hess = TSIConstraintPrimePrime(psd, Surrogate, st_args, solution.p_g, solution.q_g)
#             hess_analy_temp = h_analysis( hess;
#                                 sparsity_tol = 1e-3,
#                                 verbose  = false,
#                                 save_Hess = save_Hess,
#                                 ev_nonzero_tol = 1e-10,)

#             hess_analy_temp["grad"] = grad
#             hess_analy_temp["pg"]   = solution.p_g
#             hess_analy_temp["pl"]   = st_args["PL"]
#             hess_analy_temp["ql"]   = st_args["QL"]

#             hess_analy[iter] = hess_analy_temp
#         end
#     end
        
# 	print("done. Objective value: \$", round(solution.base_cost, digits=1),
# 		".\nWriting solution to "*solution_dir*" ... \n")

#     return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g, hess_analy

# end
