
using Ipopt, JuMP, Printf
using SCACOPFSubproblems
const MOI = JuMP.MOI

using MAT
using LinearAlgebra
using SparseArrays

jl_lib = string(path_to_tsslope,"/tsslope-pump-jl")
include(string(jl_lib,"/load_case.jl"))

const CACHE = Dict{UInt64, Any}()

function TSACOPF(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, tau; spect_info::Union{Nothing, Bool} = false, save_Hess::Union{Nothing, Bool} = false, max_iter::Int = 200, ev_nonzero_tol::Union{Nothing, String} = nothing, Hess_approx::Union{Nothing, Bool} = false, gamma::Union{Nothing, Float64} = nothing, approx_type::Union{Nothing, String} = nothing, r::Union{Nothing, Int} = nothing)

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

    # Case when Hessian approximation was requested, but none was given
    elseif Hess_approx && isnothing(approx_type)
        error("Hess_approx was chosen to be true. However no approximation type was given. The choices for approx_type are:
        Full memory SR1: Full
        Limited memory SR1: Limited
        Sparse limited memory SR1: Sparse")
    end

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
            x_temp[1:N_gen] .= pg_vec 
            if Surrogate["model_type"] == "CNF"
                x_temp[N_gen+1:2*N_gen] .= qg_vec 
            end
            push!(x, x_temp)
            push!(g, grad)

            if iter == 1
                hess = B0
            else
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
            x_temp[1:N_gen] .= pg_vec 
            if Surrogate["model_type"] == "CNF"
                x_temp[N_gen+1:2*N_gen] .= qg_vec 
            end
            push!(x, x_temp)
            push!(g, grad)

            if iter == 1
                hess = B0
            else
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

# function TSACOPF_sparse_Limited_Memory_SR1(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, opt, tau; max_iter::Int = 200, gamma::Float64 = 0., approx_type::String = "Sparse", r::Int = 6)
# 	print("done.\nCreating index lists for TSI constraint ...")
# 	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
		
# 	print("done.\nSolving basecase using sparse OPF ...")
                  
#     # get primal starting point
#     x0 = get_primal_starting_point(psd)

# 	# create model
#     m, model_data = create_basecase_model(psd, opt, x0)

#     hess_analy =  Dict{Int, Dict{String,Any}}()

#     # Used to store spectral data
#     iter = 1

#     # limited memory parameter
#     LMp = r   

#     N_gen = st_args["numb_active_gen"]

#     # For SR1 Hessian approximation
#     B0 = gamma * Matrix{Float64}(I, 2*N_gen, 2*N_gen)
#     x = Vector{Vector{Float64}}()
#     g = Vector{Vector{Float64}}()
#     S = Vector{Vector{Float64}}()
#     Y = Vector{Vector{Float64}}()
#     grad_temp = Vector{Vector{Float64}}()

#     function tsif(args...)
#         pg_vec = collect(args[1:N_gen])
#         qg_vec = collect(args[N_gen+1:2*N_gen])
#         return TSIConstraint(psd, Surrogate, st_args, pg_vec, qg_vec)
#     end

#     function tsig(g::AbstractVector, args...)
#         pg_vec = collect(args[1:N_gen])
#         qg_vec = collect(args[N_gen+1:2*N_gen])
#         idx, grad_val, grad = TSIConstraintPrime(psd, Surrogate, st_args, pg_vec, qg_vec, approx_type)
#         push!(grad_temp, grad)
#         g[1:2*N_gen] .= grad  
#     end

#     function tsih(h::AbstractMatrix, args...)

#         pg_vec = collect(args[1:N_gen])
#         qg_vec = collect(args[N_gen+1:2*N_gen])
#         hsh = hash(pg_vec)

#         if haskey(CACHE, hsh)
#             h =  CACHE[hsh]
#             return
#         else
#             grad = grad_temp[end]
#             popfirst!(grad_temp)
#             x_temp = zeros(2*N_gen)
#             x_temp[1:N_gen] .= pg_vec 
#             if Surrogate["model_type"] == "CNF"
#                 x_temp[N_gen+1:2*N_gen] .= qg_vec 
#             end
#             push!(x, x_temp)
#             push!(g, grad)

#             if iter == 1
#                 hess = B0
#             else
#                 push!(S, x[2] - x[1])
#                 push!(Y, g[2] - g[1])
#                 popfirst!(x)
#                 popfirst!(g)

#                 I, J, V = TSIConstraintHessApprox(st_args, B0, S, Y, approx_type)

#                 if length(S) > LMp
#                     popfirst!(S)
#                     popfirst!(Y)
#                 end
#             end

#             iter += 1

#             for i = 1:2*N_gen
#                 for j = 1:i
#                     h[i, j] = hess[i,j]
#                 end
#             end
#             CACHE[hsh] = h
#         end     
#     end

#     register(m, :tsicon, 2*N_gen, tsif, tsig, tsih)

#     @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= tau )
    
#     if !ispath(solution_dir)
#         mkpath(solution_dir)
#     end

#     solution, m = solve_basecase_from_model(m, psd, model_data, output_dir="output")
    
#     total_time = MOI.get(m, MOI.SolveTimeSec())

#     termination_status = MOI.get(m, MOI.TerminationStatus())
#     num_iter = MOI.get(m, MOI.BarrierIterations())
 
#     Surr_Feasibility_margin = TSIConstraint(psd, Surrogate, st_args, solution.p_g, solution.q_g) - tau
#     grad = TSIConstraintPrime(psd, Surrogate, st_args, solution.p_g, solution.q_g)
#     norm_grad = dot(grad, grad)
        
# 	print("done. Objective value: \$", round(solution.base_cost, digits=1),
# 		".\nWriting solution to "*solution_dir*" ... \n")

#     return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g, hess_analy

# end


# ---------------------------------------------------------
# Build symbolic nonlinear model
# ---------------------------------------------------------

function build_symbolic_nlp_and_bounds(jm::JuMP.Model)

    nlp = MOI.Nonlinear.Model()
    bounds = MOI.NLPBoundsPair[]

    for (F,S) in JuMP.list_of_constraint_types(jm)

        if F <: JuMP.VariableRef
            continue
        end

        for ci in JuMP.all_constraints(jm,F,S)

            obj = JuMP.constraint_object(ci)

            MOI.Nonlinear.add_constraint(nlp,obj.func,obj.set)

            if obj.set isa MOI.LessThan
                push!(bounds, MOI.NLPBoundsPair(-Inf,obj.set.upper))

            elseif obj.set isa MOI.GreaterThan
                push!(bounds, MOI.NLPBoundsPair(obj.set.lower,Inf))

            elseif obj.set isa MOI.EqualTo
                push!(bounds, MOI.NLPBoundsPair(obj.set.value,obj.set.value))

            elseif obj.set isa MOI.Interval
                push!(bounds, MOI.NLPBoundsPair(obj.set.lower,obj.set.upper))
            end
        end
    end

    MOI.Nonlinear.set_objective(nlp,JuMP.objective_function(jm))

    return nlp,bounds
end


# ---------------------------------------------------------
# Mixed evaluator
# ---------------------------------------------------------

struct TSIMixedEvaluator{E<:MOI.AbstractNLPEvaluator} <: MOI.AbstractNLPEvaluator

    sym::E

    n::Int
    m_sym::Int
    N_gen::Int

    jac_sym_struct
    hess_sym_struct

    bb_grad_idx
    bb_hess_struct

    psd
    Surrogate
    st_args

    approx_type
    LMp

    B0

    S
    Y

    x_hist
    g_hist
end


MOI.features_available(d::TSIMixedEvaluator) =
    MOI.features_available(d.sym)

function MOI.initialize(d::TSIMixedEvaluator,features)
    MOI.initialize(d.sym,features)
end


# ---------------------------------------------------------
# Objective
# ---------------------------------------------------------

MOI.eval_objective(d::TSIMixedEvaluator,x) =
    MOI.eval_objective(d.sym,x)

function MOI.eval_objective_gradient(d::TSIMixedEvaluator,g,x)
    MOI.eval_objective_gradient(d.sym,g,x)
end


# ---------------------------------------------------------
# Constraint value
# ---------------------------------------------------------

function MOI.eval_constraint(d::TSIMixedEvaluator,g,x)

    MOI.eval_constraint(d.sym,view(g,1:d.m_sym),x)

    pg = x[1:d.N_gen]
    qg = x[d.N_gen+1:2*d.N_gen]

    g[d.m_sym+1] =
        TSIConstraint(
            d.psd,
            d.Surrogate,
            d.st_args,
            pg,
            qg
        )
end


# ---------------------------------------------------------
# Jacobian structure
# ---------------------------------------------------------

function MOI.jacobian_structure(d::TSIMixedEvaluator)

    jac = copy(d.jac_sym_struct)

    row = d.m_sym + 1

    for j in d.bb_grad_idx
        push!(jac,(row,j))
    end

    return jac
end


# ---------------------------------------------------------
# Jacobian values
# ---------------------------------------------------------

function MOI.eval_constraint_jacobian(d::TSIMixedEvaluator,J,x)

    ns = length(d.jac_sym_struct)

    MOI.eval_constraint_jacobian(d.sym,view(J,1:ns),x)

    pg = x[1:d.N_gen]
    qg = x[d.N_gen+1:2*d.N_gen]

    idx,vals,grad =
        TSIConstraintPrime(
            d.psd,
            d.Surrogate,
            d.st_args,
            pg,
            qg,
            d.approx_type
        )

    push!(d.g_hist,grad)

    offset = ns

    for k in eachindex(vals)
        J[offset+k] = vals[k]
    end
end


# ---------------------------------------------------------
# Hessian structure
# ---------------------------------------------------------

function MOI.hessian_lagrangian_structure(d::TSIMixedEvaluator)

    H = copy(d.hess_sym_struct)

    append!(H,d.bb_hess_struct)

    return H
end


# ---------------------------------------------------------
# Hessian values
# ---------------------------------------------------------

function MOI.eval_hessian_lagrangian(d,Hval,x,σ,μ)

    ns = length(d.hess_sym_struct)

    MOI.eval_hessian_lagrangian(
        d.sym,
        view(Hval,1:ns),
        x,
        σ,
        view(μ,1:d.m_sym)
    )

    μ_tsi = μ[d.m_sym+1]

    x_temp = x[1:2*d.N_gen]

    push!(d.x_hist,x_temp)

    if length(d.x_hist) == 1

        rows,cols,vals = findnz(d.B0)

    else

        s = d.x_hist[end] - d.x_hist[end-1]
        y = d.g_hist[end] - d.g_hist[end-1]

        push!(d.S,s)
        push!(d.Y,y)

        if length(d.S) > d.LMp
            popfirst!(d.S)
            popfirst!(d.Y)
        end

        rows,cols,vals =
            TSIConstraintHessApprox(
                d.st_args,
                d.B0,
                d.S,
                d.Y,
                d.approx_type
            )
    end

    offset = ns

    for k in eachindex(vals)
        Hval[offset+k] = μ_tsi * vals[k]
    end

end


# ---------------------------------------------------------
# Attach evaluator
# ---------------------------------------------------------

function attach_TSI_evaluator(
    model,
    psd,
    Surrogate,
    st_args,
    tau,
    gamma,
    approx_type,
    r
)

    nlp_sym,bounds =
        build_symbolic_nlp_and_bounds(model)

    xvars = JuMP.all_variables(model)
    vidx = JuMP.index.(xvars)

    sym_eval =
        MOI.Nonlinear.Evaluator(
            nlp_sym,
            MOI.Nonlinear.SparseReverseMode(),
            vidx
        )

    MOI.initialize(sym_eval,[:Grad,:Jac,:Hess])

    jac_struct =
        MOI.jacobian_structure(sym_eval)

    hess_struct =
        MOI.hessian_lagrangian_structure(sym_eval)

    N_gen = st_args["numb_active_gen"]

    idx,_ =
        TSIConstraintPrime(
            psd,
            Surrogate,
            st_args,
            zeros(N_gen),
            zeros(N_gen),
            approx_type
        )

    if gamma == 0
        B0 = zeros(2*N_gen,2*N_gen)
    else
        B0 = gamma * Matrix{Float64}(I,2*N_gen,2*N_gen)
    end

    rows,cols,_ =
        TSIConstraintHessApprox(
            st_args,
            B0,
            [],
            [],
            approx_type
        )

    bb_struct = collect(zip(rows,cols))

    evaluator =
        TSIMixedEvaluator(
            sym_eval,
            length(xvars),
            length(bounds),
            N_gen,
            jac_struct,
            hess_struct,
            idx,
            bb_struct,
            psd,
            Surrogate,
            st_args,
            approx_type,
            r,
            B0,
            Vector{Vector{Float64}}(),
            Vector{Vector{Float64}}(),
            Vector{Vector{Float64}}(),
            Vector{Vector{Float64}}()
        )

    push!(bounds,MOI.NLPBoundsPair(tau,Inf))

    nlp_block =
        MOI.NLPBlockData(bounds,evaluator,true)

    MOI.set(model,MOI.NLPBlock(),nlp_block)

end


# ---------------------------------------------------------
# Main solver
# ---------------------------------------------------------

function TSACOPF_sparse_Limited_Memory_SR1(
    instance_dir::String,
    solution_dir::String,
    pf_limit_file::String,
    Surrogate,
    psd,
    opt,
    tau;
    max_iter::Int = 200,
    gamma::Float64 = 0.,
    approx_type::String = "Sparse",
    r::Int = 6
)

    st_args =
        load_case(psd,pf_limit_file,Surrogate["model_type"])

    x0 =
        get_primal_starting_point(psd)

    m,model_data =
        create_basecase_model(psd,opt,x0)

    set_optimizer_attribute(m,"max_iter",max_iter)

    attach_TSI_evaluator(
        m,
        psd,
        Surrogate,
        st_args,
        tau,
        gamma,
        approx_type,
        r
    )

    solution,m =
        solve_basecase_from_model(
            m,
            psd,
            model_data
        )

    total_time =
        MOI.get(m,MOI.SolveTimeSec())

    termination_status =
        MOI.get(m,MOI.TerminationStatus())

    num_iter =
        MOI.get(m,MOI.BarrierIterations())

    Surr_Feasibility_margin =
        TSIConstraint(
            psd,
            Surrogate,
            st_args,
            solution.p_g,
            solution.q_g
        ) - tau

    grad =
        TSIConstraintPrime(
            psd,
            Surrogate,
            st_args,
            solution.p_g,
            solution.q_g,
            approx_type
        )

    norm_grad = dot(grad,grad)

    return num_iter,total_time,solution.base_cost,
           Surr_Feasibility_margin,termination_status,
           norm_grad,solution.p_g

end