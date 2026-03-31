
using Ipopt, JuMP, Printf
using SCACOPFSubproblems
const MOI = JuMP.MOI

using MAT
using LinearAlgebra
using SparseArrays
using Distributions

jl_lib = string(path_to_tsslope,"/tsslope-pump-jl")
include(string(jl_lib,"/load_case.jl"))

const CACHE = Dict{UInt64, Any}()

function TSACOPF(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, tau; spect_info::Union{Nothing, Bool} = false, save_Hess::Union{Nothing, Bool} = false, max_iter::Int = 200, ev_nonzero_tol::Union{Nothing, Float64} = nothing, Hess_approx::Union{Nothing, Bool} = false, gamma::Union{Nothing, Float64} = nothing, approx_type::Union{Nothing, String} = nothing, r::Union{Nothing, Int} = nothing)

    opt = optimizer_with_attributes(Ipopt.Optimizer,
                                    "sb" => "yes",
                                    # "linear_solver" => "ma57",
                                    "max_iter" =>  max_iter,
                                    "print_timing_statistics" => "yes",
                                    # "print_level" => 10,
                                    )

    # Case when there is no surrogate 
    if Surrogate["model_type"] == nothing
        println("Solving basecase no surrogate. \n")
        return TSACOPF_No_Surrogate(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt)

    # Case when the true surrogate Hessian is used
    elseif Hess_approx == false
        println("Solving basecase with surrogate true Hessian. \n")
        return TSACOPF_True_Surrogate_Hessian(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt, tau, spect_info = spect_info, save_Hess = save_Hess, max_iter = max_iter, ev_nonzero_tol = ev_nonzero_tol)

    # Case when Hessian approximation was requested, but none was given
    elseif Hess_approx && isnothing(approx_type)
        error("Hess_approx was chosen to be true. However no approximation type was given. The choices for approx_type are:
        Full memory SR1: Full
        Limited memory SR1: Limited
        Sparse limited memory SR1: Sparse")

    # Case when full memory SR1 is used for the Hessian
    elseif approx_type == "Full"
        println("Solving basecase with surrogate and full memory SR1. \n")
        return TSACOPF_Full_Memory_SR1(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt, tau; max_iter = max_iter, gamma = gamma, approx_type = approx_type)

    # Case when limited memory SR1 is used for the Hessian
    elseif approx_type == "Limited"
        println("Solving basecase with surrogate and limited memory SR1. \n")
        return TSACOPF_Limited_Memory_SR1(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt, tau; max_iter = max_iter, gamma = gamma, approx_type = approx_type, r = r)

    # Case when sparse limited memory SR1 is used for the Hessian
    else
        #override opt for sparse case
        opt = MOI.instantiate(opt; with_bridge_type = Float64
                            )

        return TSACOPF_sparse_Limited_Memory_SR1(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt, tau; max_iter = max_iter, gamma = gamma, approx_type = approx_type, r = r)
    end
end

# For debugging only
function print_constraint_inventory(m::JuMP.Model)
    println("---- Constraint inventory ----")
    total = 0
    for (F, S) in JuMP.list_of_constraint_types(m)
        n = JuMP.num_constraints(m, F, S)
        println("F = ", F, "   S = ", S, "   count = ", n)
        total += n
    end
    println("total typed constraints = ", total)

    println("num_nonlinear_constraints = ", JuMP.num_nonlinear_constraints(m))
    println(
        "all_constraints(include_variable_in_set_constraints=true) = ",
        length(JuMP.all_constraints(m; include_variable_in_set_constraints = true)),
    )
    println(
        "all_constraints(include_variable_in_set_constraints=false) = ",
        length(JuMP.all_constraints(m; include_variable_in_set_constraints = false)),
    )
    return
end

# For debugging only
function print_ipopt_problem_summary(m::JuMP.Model)

    backend = JuMP.backend(m)

    println("\n---- Solver problem summary (MOI) ----")

    # variable counts
    nvar = MOI.get(backend, MOI.NumberOfVariables())
    println("Total number of variables = ", nvar)

    # constraint counts
    neq_aff = MOI.get(backend,
        MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},
                                MOI.EqualTo{Float64}}())

    neq_quad = MOI.get(backend,
        MOI.NumberOfConstraints{MOI.ScalarQuadraticFunction{Float64},
                                MOI.EqualTo{Float64}}())

    neq_nlp = MOI.get(backend,
        MOI.NumberOfConstraints{MOI.ScalarNonlinearFunction,
                                MOI.EqualTo{Float64}}())

    nineq_aff = MOI.get(backend,
        MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},
                                MOI.LessThan{Float64}}()) +
                MOI.get(backend,
        MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},
                                MOI.GreaterThan{Float64}}())

    nineq_quad = MOI.get(backend,
        MOI.NumberOfConstraints{MOI.ScalarQuadraticFunction{Float64},
                                MOI.LessThan{Float64}}()) +
                 MOI.get(backend,
        MOI.NumberOfConstraints{MOI.ScalarQuadraticFunction{Float64},
                                MOI.GreaterThan{Float64}}())

    nineq_nlp = MOI.get(backend,
        MOI.NumberOfConstraints{MOI.ScalarNonlinearFunction,
                                MOI.LessThan{Float64}}()) +
                MOI.get(backend,
        MOI.NumberOfConstraints{MOI.ScalarNonlinearFunction,
                                MOI.GreaterThan{Float64}}())

    neq = neq_aff + neq_quad + neq_nlp
    nineq = nineq_aff + nineq_quad + nineq_nlp

    println("Total number of equality constraints = ", neq)
    println("Total number of inequality constraints = ", nineq)

    println("\nBreakdown:")
    println("  affine equalities   = ", neq_aff)
    println("  quadratic equalities = ", neq_quad)
    println("  nonlinear equalities = ", neq_nlp)

    println("  affine inequalities  = ", nineq_aff)
    println("  quadratic inequalities = ", nineq_quad)
    println("  nonlinear inequalities = ", nineq_nlp)

    println("--------------------------------------\n")

end

function TSACOPF_No_Surrogate(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, opt)
	print("done.\nCreating index lists for TSI constraint ...")
	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
		
	print("done.\nSolving basecase using sparse OPF ...")

    # get primal starting point
    x0 = get_primal_starting_point(psd)

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    print_constraint_inventory(m)
    # print_sample_constraints(m, max_per_type = 2)
    
    print_ipopt_problem_summary(m)   # <- ADD THIS

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

function TSACOPF_True_Surrogate_Hessian(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, opt, tau; spect_info::Bool = false, save_Hess::Bool = false, max_iter::Int = 200, ev_nonzero_tol::Union{Nothing, Float64} = 1e-10)
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

            hsh = hash((pg_vec, qg_vec))
    
            if haskey(CACHE, hsh)
                h_cached = CACHE[hsh]
                for i = 1:2*N_gen
                    for j = 1:i
                        h[i, j] = h_cached[i, j]
                    end
                end
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
                        # println("Hess[$i,$j]: ", hess[i,j])
                        h[i, j] = hess[i,j]
                    end
                end
                CACHE[hsh] = hess
            end     
        end

        register(m, :tsicon, 2*N_gen, tsif, tsig, tsih)

        @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= 0.5 )
        println("Mixed MOI Not a happy thursday")
        
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
        hsh = hash((pg_vec, qg_vec))

        if haskey(CACHE, hsh)
            h_cached = CACHE[hsh]
            for i = 1:2*N_gen
                for j = 1:i
                    h[i, j] = h_cached[i, j]
                end
            end
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

                iter += 1
            end

            for i = 1:2*N_gen
                for j = 1:i
                    h[i, j] = hess[i,j]
                end
            end
            CACHE[hsh] = hess
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

    println("Using limited SR1")
		                  
    # get primal starting point
    x0 = get_primal_starting_point(psd)

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    hess_analy =  Dict{Int, Dict{String,Any}}()

    # Used to store spectral data
    iter = 1

    # limited memory parameter
    # println(r)
    LMp = r   

    N_gen = st_args["numb_active_gen"]

    # For SR1 Hessian approximation
    B0 = gamma * Matrix{Float64}(I, 2*N_gen, 2*N_gen)
    x = Vector{Vector{Float64}}()
    g = Vector{Vector{Float64}}()
    S = Vector{Vector{Float64}}()
    Y = Vector{Vector{Float64}}()
    grad_temp = Vector{Vector{Float64}}()

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
        hsh = hash((pg_vec, qg_vec))

        if haskey(CACHE, hsh)
            h_cached = CACHE[hsh]
            for i = 1:2*N_gen
                for j = 1:i
                    h[i, j] = h_cached[i, j]
                end
            end
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
                iter += 1

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

            max_val = 0
            for i = 1:2*N_gen
                for j = 1:i
                    h[i, j] += hess[i,j]
                    val = hess[i,j]
                    max_val = max(max_val, val)
                end
            end
            println("Max_val: ", max_val)
            CACHE[hsh] = hess
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




# ---------------------------------------------------------
# TSI black-box API
# x is in solver variable order
# ---------------------------------------------------------

_bounds_pair(set) = begin
    if set isa MOI.LessThan
        return MOI.NLPBoundsPair(-Inf, set.upper)
    elseif set isa MOI.GreaterThan
        return MOI.NLPBoundsPair(set.lower, Inf)
    elseif set isa MOI.EqualTo
        return MOI.NLPBoundsPair(set.value, set.value)
    elseif set isa MOI.Interval
        return MOI.NLPBoundsPair(set.lower, set.upper)
    else
        error("Unsupported nonlinear constraint set type: $(typeof(set))")
    end
end


function build_symbolic_nlp_and_bounds(model::JuMP.Model)
    nlp = JuMP.nonlinear_model(model)

    if nlp === nothing
        return nothing, MOI.NLPBoundsPair[], false
    end

    nl_cons = JuMP.all_nonlinear_constraints(model)
    bounds = MOI.NLPBoundsPair[]

    for cref in nl_cons
        # For legacy NonlinearConstraintRef, use the nonlinear constraint index
        # to access the corresponding MOI.Nonlinear.Constraint in `nlp`.
        idx = try
            JuMP.index(cref)
        catch
            cref.index
        end

        nl_con = nlp[idx]   # MOI.Nonlinear.Constraint
        push!(bounds, _bounds_pair(nl_con.set))
    end

    # Your objective is QuadExpr, not nonlinear
    has_nl_obj = false

    return nlp, bounds, has_nl_obj
end

# Only used for debugging
function print_nlp_debug_info(m::JuMP.Model)
    println("---- JuMP nonlinear debug info ----")
    try
        println("num_nonlinear_constraints = ", JuMP.num_nonlinear_constraints(m))
    catch
        println("num_nonlinear_constraints not available")
    end

    try
        println("objective_function_type = ", JuMP.objective_function_type(m))
    catch
        println("objective_function_type not available")
    end

    src_backend = JuMP.backend(m)
    src_block = try
        MOI.get(src_backend, MOI.NLPBlock())
    catch
        nothing
    end

    println("backend has NLPBlock = ", src_block !== nothing)
    if src_block !== nothing
        println("backend # NLP constraints = ", length(src_block.constraint_bounds))
        println("backend has nonlinear objective = ", src_block.has_objective)
    end
end

# ---------------------------------------------------------
# TSI value in solver-variable ordering
# ---------------------------------------------------------
function TSI_g_bb(
    x::Vector{Float64},
    p_idx::Vector{Int},
    q_idx::Vector{Int},
    psd,
    Surrogate,
    st_args,
)
    pg = x[p_idx]
    qg = x[q_idx]
    return TSIConstraint(psd, Surrogate, st_args, pg, qg) - 0.5
end

# ---------------------------------------------------------
# Dense TSI gradient in solver-variable ordering
#
# Returns:
#   grad_full :: Vector{Float64}   (length 2N_gen, ordering [pg; qg])
# ---------------------------------------------------------
function TSI_g_bb_grad_sparse(
    x::Vector{Float64},
    p_idx::Vector{Int},
    q_idx::Vector{Int},
    psd,
    Surrogate,
    st_args,
)

    pg = x[p_idx]
    qg = x[q_idx]

    grad_full = TSIConstraintPrime(
        psd,
        Surrogate,
        st_args,
        pg,
        qg
    )

    return grad_full
end

# ---------------------------------------------------------
# Sparse TSI Hessian in local [pg;qg] ordering
# ---------------------------------------------------------
function TSI_g_bb_hess_values(
    st_args,
    B0,
    S,
    Y,
    approx_type::String,
)
    B = TSIConstraintHessApprox(
        st_args,
        B0,
        S,
        Y,
        approx_type,
    )
    return B
end

function TSI_g_bb_hess_values(
    x::Vector{Float64},
    p_idx::Vector{Int},
    q_idx::Vector{Int},
    psd,
    Surrogate,
    st_args,
)

    pg = x[p_idx]
    qg = x[q_idx]

    Hess = TSIConstraintPrimePrime(
        psd,
        Surrogate,
        st_args,
        pg,
        qg
    )

    return Hess
end

# =========================================================
# Mixed evaluator:
#   symbolic nonlinear block from JuMP/MOI AD
#   + one scalar TSI black-box constraint
# =========================================================
struct MixedTSIEvaluator{E<:MOI.AbstractNLPEvaluator} <: MOI.AbstractNLPEvaluator
    # symbolic evaluator from the original JuMP model
    sym::E
    has_sym_objective::Bool

    n::Int
    m_sym::Int
    N_gen::Int

    # solver variable indices for active generators
    p_idx::Vector{Int}
    q_idx::Vector{Int}

    # cached symbolic sparsity
    jac_sym_struct::Vector{Tuple{Int,Int}}
    hess_sym_struct::Vector{Tuple{Int,Int}}

    # TSI Jacobian layout (dense row over [p_g; q_g] in solver ordering)
    bb_grad_cols::Vector{Int}
    bb_grad_offset::Int

    # fixed TSI Hessian sparsity (global / solver ordering)
    bb_rows_local::Vector{Int}
    bb_cols_local::Vector{Int}
    bb_rows::Vector{Int}
    bb_cols::Vector{Int}

    # problem data
    psd
    Surrogate
    st_args

    approx_type::String
    LMp::Int
    B0::SparseMatrixCSC{Float64,Int}
    I_gamma::SparseMatrixCSC{Float64,Int}

    # SR1 history
    x_hist::Vector{Vector{Float64}}
    g_hist::Vector{Vector{Float64}}
    g_hist_temp::Vector{Vector{Float64}}
    S::Vector{Vector{Float64}}
    Y::Vector{Vector{Float64}}
    gamma_k::Vector{Float64}
    Bk::Vector{Matrix{Float64}}
    r_stop::Float64
end

function MOI.features_available(d::MixedTSIEvaluator)
    return MOI.features_available(d.sym)
end

function MOI.initialize(d::MixedTSIEvaluator, requested_features)
    MOI.initialize(d.sym, requested_features)
    println("Mixed TSI evaluator initialized with ", requested_features)
    return
end

# ---------------------------------------------------------
# Objective: delegate to symbolic evaluator
# ---------------------------------------------------------
function MOI.eval_objective(d::MixedTSIEvaluator, x)
    return MOI.eval_objective(d.sym, x)
end

function MOI.eval_objective_gradient(d::MixedTSIEvaluator, g, x)
    MOI.eval_objective_gradient(d.sym, g, x)
    return
end

# ---------------------------------------------------------
# Constraint values
# symbolic constraints first, then TSI as last row
# ---------------------------------------------------------
function MOI.eval_constraint(d::MixedTSIEvaluator, g, x)
    MOI.eval_constraint(d.sym, view(g, 1:d.m_sym), x)

    g[d.m_sym + 1] = TSI_g_bb(
        x,
        d.p_idx,
        d.q_idx,
        d.psd,
        d.Surrogate,
        d.st_args,
    )
    return
end

# ---------------------------------------------------------
# Jacobian structure
# symbolic rows kept as-is; TSI is appended as row m_sym+1
# ---------------------------------------------------------
function MOI.jacobian_structure(d::MixedTSIEvaluator)

    jac = Vector{Tuple{Int,Int}}()

    append!(jac, d.jac_sym_struct)

    bb_row = d.m_sym + 1

    for j in d.bb_grad_cols
        push!(jac, (bb_row, j))
    end

    return jac
end

# ---------------------------------------------------------
# Jacobian values
#
# IMPORTANT:
# symbolic Jacobian is evaluated on FULL J
# TSI row is dense over [pg; qg]
# ---------------------------------------------------------
function MOI.eval_constraint_jacobian(d::MixedTSIEvaluator, J, x)

    ns = length(d.jac_sym_struct)

    # Fill symbolic Jacobian
    MOI.eval_constraint_jacobian(d.sym, view(J, 1:ns), x)

    # Compute dense TSI gradient
    grad_full = TSI_g_bb_grad_sparse(
        x,
        d.p_idx,
        d.q_idx,
        d.psd,
        d.Surrogate,
        d.st_args
    )

    @assert length(grad_full) == length(d.bb_grad_cols)

    # Store gradient history for SR1
    push!(d.g_hist_temp, copy(grad_full))

    # Fill TSI row (dense ordering matches jacobian_structure)
    offset = ns

    for k in eachindex(grad_full)
        J[offset + k] = grad_full[k]
    end

    return
end

# ---------------------------------------------------------
# Hessian structure of the Lagrangian
# concatenate symbolic Hessian structure + TSI Hessian structure
# ---------------------------------------------------------
function MOI.hessian_lagrangian_structure(d::MixedTSIEvaluator)
    H = Vector{Tuple{Int,Int}}()
    # println("Checking lenths")
    # println(length(d.hess_sym_struct))
    # println(length(d.bb_rows))
    append!(H, d.hess_sym_struct)
    append!(H, collect(zip(d.bb_rows, d.bb_cols)))
    # println(length(H))
    return H
end

# ---------------------------------------------------------
# Hessian values
#
# H(x, σ, μ) = σ∇²f + Σ μ_i ∇²g_i + μ_tsi ∇²TSI
# ---------------------------------------------------------
function MOI.eval_hessian_lagrangian(d::MixedTSIEvaluator, Hval, x, σ, μ)
    fill!(Hval, 0.0)

    ns = length(d.hess_sym_struct)

    σ = 1.0
    μ = ones(d.m_sym + 1)

    # symbolic part uses μ[1:m_sym]
    MOI.eval_hessian_lagrangian(
        d.sym,
        view(Hval, 1:ns),
        x,
        σ,
        view(μ, 1:d.m_sym),
    )

    μ_tsi = μ[d.m_sym + 1]

    # local state [pg; qg]
    if d.Surrogate["model_type"] == "CNF"
        x_local = vcat(x[d.p_idx], x[d.q_idx])
    else 
        x_local = vcat(x[d.p_idx], x[d.q_idx] .* 0)
    end    

    grad = d.g_hist_temp[end]
    popfirst!(d.g_hist_temp)

    push!(d.x_hist, copy(x_local))
    push!(d.g_hist, grad)

    if length(d.x_hist) == 1 || length(d.g_hist) < 2
        B = d.B0
    else
        s = d.x_hist[2] - d.x_hist[1]
        y = d.g_hist[2] - d.g_hist[1]

        # gamma_k = dot(s, y)/dot(s,s)
        gamma_k = dot(y, y)/dot(s,y)
        push!(d.gamma_k, gamma_k)

        # println("d.x_hist[2]:", d.x_hist[2])
        # println("d.g_hist[2]:", d.g_hist[2])
        # println("s:", s)
        # println("y:", y)

        popfirst!(d.x_hist)
        popfirst!(d.g_hist)

        push!(d.S, s)
        push!(d.Y, y)

        yBs = y - d.Bk[1]*s
        # # Uncomment to run full memory SR1
        B = TSI_g_bb_hess_values(
                d.st_args,
                d.Bk[1],
                d.S[end],
                d.Y[end],
                d.approx_type,
            )
        println(abs(dot(s, yBs)), " , ", d.r_stop * norm(s) * norm(yBs), ", ", norm(s), ", ", norm(yBs)  )
        # push!(d.Bk, B)
        # popfirst!(d.Bk)
        # if abs(dot(s, yBs)) >= d.r_stop * norm(s) * norm(yBs) 
        #     # # For debugging choose the SR1 method
        #     # SR1_type = "Sparse"
        #     SR1_type = "Limited"
        #     B = TSI_g_bb_hess_values(
        #         d.st_args,
        #         d.gamma_k[end] * d.I_gamma,
        #         d.S,
        #         d.Y,
        #         SR1_type,
        #     )
        #     println(abs(dot(s, yBs)), " , ", d.r_stop * norm(s) * norm(yBs), ", ", norm(s), ", ", norm(yBs)  )
        #     push!(d.Bk, B)
        #     popfirst!(d.Bk)
        # else
        #     B = d.Bk[1]
        #     println("SR1 was skipped")
        # end

        # for i in 1:length(d.Y)
        #     println(
        #         " S info: ", minimum(d.S[i]), ", ", maximum(d.S[i]), ", ", minimum(abs.(d.S[i])), ", ", maximum(abs.(d.S[i])),
        #     )
        # end

        # for i in 1:length(d.Y)
        #     println(            
        #         " Y info: ", minimum(d.Y[i]), ", ", maximum(d.Y[i]), ", ", minimum(abs.(d.Y[i])), ", ", maximum(abs.(d.Y[i])),
        #     )
        # end

        if length(d.S) > d.LMp
            popfirst!(d.S)
            popfirst!(d.Y)
        end
    end

    # # uncomment to run the true Hessian
    B = TSI_g_bb_hess_values(
        x,
        d.p_idx,
        d.q_idx,
        d.psd,
        d.Surrogate,
        d.st_args
    )

    vals = [B[i,j] for (i,j) in zip(d.bb_rows_local, d.bb_cols_local)]

    offset = ns
    for k in eachindex(vals)
        Hval[offset + k] = μ_tsi * vals[k] 
    end

    println("Max_val: ", maximum(vals))

    # for k in 1:min(20, length(vals))
    #     println(
    #         "k=", k,
    #         " local=(", d.bb_rows_local[k], ",", d.bb_cols_local[k], ")",
    #         " global=(", d.bb_rows[k], ",", d.bb_cols[k], ")",
    #         " val=", B[d.bb_rows_local[k], d.bb_cols_local[k]],
    #         " stored_index=", offset + k
    #     )
    # end
    return
end

# ONly used for debugging
function print_mixed_problem_summary(dest)

    println("\n---- Mixed solver problem summary ----")

    nvar = MOI.get(dest, MOI.NumberOfVariables())
    println("Total variables = ", nvar)

    # equality constraints
    eq_aff = MOI.get(dest,
        MOI.NumberOfConstraints{
            MOI.ScalarAffineFunction{Float64},
            MOI.EqualTo{Float64}
        }())

    eq_quad = MOI.get(dest,
        MOI.NumberOfConstraints{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.EqualTo{Float64}
        }())

    # nonlinear equality constraints (usually zero here)
    eq_nl = MOI.get(dest,
        MOI.NumberOfConstraints{
            MOI.ScalarNonlinearFunction,
            MOI.EqualTo{Float64}
        }())

    # inequality constraints
    ineq_aff =
        MOI.get(dest,
            MOI.NumberOfConstraints{
                MOI.ScalarAffineFunction{Float64},
                MOI.GreaterThan{Float64}
            }()) +
        MOI.get(dest,
            MOI.NumberOfConstraints{
                MOI.ScalarAffineFunction{Float64},
                MOI.LessThan{Float64}
            }())

    ineq_quad =
        MOI.get(dest,
            MOI.NumberOfConstraints{
                MOI.ScalarQuadraticFunction{Float64},
                MOI.GreaterThan{Float64}
            }()) +
        MOI.get(dest,
            MOI.NumberOfConstraints{
                MOI.ScalarQuadraticFunction{Float64},
                MOI.LessThan{Float64}
            }())

    ineq_nl =
        MOI.get(dest,
            MOI.NumberOfConstraints{
                MOI.ScalarNonlinearFunction,
                MOI.GreaterThan{Float64}
            }()) +
        MOI.get(dest,
            MOI.NumberOfConstraints{
                MOI.ScalarNonlinearFunction,
                MOI.LessThan{Float64}
            }())

    println("Total equality constraints = ", eq_aff + eq_quad + eq_nl)
    println("Total inequality constraints = ", ineq_aff + ineq_quad + ineq_nl)

    println("\nBreakdown")
    println("  affine eq      = ", eq_aff)
    println("  quadratic eq   = ", eq_quad)
    println("  nonlinear eq   = ", eq_nl)

    println("  affine ineq    = ", ineq_aff)
    println("  quadratic ineq = ", ineq_quad)
    println("  nonlinear ineq = ", ineq_nl)

    println("--------------------------------------\n")

end

# Only used for debugging
function dump_full_hessian(d::MixedTSIEvaluator, x)

    println("\n==== Dumping full Hessian (before solve) ====")

    n = d.n

    # Hessian structure
    Hstruct = MOI.hessian_lagrangian_structure(d)
    nnz = length(Hstruct)

    # Allocate values
    Hval = zeros(nnz)

    # Use σ=1, μ=ones (or customize)
    σ = 1.0
    μ = ones(d.m_sym + 1)

    # Evaluate Hessian
    MOI.eval_hessian_lagrangian(d, Hval, x, σ, μ)

    # Reconstruct full matrix
    H = zeros(n, n)

    for k in eachindex(Hval)
        row, col = Hstruct[k]

        H[row, col] += Hval[k]
        if row == d.bb_rows[1] && col == d.bb_rows[1]
            print("Happy Friday")
        end
        if row != col
            H[col, row] += Hval[k]  # symmetry
        end
    end

    println("Hessian size: ", size(H))

    # Print small subset (avoid explosion)
    # println("\nTop-left 10x10 block:")
    # println(H[1:10, 1:10])
    # println("\nChecking TSI block entries:\n")

    # println(H[d.bb_rows[1]:d.bb_rows[5], d.bb_rows[1]:d.bb_rows[5]])

    for k in 1:min(55, length(d.bb_rows))
        r = d.bb_rows[k]
        c = d.bb_cols[k]

        # println("H[$r,$c] = ", H[r,c])
    end

    return H
end

# ---------------------------------------------------------
# Build MOI/Ipopt solver from JuMP model and REPLACE the
# existing nonlinear block with a mixed evaluator:
#
#     original symbolic nonlinear block
#     + appended scalar TSI constraint
# ---------------------------------------------------------
function build_moi_solver_with_TSI_mixed!(
    m::JuMP.Model,
    psd,
    opt,
    x0,
    Surrogate,
    st_args,
    tau::Float64,
    max_iter::Int;
    gamma::Float64 = 0.0,
    approx_type::String = "Sparse",
    r::Int = 6,
)
    print_nlp_debug_info(m)

    src_backend = JuMP.backend(m)

    # -----------------------------------------------------
    # 1) Copy JuMP backend into destination optimizer
    # -----------------------------------------------------
    index_map = MOI.copy_to(opt, src_backend)

    # println("index_map: ", index_map)

    old_opt_block =
        try
            MOI.get(opt, MOI.NLPBlock())
        catch
            nothing
        end

    
    # -----------------------------------------------------
    # 2) Rebuild symbolic nonlinear model directly from JuMP
    # -----------------------------------------------------
    nlp_sym, bounds_sym, has_sym_obj = build_symbolic_nlp_and_bounds(m)

    println("num_nonlinear_constraints(model) = ", JuMP.num_nonlinear_constraints(m))
    println("length(all_nonlinear_constraints(model)) = ", length(JuMP.all_nonlinear_constraints(m)))
    println("nlp_sym === nothing ? ", nlp_sym === nothing)
    println("length(bounds_sym) = ", length(bounds_sym))

    if nlp_sym === nothing || length(bounds_sym) == 0
        error("No symbolic nonlinear constraints were recovered from the JuMP model.")
    end

    # -----------------------------------------------------
    # 3) Variable ordering for evaluator
    # -----------------------------------------------------
    xvars = JuMP.all_variables(m)
    vidx = JuMP.index.(xvars)

    sym_eval = MOI.Nonlinear.Evaluator(
        nlp_sym,
        MOI.Nonlinear.SparseReverseMode(),
        vidx,
    )
    MOI.initialize(sym_eval, [:Grad, :Jac, :Hess])

    jac_sym_struct = MOI.jacobian_structure(sym_eval)
    hess_sym_struct = MOI.hessian_lagrangian_structure(sym_eval)
    m_sym = length(bounds_sym)

    # println(hess_sym_struct)
    max_val = 0
    for k in eachindex(hess_sym_struct)
        row = hess_sym_struct[k][1]
        col = hess_sym_struct[k][2]
        max_val = max(max_val,max(row, col))
    end
    println("Max_val:", max_val)

    println("symbolic Jacobian nnz = ", length(jac_sym_struct))
    println("symbolic Hessian nnz = ", length(hess_sym_struct))

    # -----------------------------------------------------
    # 4) Recover p_g and q_g destination indices
    # -----------------------------------------------------
    p_src = JuMP.index.(m[:p_g])
    q_src = JuMP.index.(m[:q_g])

    # println(p_src)
    # println(q_src)

    p_dest = [index_map[vi].value for vi in p_src]
    q_dest = [index_map[vi].value for vi in q_src]

    # vars = [:p_g, :v_n, :theta_n, :p_li, :q_li, :p_ti, :q_ti, :b_s, :q_g, :c_g, :pslackm_n, :pslackp_n, :qslackm_n, :qslackp_n, :sslack_li, :sslack_ti]

    # for l in vars
    #     var_src = JuMP.index.(m[l])
    #     var_dest = [index_map[vi].value for vi in var_src]
    #     println(l,": ", var_dest)
    # end 

    n = MOI.get(opt, MOI.NumberOfVariables())
    N_gen = st_args["numb_active_gen"]
    println("Number of active generators: ", N_gen)

    # -----------------------------------------------------
    # 5) Fix TSI sparsity pattern once
    # -----------------------------------------------------
    pg = x0[:p_g]
    qg = x0[:q_g]

    # Dense TSI gradient layout in solver ordering
    bb_grad_cols = vcat(p_dest, q_dest)
    bb_grad_offset = length(jac_sym_struct) + 1

    gamma_k = Vector{Float64}()
    push!(gamma_k, gamma)

    I_gamma = sparse(1.0I, 2 * N_gen, 2 * N_gen)

    B0 =
        gamma == 0.0 ?
        spzeros(2 * N_gen, 2 * N_gen) :
        gamma * sparse(I, 2 * N_gen, 2 * N_gen)  

    x_hist = Vector{Vector{Float64}}()
    g_hist = Vector{Vector{Float64}}()
    S = Vector{Vector{Float64}}()
    Y = Vector{Vector{Float64}}()

    # SR1 to choose sparse pattern for Hessian
    for i in 1:(r + 1)
        x_temp = zeros(2 * N_gen)

        pg_noise = pg + rand(Uniform(-1e-4, 1e-4), length(pg))
        x_temp[1:N_gen] .= pg_noise

        qg_noise = zeros(length(qg))
        if Surrogate["model_type"] == "CNF"
            qg_noise = qg + rand(Uniform(-1e-4, 1e-4), length(qg))
            x_temp[N_gen+1:2*N_gen] .= qg_noise
        else
            qg_noise .= qg
        end

        grad = TSIConstraintPrime(
            psd,
            Surrogate,
            st_args,
            pg_noise,
            qg_noise,
            approx_type,
        )

        push!(x_hist, x_temp)
        push!(g_hist, grad)

        if i > 1
            push!(S, x_hist[2] - x_hist[1])
            push!(Y, g_hist[2] - g_hist[1])
            popfirst!(x_hist)
            popfirst!(g_hist)
        end
    end

    rows_local, cols_local, top = TSIConstraintHessApprox(
        st_args,
        B0,
        S,
        Y,
        "Sparse_pattern",
    )

    println("Length of top_indices for Hessian: $(length(top))")
    println("top_indices for Hessian: $(sort(top) .+ 1)")
    println("top_indices for Hessian: $p_dest[sort(top) .+ 1)")

    st_args["top_idx"] = top

    # Force lower-triangular canonical ordering
    for k in eachindex(rows_local)
        if rows_local[k] < cols_local[k]
            # println("($(cols_local[k]), $(rows_local[k]))")
            rows_local[k], cols_local[k] = cols_local[k], rows_local[k]
        end
    end

    perm = sortperm(collect(zip(rows_local, cols_local)))
    # rows_local_temp = rows_local
    # cols_local_temp = cols_local
    rows_local = rows_local[perm]
    cols_local = cols_local[perm]

    idx_map = vcat(p_dest, q_dest)
    rows_global = idx_map[rows_local]
    cols_global = idx_map[cols_local]

    # for k in eachindex(rows_local)
    #     println("($(cols_local[k]), $(cols_local_temp[k]), $(cols_global[k]), $(rows_local[k]), $(rows_local_temp[k]), $(rows_global[k]))")
    # end

    println("TSI Jacobian nnz = ", length(bb_grad_cols))
    println("TSI Hessian nnz = ", length(rows_local))

    # -----------------------------------------------------
    # 6) Build mixed evaluator
    # -----------------------------------------------------
    Bk = Vector{Matrix{Float64}}()
    push!(Bk, Matrix(B0))
    mixed_eval = MixedTSIEvaluator(
        sym_eval,
        has_sym_obj,
        n,
        m_sym,
        N_gen,
        p_dest,
        q_dest,
        jac_sym_struct,
        hess_sym_struct,
        bb_grad_cols,
        bb_grad_offset,
        rows_local,
        cols_local,
        rows_global,
        cols_global,
        psd,
        Surrogate,
        st_args,
        approx_type,
        r,
        B0,
        I_gamma,
        Vector{Vector{Float64}}(),
        Vector{Vector{Float64}}(),
        Vector{Vector{Float64}}(),
        Vector{Vector{Float64}}(),
        Vector{Vector{Float64}}(),
        gamma_k,
        Bk,
        1e-4
    )

    # -----------------------------------------------------
    # DEBUG: evaluate Hessian BEFORE solver
    # -----------------------------------------------------

    x_test = zeros(n)

    # put initial point into solver ordering
    for (i, idx) in enumerate(vcat(p_dest, q_dest))
        x_test[idx] = vcat(x0[:p_g], x0[:q_g])[i]
    end

    # H = dump_full_hessian(mixed_eval, x_test)

    # -----------------------------------------------------
    # 7) Attach combined NLP block
    # -----------------------------------------------------
    new_bounds = copy(bounds_sym)
    push!(new_bounds, MOI.NLPBoundsPair(0.0, Inf))
    println("Mixed MOI Not a happy thursday")

    new_nlp_block = MOI.NLPBlockData(
        new_bounds,
        mixed_eval,
        has_sym_obj,
    )

    MOI.set(opt, MOI.NLPBlock(), new_nlp_block)

    # -----------------------------------------------------
    # 8) Final check
    # -----------------------------------------------------
    new_opt_block =
        try
            MOI.get(opt, MOI.NLPBlock())
        catch
            nothing
        end

    println("---- NLP block check: destination after mixed replacement ----")
    println("dest has NLP block: ", new_opt_block !== nothing)
    if new_opt_block !== nothing
        println("dest # NLP constraints: ", length(new_opt_block.constraint_bounds))
        println("dest has nonlinear objective: ", new_opt_block.has_objective)
        println("expected # NLP constraints = symbolic + 1 = ", m_sym + 1)
    end

    @assert new_opt_block !== nothing
    @assert length(new_opt_block.constraint_bounds) == m_sym + 1

    print_mixed_problem_summary(opt)

    return index_map, src_backend
end


# ---------------------------------------------------------
# Extract JuMP variable primals from opt using index_map
# ---------------------------------------------------------
function get_primal_from_opt(
    opt,
    index_map,
    xref::AbstractArray{JuMP.VariableRef},
)
    vals = similar(Float64.(zeros(size(xref))))
    for I in eachindex(xref)
        src_vi = JuMP.index(xref[I])
        dst_vi = index_map[src_vi]
        vals[I] = MOI.get(opt, MOI.VariablePrimal(), dst_vi)
    end
    return vals
end


function TSACOPF_sparse_Limited_Memory_SR1(
    instance_dir::String,
    solution_dir::String,
    pf_limit_file::String,
    Surrogate,
    psd,
    opt,
    tau;
    max_iter::Int = 200,
    gamma::Float64 = 0.0,
    approx_type::String = "Sparse",
    r::Int = 6,
)
    st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
    x0 = get_primal_starting_point(psd)

    # Build JuMP ACOPF model
    m, model_data = create_basecase_model(psd, nothing, x0)

    # Attach mixed NLP block
    index_map, src_backend = build_moi_solver_with_TSI_mixed!(
        m,
        psd,
        opt,
        x0,
        Surrogate,
        st_args,
        tau,
        max_iter;
        gamma = gamma,
        approx_type = approx_type,
        r = r,
    )

    # Solve
    MOI.optimize!(opt)

    termination_status = MOI.get(opt, MOI.TerminationStatus())
    total_time = MOI.get(opt, MOI.SolveTimeSec())

    num_iter =
        try
            MOI.get(opt, MOI.BarrierIterations())
        catch
            missing
        end

    p_g_sol = get_primal_from_opt(opt, index_map, m[:p_g])
    q_g_sol = get_primal_from_opt(opt, index_map, m[:q_g])

    base_cost =
        try
            MOI.get(opt, MOI.ObjectiveValue())
        catch
            NaN
        end

    surr_margin =
        TSIConstraint(
            psd,
            Surrogate,
            st_args,
            p_g_sol,
            q_g_sol,
        ) - tau

    grad =
        TSIConstraintPrime(
            psd,
            Surrogate,
            st_args,
            p_g_sol,
            q_g_sol,
            approx_type,
        )

    norm_grad = dot(grad, grad)

    hess_analy = Dict{Int, Dict{String,Any}}()

    return num_iter,
           total_time,
           base_cost,
           surr_margin,
           termination_status,
           norm_grad,
           p_g_sol,
           hess_analy
end