
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

    if approx_type == "Sparse"
        opt = MOI.instantiate(optimizer_with_attributes(
                                Ipopt.Optimizer,
                                "sb" => "yes",
                                "max_iter" => max_iter,
                                "print_timing_statistics" => "yes",
                                );
                                with_bridge_type = Float64
                            )
    else
        opt = optimizer_with_attributes(Ipopt.Optimizer,
                                        "sb" => "yes",
                                        # "linear_solver" => "ma57",
                                        "max_iter" =>  max_iter,
                                        "print_timing_statistics" => "yes",
                                        )
    end

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

    # Case when full memory SR1 is used for the Hessian
    elseif approx_type == "Full"
        return TSACOPF_Full_Memory_SR1(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt, tau; max_iter = max_iter, gamma = gamma, approx_type = approx_type)

    # Case when limited memory SR1 is used for the Hessian
    elseif approx_type == "Limited"
        println("In sparse")
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

                iter += 1
            end

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
    println(r)
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

    for i = 1:r
        println("S[$i]: max=", maximum(S[i]), 
                " min=", minimum(S[i]))
        println("Y[$i]: max=", maximum(Y[i]), 
                " min=", minimum(Y[i]))
    end
    
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

const MOI = JuMP.MOI

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
    return TSIConstraint(psd, Surrogate, st_args, pg, qg)
end


# ---------------------------------------------------------
# Sparse TSI gradient in solver-variable ordering
#
# Returns:
#   idx_global :: Vector{Int}
#   vals       :: Vector{Float64}
#   grad_full  :: Vector{Float64}   (length 2N_gen, local ordering [pg; qg])
# ---------------------------------------------------------
function TSI_g_bb_grad_sparse(
    x::Vector{Float64},
    p_idx::Vector{Int},
    q_idx::Vector{Int},
    psd,
    Surrogate,
    st_args,
    approx_type::String,
)
    pg = x[p_idx]
    qg = x[q_idx]

    idx_local, vals, grad_full = TSIConstraintPrime(
        psd,
        Surrogate,
        st_args,
        pg,
        qg,
        approx_type,
    )

    # local ordering is [pg; qg]
    idx_map = vcat(p_idx, q_idx)
    idx_global = idx_map[idx_local]

    return idx_global, vals, grad_full
end


# ---------------------------------------------------------
# Sparse TSI Hessian in solver-variable ordering
#
# Returns:
#   rows_global :: Vector{Int}
#   cols_global :: Vector{Int}
#   vals        :: Vector{Float64}
#   top         :: Any
# ---------------------------------------------------------
function TSI_g_bb_hess_sparse(
    x::Vector{Float64},
    p_idx::Vector{Int},
    q_idx::Vector{Int},
    st_args,
    B0,
    S,
    Y,
    approx_type::String,
)
    rows_local, cols_local, vals, top = TSIConstraintHessApprox(
        st_args,
        B0,
        S,
        Y,
        approx_type,
    )

    idx_map = vcat(p_idx, q_idx)
    rows_global = idx_map[rows_local]
    cols_global = idx_map[cols_local]

    return rows_global, cols_global, vals, top
end

# ---------------------------------------------------------
# Evaluator for one black-box nonlinear constraint:
#       TSI(pg, qg) >= tau
# ---------------------------------------------------------
mutable struct TSIEvaluator <: MOI.AbstractNLPEvaluator
    n::Int
    N_gen::Int

    p_idx::Vector{Int}
    q_idx::Vector{Int}

    bb_grad_idx::Vector{Int}
    bb_hess_struct::Vector{Tuple{Int,Int}}
    bb_hess_pos::Dict{Tuple{Int,Int},Int}

    psd
    Surrogate
    st_args

    approx_type::String
    LMp::Int
    B0

    S::Vector{Vector{Float64}}
    Y::Vector{Vector{Float64}}

    x_hist::Vector{Vector{Float64}}
    g_hist::Vector{Vector{Float64}}
end

function MOI.features_available(::TSIEvaluator)
    return [:Grad, :Jac, :Hess]
end

function MOI.initialize(d::TSIEvaluator, requested_features)
    println("TSI evaluator initialized with ", requested_features)
    return
end

# We do NOT provide a nonlinear objective.
# These are harmless to define anyway.
function MOI.eval_objective(d::TSIEvaluator, x)
    return 0.0
end

function MOI.eval_objective_gradient(d::TSIEvaluator, g, x)
    fill!(g, 0.0)
    return
end

# ---------------------------------------------------------
# Nonlinear constraint values
# Only one NLP constraint, so g has length 1.
# ---------------------------------------------------------
function MOI.eval_constraint(d::TSIEvaluator, g, x)
    println("Inside eval_constraint")
    g[1] = TSI_g_bb(
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
# One row only, with sparse columns bb_grad_idx
# ---------------------------------------------------------
function MOI.jacobian_structure(d::TSIEvaluator)
    return [(1, j) for j in d.bb_grad_idx]
end

# ---------------------------------------------------------
# Jacobian values
# ---------------------------------------------------------
function MOI.eval_constraint_jacobian(d::TSIEvaluator, J, x)
    println("Inside eval_constraint_jacobian")

    idx, vals, grad_full = TSI_g_bb_grad_sparse(
        x,
        d.p_idx,
        d.q_idx,
        d.psd,
        d.Surrogate,
        d.st_args,
        d.approx_type,
    )

    @assert idx == d.bb_grad_idx

    push!(d.g_hist, grad_full)
    if length(d.g_hist) > 2
        popfirst!(d.g_hist)
    end

    for k in eachindex(vals)
        J[k] = vals[k]
    end
    return
end

# ---------------------------------------------------------
# Hessian structure of the Lagrangian
# Only contribution is μ[1] * ∇² TSI
# ---------------------------------------------------------
function MOI.hessian_lagrangian_structure(d::TSIEvaluator)
    return d.bb_hess_struct
end

# ---------------------------------------------------------
# Hessian values
# ---------------------------------------------------------
function MOI.eval_hessian_lagrangian(d::TSIEvaluator, Hval, x, σ, μ)
    println("Inside eval_hessian_lagrangian")

    fill!(Hval, 0.0)

    μ_tsi = μ[1]

    # local state vector for SR1 memory = [pg; qg]
    x_local = vcat(x[d.p_idx], x[d.q_idx])
    push!(d.x_hist, copy(x_local))
    if length(d.x_hist) > 2
        popfirst!(d.x_hist)
    end

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    if length(d.x_hist) == 1
        rows0, cols0, vals0 = findnz(sparse(d.B0))
        idx_map = vcat(d.p_idx, d.q_idx)
        rows = idx_map[rows0]
        cols = idx_map[cols0]
        vals = vals0
    else
        s = d.x_hist[2] - d.x_hist[1]
        y = d.g_hist[2] - d.g_hist[1]

        push!(d.S, s)
        push!(d.Y, y)

        rows, cols, vals, top = TSI_g_bb_hess_sparse(
            x,
            d.p_idx,
            d.q_idx,
            d.st_args,
            d.B0,
            d.S,
            d.Y,
            d.approx_type,
        )

        push!(d.top_indices, top)

        if length(d.S) > d.LMp
            popfirst!(d.S)
            popfirst!(d.Y)
        end
    end

    # scatter into the fixed Hessian structure
    for k in eachindex(vals)
        p = d.bb_hess_pos[(rows[k], cols[k])]
        Hval[p] += μ_tsi * vals[k]
    end

    return
end

# ---------------------------------------------------------
# Build an MOI/Ipopt solver from the JuMP basecase model,
# copy the ACOPF model into it, and attach the TSI NLP block.
#
# Returns:
#   opt       :: MOI optimizer
#   index_map :: MOI index map from source backend -> opt
#   src_backend
# ---------------------------------------------------------
function build_moi_solver_with_TSI!(
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
    println("Attaching TSI evaluator on MOI solver")

    # -----------------------------------------------------
    # 2) Copy JuMP backend into solver model
    # -----------------------------------------------------
    src_backend = JuMP.backend(m)
    index_map = MOI.copy_to(opt, src_backend)

    # -----------------------------------------------------
    # 3) Recover p_g and q_g destination indices in solver
    # -----------------------------------------------------
    p_src = JuMP.index.(m[:p_g])
    q_src = JuMP.index.(m[:q_g])

    p_dest = [index_map[vi].value for vi in p_src]
    q_dest = [index_map[vi].value for vi in q_src]

    println("p_g indices = ", p_dest[1:min(end,5)], " ...")
    println("q_g indices = ", q_dest[1:min(end,5)], " ...")

    n = MOI.get(opt, MOI.NumberOfVariables())
    N_gen = st_args["numb_active_gen"]

    # -----------------------------------------------------
    # 4) Fix the sparse pattern once
    # -----------------------------------------------------

    # fixed pattern for gradient
    pg = x0[:p_g]
    qg = x0[:q_g]

    idx_local = 1:(length(pg) + length(qg))

    idx_map = vcat(p_dest, q_dest)
    bb_grad_idx = idx_map[idx_local]


    # Fixed pattern for Hessian
    B0 =
        gamma == 0.0 ?
        spzeros(2 * N_gen, 2 * N_gen) :
        gamma * sparse(I, 2 * N_gen, 2 * N_gen)


    x = Vector{Vector{Float64}}()
    g = Vector{Vector{Float64}}()
    S = Vector{Vector{Float64}}()
    Y = Vector{Vector{Float64}}()

    for i = 1:(r+1)
        x_temp = zeros(2*N_gen)
        pg_noise = pg + rand(Uniform(-1e-4, 1e-4), length(pg))
        x_temp[1:N_gen] .= pg_noise 
        if Surrogate["model_type"] == "CNF"
            qg_noise = qg + rand(Uniform(-1e-4, 1e-4), length(qg))
            x_temp[N_gen+1:2*N_gen] .= qg_noise
        end
        grad = TSIConstraintPrime(psd, Surrogate, st_args, pg_noise, qg_noise)
        push!(x, x_temp)
        push!(g, grad)
        if i > 1
            push!(S, x[2] - x[1])
            push!(Y, g[2] - g[1])
            popfirst!(x)
            popfirst!(g)
        end
    end

    rows_local, cols_local, top = TSIConstraintHessApprox(
        st_args,
        B0,
        S,
        Y,
        "Sparse_pattern",
    )

    st_args["top_idx"] = top

    idx_map = vcat(p_dest, q_dest)
    rows0 = idx_map[rows_local]
    cols0 = idx_map[cols_local]


    bb_hess_struct = collect(zip(rows0, cols0))
    bb_hess_pos = Dict{Tuple{Int,Int},Int}()
    for (k, rc) in enumerate(bb_hess_struct)
        bb_hess_pos[rc] = k
    end

    # -----------------------------------------------------
    # 5) Create evaluator
    # -----------------------------------------------------
    evaluator = TSIEvaluator(
        n,
        N_gen,
        p_dest,
        q_dest,
        bb_grad_idx,
        bb_hess_struct,
        bb_hess_pos,
        psd,
        Surrogate,
        st_args,
        approx_type,
        r,
        Matrix(B0),
        Vector{Vector{Float64}}(),
        Vector{Vector{Float64}}(),
        Vector{Vector{Float64}}(),
        Vector{Vector{Float64}}(),
    )

    # -----------------------------------------------------
    # 6) Attach nonlinear block
    # One nonlinear constraint: TSI(pg,qg) >= tau
    # Objective is already copied from the JuMP model,
    # so has_objective = false.
    # -----------------------------------------------------
    bounds = [MOI.NLPBoundsPair(tau, Inf)]
    nlp_block = MOI.NLPBlockData(bounds, evaluator, false)

    MOI.set(opt, MOI.NLPBlock(), nlp_block)

    block = MOI.get(opt, MOI.NLPBlock())
    println("NLP block attached: ", block !== nothing)
    println("Number of NLP constraints: ", length(block.constraint_bounds))
    println("Constraint bounds: ", block.constraint_bounds)

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
    println("Inside the solver for sparse")

    st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
    x0 = get_primal_starting_point(psd)

    # -----------------------------------------------------
    # 1) Build JuMP ACOPF model exactly as before
    # -----------------------------------------------------
    m, model_data = create_basecase_model_TSI(psd, x0)

    println("Constraint types in original JuMP model:")
    for (F,S) in JuMP.list_of_constraint_types(m)
        println("F = ", F, "   S = ", S,
                "   count = ", JuMP.num_constraints(m, F, S))
    end

    # -----------------------------------------------------
    # 2) Build MOI solver and attach TSI evaluator
    # -----------------------------------------------------
    index_map, src_backend = build_moi_solver_with_TSI!(
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

    # -----------------------------------------------------
    # 3) Solve directly with MOI
    # -----------------------------------------------------
    MOI.optimize!(opt)

    termination_status = MOI.get(opt, MOI.TerminationStatus())
    total_time = MOI.get(opt, MOI.SolveTimeSec())

    num_iter =
        try
            MOI.get(opt, MOI.BarrierIterations())
        catch
            missing
        end

    # -----------------------------------------------------
    # 4) Recover solution pieces you care about
    # -----------------------------------------------------
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

    _, _, grad =
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