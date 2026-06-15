
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

function TSACOPF(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, tau; spect_info::Union{Nothing, Bool} = false, save_Hess::Union{Nothing, Bool} = false, max_iter::Int = 200, ev_nonzero_tol::Union{Nothing, Float64} = nothing, Hess_approx::Union{Nothing, Bool} = false, gamma::Union{Nothing, Float64} = 0., approx_type::Union{Nothing, String} = nothing, r::Union{Nothing, Int} = nothing, gamma_update::Union{Nothing, Bool} = nothing, gen_type::Union{Nothing, String} = nothing, SR1_hist = nothing, Mel = nothing, diag_pattern = false, save_SR1_hist = false, save_warm_start = false, warm_start::Union{Nothing,Dict}=nothing)

    opt = optimizer_with_attributes(Ipopt.Optimizer,
                                    "sb" => "yes",
                                    "linear_solver" => "ma57",
                                    "ma57_pivot_order" => 2,
                                    "max_iter" =>  max_iter,
                                    "print_timing_statistics" => "no",
                                    # "print_level" => 10,
                                    )

    # Case when there is no surrogate 
    if isnothing(Surrogate) || Surrogate["model_type"] == nothing 
        println("Solving basecase no surrogate. \n")
        return TSACOPF_No_Surrogate(instance_dir, solution_dir, pf_limit_file, psd, opt, gen_type, save_warm_start)

    # Case when the true surrogate Hessian is used
    elseif Hess_approx == false
        println("Solving basecase with surrogate true Hessian. \n")
        return TSACOPF_True_Surrogate_Hessian(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt, tau; spect_info = spect_info, save_Hess = save_Hess, max_iter = max_iter, ev_nonzero_tol = ev_nonzero_tol, gen_type = gen_type)

    # Case when Hessian approximation was requested, but none was given
    elseif Hess_approx && isnothing(approx_type)
        error("Hess_approx was chosen to be true. However no approximation type was given. The choices for approx_type are:
        Full memory SR1: Full
        Limited memory SR1: Limited
        Sparse limited memory SR1: Sparse")

    # Case when full memory SR1 is used for the Hessian
    elseif approx_type == "Full"
        println("Solving basecase with surrogate and full memory SR1. \n")
        return TSACOPF_Full_Memory_SR1(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt, tau; max_iter = max_iter, gamma = gamma, approx_type = approx_type, gen_type)

    # Case when limited memory SR1 is used for the Hessian
    elseif approx_type == "Limited"
        println("Solving basecase with surrogate and limited memory SR1. \n")
        return TSACOPF_Limited_Memory_SR1(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt, tau; max_iter = max_iter, gamma = gamma, approx_type = approx_type, r = r, gamma_update = gamma_update, gen_type = gen_type, save_SR1_hist = save_SR1_hist)

    # Case when sparse limited memory SR1 is used for the Hessian
    else
        #override opt for sparse case
        opt = MOI.instantiate(opt; with_bridge_type = Float64
                            )

        return TSACOPF_sparse_Limited_Memory_SR1(instance_dir, solution_dir, pf_limit_file, Surrogate, psd, opt, tau; max_iter = max_iter, gamma = gamma, approx_type = approx_type, r = r, gamma_update = gamma_update, gen_type = gen_type, SR1_hist = SR1_hist, Mel = Mel, diag_pattern = diag_pattern, save_SR1_hist = save_SR1_hist, warm_start = warm_start)
    end
end

function extract_warm_start(m::JuMP.Model)
    primal = Dict{String, Float64}()

    for v in all_variables(m)
        name = JuMP.name(v)
        if !isempty(name)
            primal[name] = JuMP.value(v)
        end
    end

    duals = Dict{Tuple{DataType, DataType, Int}, Any}()

    for (F, S) in list_of_constraint_types(m)
        cons = all_constraints(m, F, S)
        for (i, con) in enumerate(cons)
            duals[(F, S, i)] = JuMP.dual(con)
        end
    end

    return Dict(
        "primal" => primal,
        "dual"   => duals,
    )
end

function apply_warm_start!(m::JuMP.Model, ws::Dict; mu_init=1e-4)
    println("Applying warm start.")
    primal = ws["primal"]
    duals  = ws["dual"]

    # Apply primal starts by variable name
    for v in all_variables(m)
        name = JuMP.name(v)
        if haskey(primal, name)
            JuMP.set_start_value(v, primal[name])
        end
    end

    # Apply dual starts by constraint type and order
    for (F, S) in list_of_constraint_types(m)
        cons = all_constraints(m, F, S)

        for (i, con) in enumerate(cons)
            key = (F, S, i)

            if haskey(duals, key)
                JuMP.set_dual_start_value(con, duals[key])
            end
        end
    end

    set_optimizer_attribute(m, "warm_start_init_point", "yes")
    set_optimizer_attribute(m, "mu_init", 1e-4)

    return
end

function check_primal_warm_start(m::JuMP.Model, ws::Dict)
    primal = ws["primal"]

    nmatch = 0
    nmiss = 0
    maxdiff = 0.0
    worst_name = ""

    for v in JuMP.all_variables(m)
        name = JuMP.name(v)

        if haskey(primal, name)
            nmatch += 1

            sv = JuMP.start_value(v)
            diff = abs(sv - primal[name])

            if diff > maxdiff
                maxdiff = diff
                worst_name = name
            end
        else
            nmiss += 1
        end
    end

    # println("Matched primal starts: ", nmatch)
    # println("Missing primal starts: ", nmiss)
    # println("Max start mismatch: ", maxdiff)
    # println("Worst variable: ", worst_name)
end

function TSACOPF_No_Surrogate(instance_dir::String, solution_dir::String, pf_limit_file::String, psd, opt, gen_type::Union{Nothing, String} = nothing, save_warm_start = false)
		
	print("done.\nSolving basecase using sparse OPF ...")

    # get primal starting point
    x0 = get_primal_starting_point(psd)

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir=solution_dir)
    
    total_time = MOI.get(m, MOI.SolveTimeSec())

    termination_status = MOI.get(m, MOI.TerminationStatus())
    num_iter = MOI.get(m, MOI.BarrierIterations())

    Surr_Feasibility_margin = NaN
    norm_grad = NaN
        
	print("done. Objective value: \$", round(solution.base_cost, digits=1),
		".\nWriting solution to "*solution_dir*" ... \n")

    if save_warm_start
        return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g, extract_warm_start(m)
    else
        return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g
    end

end

function TSACOPF_True_Surrogate_Hessian(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, opt, tau; spect_info::Bool = false, save_Hess::Bool = false, max_iter::Int = 200, ev_nonzero_tol::Union{Nothing, Float64} = 1e-10, gen_type::Union{Nothing, String} = nothing)
	print("done.\nCreating index lists for TSI constraint ...")
	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"], gen_type)
		
	print("done.\nSolving basecase using sparse OPF ...")
             
    # get primal starting point
    x0 = get_primal_starting_point(psd)

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    if spect_info
        hess_analy =  Dict{Int, Dict{String,Any}}()
    end
    # Used to store spectral data
    iter = 1

    if Surrogate["model_type"] != nothing
        N_gen = st_args["numb_active_gen"]


        function tsif(args...)
            pg_vec = collect(args[1:N_gen])
            qg_vec = collect(args[N_gen+1:2*N_gen])

            TSI = TSIConstraint(psd, Surrogate, st_args, pg_vec, qg_vec)

            # println("TSI = $TSI")

            # return TSIConstraint(psd, Surrogate, st_args, pg_vec, qg_vec)
            return TSI
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
                print_once = 0
                print_once2 = 0
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

                print_once = 0
                print_once2 = 0
                for i = 1:2*N_gen
                    for j = 1:i   
                        h[i, j] = hess[i, j]
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
    end

    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir=solution_dir)
    
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
        
        print("done. Objective value: \$", round(solution.base_cost, digits=1),
            ".\nWriting solution to "*solution_dir*" ... \n")
    
        return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g, hess_analy
    end
        
	print("done. Objective value: \$", round(solution.base_cost, digits=1),
		".\nWriting solution to "*solution_dir*" ... \n")

    return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g
end

function TSACOPF_Full_Memory_SR1(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, opt, tau; max_iter = max_iter, gamma = gamma, approx_type = approx_type, gen_type::Union{Nothing, String} = nothing)
	print("done.\nCreating index lists for TSI constraint ...")
	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"], gen_type)
		
	print("done.\nSolving basecase using sparse OPF ...")
                  
    # get primal starting point
    x0 = get_primal_starting_point(psd)

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    iter = 1

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

                iter += 1
            else
                push!(S, x[2] - x[1])
                push!(Y, g[2] - g[1])
                popfirst!(x)
                popfirst!(g)

                hess = TSIConstraintHessApprox(st_args, B[1], S[end], Y[end], approx_type)

                push!(B, hess)
                popfirst!(B)
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

    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir=solution_dir)
    
    total_time = MOI.get(m, MOI.SolveTimeSec())

    termination_status = MOI.get(m, MOI.TerminationStatus())
    num_iter = MOI.get(m, MOI.BarrierIterations())

    Surr_Feasibility_margin = TSIConstraint(psd, Surrogate, st_args, solution.p_g, solution.q_g) - tau
    grad = TSIConstraintPrime(psd, Surrogate, st_args, solution.p_g, solution.q_g)
    norm_grad = dot(grad, grad)

        
	print("done. Objective value: \$", round(solution.base_cost, digits=1),
		".\nWriting solution to "*solution_dir*" ... \n")

    return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g

end

function TSACOPF_Limited_Memory_SR1(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate, psd, opt, tau; max_iter::Int = 200, gamma::Float64 = 0., approx_type::String = "Sparse", r::Int = 6, gamma_update::Bool=true, gen_type::Union{Nothing, String} = nothing, save_SR1_hist = false)
	print("done.\nCreating index lists for TSI constraint ...")
	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"], gen_type)

    println("Using limited SR1")
		                  
    # get primal starting point
    x0 = get_primal_starting_point(psd)

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    if save_SR1_hist
        SR1_hist =  Dict{String, Vector{Vector{Float64}}}()
    end

    int_iter = true

    # limited memory parameter
    LMp = r   

    N_gen = st_args["numb_active_gen"]

    # For SR1 Hessian approximation
    B0_eye = Matrix{Float64}(I, 2*N_gen, 2*N_gen)
    B0     = gamma * B0_eye
    x = Vector{Vector{Float64}}()
    g = Vector{Vector{Float64}}()
    x_storage = Vector{Vector{Float64}}()
    g_storage = Vector{Vector{Float64}}()
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

            if save_SR1_hist
                push!(x_storage, x_temp)
                push!(g_storage, grad)
            end

            if int_iter
                hess = B0
                int_iter = false
            else
                s = x[2] - x[1]
                y = g[2] - g[1]

                if gamma_update
                    gamma_k = dot(y, y)/dot(s,y)
                    B0_gamma = gamma_k * B0_eye
                else
                    B0_gamma = B0
                end

                push!(S, copy(s))
                push!(Y, copy(y))
                popfirst!(x)
                popfirst!(g)

                try 
                    hess = TSIConstraintHessApprox(st_args, B0_gamma, S, Y, approx_type)
                catch err
                    @warn "SR1 update crashed at iteration $k. Restarting SR1 memory." exception=(err, catch_backtrace())

                    empty!(S)
                    empty!(Y)

                    push!(S, copy(s))
                    push!(Y, copy(y))
                    hess = TSIConstraintHessApprox(st_args, B0_gamma, S, Y, approx_type)
                end

                if length(S) > LMp
                    popfirst!(S)
                    popfirst!(Y)
                end
                if length(x_storage) > LMp+1
                    popfirst!(x_storage)
                    popfirst!(g_storage)
                end
            end

            for i = 1:2*N_gen
                for j = 1:i
                    h[i, j] += hess[i,j]
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

    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir=solution_dir)
    
    if save_SR1_hist
        SR1_hist["S"] = S
        SR1_hist["Y"] = Y
    end

    total_time = MOI.get(m, MOI.SolveTimeSec())

    termination_status = MOI.get(m, MOI.TerminationStatus())
    num_iter = MOI.get(m, MOI.BarrierIterations())

    Surr_Feasibility_margin = TSIConstraint(psd, Surrogate, st_args, solution.p_g, solution.q_g) - tau
    grad = TSIConstraintPrime(psd, Surrogate, st_args, solution.p_g, solution.q_g)
    norm_grad = dot(grad, grad)
        
	print("done. Objective value: \$", round(solution.base_cost, digits=1),
		".\nWriting solution to "*solution_dir*" ... \n")

    if save_SR1_hist
        return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g, SR1_hist
    end

    return num_iter, total_time, solution.base_cost, Surr_Feasibility_margin, termination_status, norm_grad, solution.p_g
end



# ---------------------------------------------------------
# Convert MOI set → NLP bounds pair
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


# ---------------------------------------------------------
# Extract symbolic nonlinear model + constraint bounds
# ---------------------------------------------------------
function build_symbolic_nlp_and_bounds(model::JuMP.Model)
    nlp = JuMP.nonlinear_model(model)

    if nlp === nothing
        return nothing, MOI.NLPBoundsPair[], false
    end

    nl_cons = JuMP.all_nonlinear_constraints(model)
    bounds = MOI.NLPBoundsPair[]

    for cref in nl_cons
        # Get constraint index (handles legacy refs)
        idx = try
            JuMP.index(cref)
        catch
            cref.index
        end

        nl_con = nlp[idx]
        push!(bounds, _bounds_pair(nl_con.set))
    end

    # Objective is quadratic → not part of NLP block
    has_nl_obj = false

    return nlp, bounds, has_nl_obj
end


# ---------------------------------------------------------
# TSI constraint value: g(x) = TSI(pg,qg) - τ
# ---------------------------------------------------------
function TSI_g_bb(x, p_idx, q_idx, psd, Surrogate, st_args)
    pg = x[p_idx]
    qg = x[q_idx]
    return TSIConstraint(psd, Surrogate, st_args, pg, qg) - st_args["tau"]
end


# ---------------------------------------------------------
# TSI gradient wrt [pg; qg] (local ordering)
# ---------------------------------------------------------
function TSI_g_bb_grad(x, p_idx, q_idx, psd, Surrogate, st_args)
    pg = x[p_idx]
    qg = x[q_idx]

    return TSIConstraintPrime(psd, Surrogate, st_args, pg, qg)
end


# ---------------------------------------------------------
# TSI Hessian via SR1 approximation (local ordering)
# ---------------------------------------------------------
function TSI_g_bb_hess_values(st_args, B0, S, Y, approx_type::String)
    return TSIConstraintHessApprox(st_args, B0, S, Y, approx_type)
end


# ---------------------------------------------------------
# Exact TSI Hessian (local ordering)
# ---------------------------------------------------------
function TSI_g_bb_hess_values(x, p_idx, q_idx, psd, Surrogate, st_args)
    pg = x[p_idx]
    qg = x[q_idx]

    return TSIConstraintPrimePrime(psd, Surrogate, st_args, pg, qg)
end

# =========================================================
# Mixed NLP evaluator:
#   H = symbolic (JuMP AD) + 1 black-box TSI constraint
# =========================================================
struct MixedTSIEvaluator{E<:MOI.AbstractNLPEvaluator} <: MOI.AbstractNLPEvaluator
    # Symbolic evaluator (JuMP AD)
    sym::E
    has_sym_objective::Bool

    # Problem dimensions
    n::Int
    m_sym::Int
    N_gen::Int

    # Indices of pg, qg in solver ordering
    p_idx::Vector{Int}
    q_idx::Vector{Int}

    # Symbolic sparsity patterns
    jac_sym_struct::Vector{Tuple{Int,Int}}
    hess_sym_struct::Vector{Tuple{Int,Int}}

    # TSI Jacobian structure (single dense row)
    bb_grad_cols::Vector{Int}
    bb_grad_offset::Int

    # TSI Hessian sparsity (local + global)
    bb_rows_local::Vector{Int}
    bb_cols_local::Vector{Int}
    bb_rows::Vector{Int}
    bb_cols::Vector{Int}

    # Problem data
    psd
    Surrogate
    st_args

    # SR1 parameters
    approx_type::String
    LMp::Int
    B0::SparseMatrixCSC{Float64,Int}
    I_gamma::SparseMatrixCSC{Float64,Int}

    # SR1 history (limited memory)
    x_hist::Vector{Vector{Float64}}
    g_hist::Vector{Vector{Float64}}
    g_hist_temp::Vector{Vector{Float64}}
    S::Vector{Vector{Float64}}
    Y::Vector{Vector{Float64}}
    gamma_k::Vector{Float64}
    Bk::Vector{Matrix{Float64}}

    # SR1 safeguards
    r_stop::Float64
    gamma_update::Bool
end

function MOI.features_available(d::MixedTSIEvaluator)
    return MOI.features_available(d.sym)
end

function MOI.initialize(d::MixedTSIEvaluator, requested_features)
    MOI.initialize(d.sym, requested_features)
    return
end

function MOI.eval_objective(d::MixedTSIEvaluator, x)
    return MOI.eval_objective(d.sym, x)
end

function MOI.eval_objective_gradient(d::MixedTSIEvaluator, g, x)
    MOI.eval_objective_gradient(d.sym, g, x)
    return
end

# ---------------------------------------------------------
# Evaluate constraints:
#   g = [g_symbolic; g_TSI]
# ---------------------------------------------------------
function MOI.eval_constraint(d::MixedTSIEvaluator, g, x)
    MOI.eval_constraint(d.sym, view(g, 1:d.m_sym), x)

    g[d.m_sym + 1] = TSI_g_bb(
        x, d.p_idx, d.q_idx, d.psd, d.Surrogate, d.st_args
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
# Jacobian:
#   J = [J_symbolic;
#        ∇TSI (dense over pg,qg)]
# ---------------------------------------------------------
function MOI.eval_constraint_jacobian(d::MixedTSIEvaluator, J, x)

    ns = length(d.jac_sym_struct)

    # Symbolic part
    MOI.eval_constraint_jacobian(d.sym, view(J, 1:ns), x)

    # TSI gradient
    grad_full = TSI_g_bb_grad(
        x, d.p_idx, d.q_idx, d.psd, d.Surrogate, d.st_args
    )

    @assert length(grad_full) == length(d.bb_grad_cols)

    # Store gradient for SR1
    push!(d.g_hist_temp, copy(grad_full))

    # Fill TSI row
    for k in eachindex(grad_full)
        J[ns + k] = grad_full[k]
    end

    return
end

# ---------------------------------------------------------
# Hessian structure of the Lagrangian
# concatenate symbolic Hessian structure + TSI Hessian structure
# ---------------------------------------------------------
function MOI.hessian_lagrangian_structure(d::MixedTSIEvaluator)
    H = Vector{Tuple{Int,Int}}()
    append!(H, d.hess_sym_struct)
    append!(H, collect(zip(d.bb_rows, d.bb_cols)))
    return H
end

# ---------------------------------------------------------
# Hessian of Lagrangian:
#   H = σ∇²f + Σ μ_i ∇²g_i + μ_TSI ∇²TSI
# ---------------------------------------------------------
function MOI.eval_hessian_lagrangian(d::MixedTSIEvaluator, Hval, x, σ, μ)

    fill!(Hval, 0.0)

    ns = length(d.hess_sym_struct)

    # Symbolic Hessian
    MOI.eval_hessian_lagrangian(
        d.sym,
        view(Hval, 1:ns),
        x,
        σ,
        view(μ, 1:d.m_sym),
    )

    μ_tsi = μ[d.m_sym + 1]

    # Local variables [pg; qg]
    x_local = if d.Surrogate["model_type"] == "CNF"
        vcat(x[d.p_idx], x[d.q_idx])
    else
        vcat(x[d.p_idx], zeros(length(d.q_idx)))
    end

    # SR1 update logic
    grad = d.g_hist_temp[end]
    popfirst!(d.g_hist_temp)

    push!(d.x_hist, copy(x_local))
    push!(d.g_hist, grad)

    if length(d.x_hist) == 1 || length(d.g_hist) < 2
        B = d.B0
    else
        s = d.x_hist[2] - d.x_hist[1]
        y = d.g_hist[2] - d.g_hist[1]

        gamma_k = d.gamma_k[end]
        if d.gamma_update
            # gamma_k = dot(y,y)/dot(s,y)
            gamma_k = dot(s,y)/dot(s,s)
        end
        push!(d.gamma_k, gamma_k)

        popfirst!(d.x_hist)
        popfirst!(d.g_hist)

        push!(d.S, s)
        push!(d.Y, y)

        if d.st_args["top_idx"] == 0 
            B = d.gamma_k[end] * d.I_gamma

        else
            # SR1 safeguard condition
            yBs = y - d.Bk[1] * s
            if abs(dot(s, yBs)) >= d.r_stop * norm(s) * norm(yBs)
                try 
                    B = TSI_g_bb_hess_values(
                        d.st_args,
                        d.gamma_k[end] * d.I_gamma,
                        d.S,
                        d.Y,
                        "Sparse",
                    )
                catch err
                    @warn "SR1 update crashed at iteration $k. Restarting SR1 memory." exception=(err, catch_backtrace())

                    empty!(d.S)
                    empty!(d.Y)

                    push!(d.S, copy(s))
                    push!(d.Y, copy(y))
                    B = TSI_g_bb_hess_values(
                        d.st_args,
                        d.gamma_k[end] * d.I_gamma,
                        d.S,
                        d.Y,
                        "Sparse",
                    )
                end
                push!(d.Bk, B)
                popfirst!(d.Bk)
            else
                B = d.Bk[end]
                println("SR1 was skipped")
            end
        end

        if length(d.S) >= d.LMp 
            popfirst!(d.S)
            popfirst!(d.Y)
        end
    end

    # Extract sparse entries
    vals = [B[i,j] for (i,j) in zip(d.bb_rows_local, d.bb_cols_local)]

    for k in eachindex(vals)
        Hval[ns + k] = μ_tsi * vals[k]
    end

    return
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
    gamma_update::Bool = true,
    SR1_hist = nothing,
    Mel = nothing,
    diag_pattern = false,
    warm_start::Union{Nothing,Dict}=nothing
)
    src_backend = JuMP.backend(m)

    # -----------------------------------------------------
    # 1) Copy JuMP backend into destination optimizer
    # -----------------------------------------------------
    index_map = MOI.copy_to(opt, src_backend)

    if warm_start !== nothing
        primal = warm_start["primal"]
    
        for v in JuMP.all_variables(m)
            name = JuMP.name(v)
            if haskey(primal, name)
                src_vi = JuMP.index(v)
                dest_vi = index_map[src_vi]
                MOI.set(opt, MOI.VariablePrimalStart(), dest_vi, primal[name])
            end
        end
    
        MOI.set(opt, MOI.RawOptimizerAttribute("warm_start_init_point"), "yes")
        MOI.set(opt, MOI.RawOptimizerAttribute("mu_init"), 1e-4)

        duals = warm_start["dual"]

        for (F, S) in JuMP.list_of_constraint_types(m)
            cons = JuMP.all_constraints(m, F, S)

            for (i, con) in enumerate(cons)
                key = (F, S, i)

                if haskey(duals, key)
                    src_ci = JuMP.index(con)
                    dest_ci = index_map[src_ci]
                    MOI.set(opt, MOI.ConstraintDualStart(), dest_ci, duals[key])
                end
            end
        end
    end
    

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

    for k in eachindex(hess_sym_struct)
        row = hess_sym_struct[k][1]
        col = hess_sym_struct[k][2]
    end

    # -----------------------------------------------------
    # 4) Recover p_g and q_g destination indices
    # -----------------------------------------------------
    p_src = JuMP.index.(m[:p_g])
    q_src = JuMP.index.(m[:q_g])

    p_dest = [index_map[vi].value for vi in p_src]
    q_dest = [index_map[vi].value for vi in q_src]

    n = MOI.get(opt, MOI.NumberOfVariables())
    N_gen = st_args["numb_active_gen"]

    # -----------------------------------------------------
    # 5) Fix TSI sparsity pattern once
    # -----------------------------------------------------
    pg = JuMP.start_value.(m[:p_g])
    qg = JuMP.start_value.(m[:q_g])

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

    if diag_pattern 

        if Surrogate["model_type"] == "CNF"
            diag_n = 2 * N_gen
            
        else
            diag_n = N_gen
        end

        top = 0

        I_n = Matrix{Float64}(I, diag_n, diag_n)

        rows_local, cols_local, vals = findnz(sparse(I_n))
    else
        # SR1 to choose sparse pattern for Hessian
        if SR1_hist == nothing
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
        else
            println("SR1_hist was given.")

            S = SR1_hist["S"]
            Y = SR1_hist["Y"]
        end

        rows_local, cols_local, top = TSIConstraintHessApprox(
            st_args,
            B0,
            S,
            Y,
            "Sparse_pattern",
        )

        
    end

    st_args["top_idx"] = top

    # Force lower-triangular canonical ordering
    for k in eachindex(rows_local)
        if rows_local[k] < cols_local[k]
            rows_local[k], cols_local[k] = cols_local[k], rows_local[k]
        end
    end

    perm = sortperm(collect(zip(rows_local, cols_local)))
    rows_local = rows_local[perm]
    cols_local = cols_local[perm]

    idx_map = vcat(p_dest, q_dest)
    rows_global = idx_map[rows_local]
    cols_global = idx_map[cols_local]

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
        1e-8,
        gamma_update,
    )

    # -----------------------------------------------------
    # 7) Attach combined NLP block
    # -----------------------------------------------------
    new_bounds = copy(bounds_sym)
    push!(new_bounds, MOI.NLPBoundsPair(0.0, Inf))

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

    @assert new_opt_block !== nothing
    @assert length(new_opt_block.constraint_bounds) == m_sym + 1

    return index_map, src_backend, mixed_eval
end


# ---------------------------------------------------------
# Extract JuMP variable primals from opt using index_map
# ---------------------------------------------------------
function get_primal_from_opt(
    opt,
    index_map,
    xref,
)
    vals = zeros(Float64, size(xref))

    if isempty(xref)
        return vals
    end

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
    gamma_update::Bool = true, 
    gen_type::Union{Nothing, String} = nothing,
    SR1_hist = nothing,
    Mel = nothing,
    diag_pattern = false,
    save_SR1_hist = false,
    warm_start::Union{Nothing,Dict}=nothing
)
    st_args = load_case(psd, pf_limit_file, Surrogate["model_type"], gen_type)
    st_args["tau"] = tau
    x0 = get_primal_starting_point(psd)

    # Build JuMP ACOPF model
    m, model_data = create_basecase_model(psd, nothing, x0)

    t0 = time()
    # Attach mixed NLP block
    index_map, src_backend, mixed_eval = build_moi_solver_with_TSI_mixed!(
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
        gamma_update = gamma_update,
        SR1_hist = SR1_hist, 
        Mel = Mel,
        diag_pattern = diag_pattern,
        warm_start = warm_start
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

    if solution_dir !== nothing
        v_n_sol      = get_primal_from_opt(opt, index_map, m[:v_n])
        theta_n_sol  = get_primal_from_opt(opt, index_map, m[:theta_n])

        b_s = get_primal_from_opt(opt, index_map, m[:b_s])

        p_li_sol = get_primal_from_opt(opt, index_map, m[:p_li])
        q_li_sol = get_primal_from_opt(opt, index_map, m[:q_li])
        p_ti_sol = get_primal_from_opt(opt, index_map, m[:p_ti])
        q_ti_sol = get_primal_from_opt(opt, index_map, m[:q_ti])

        pslackm_n_sol = get_primal_from_opt(opt, index_map, m[:pslackm_n])
        pslackp_n_sol = get_primal_from_opt(opt, index_map, m[:pslackp_n])
        qslackm_n_sol = get_primal_from_opt(opt, index_map, m[:qslackm_n])
        qslackp_n_sol = get_primal_from_opt(opt, index_map, m[:qslackp_n])

        sslack_li_sol = get_primal_from_opt(opt, index_map, m[:sslack_li])
        sslack_ti_sol = get_primal_from_opt(opt, index_map, m[:sslack_ti])
    
        solution = BasecaseSolution(psd, v_n_sol, theta_n_sol,
            convert(Vector{Float64}, b_s), p_g_sol, q_g_sol,
            0.0, 0.0)

        save_Mixed_MOI_outputs(
            solution_dir,
            psd,
            solution;
            v_n = v_n_sol,
            theta_n = theta_n_sol,
            p_li = p_li_sol,
            q_li = q_li_sol,
            p_ti = p_ti_sol,
            q_ti = q_ti_sol,
            b_s = b_s,
            p_g = p_g_sol,
            q_g = q_g_sol,
            pslackm_n = pslackm_n_sol,
            pslackp_n = pslackp_n_sol,
            qslackm_n = qslackm_n_sol,
            qslackp_n = qslackp_n_sol,
            sslack_li = sslack_li_sol,
            sslack_ti = sslack_ti_sol,
        )
    end
    

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

    println("done. Objective value: \$", round(base_cost, digits=1), ".\nWriting solution to "*solution_dir*" ... \n")
    
    if save_SR1_hist
        SR1_hist =  Dict{String, Vector{Vector{Float64}}}()
        SR1_hist["S"] = mixed_eval.S
        SR1_hist["Y"] = mixed_eval.Y
        return num_iter,
            total_time,
            base_cost,
            surr_margin,
            termination_status,
            norm_grad,
            p_g_sol,
            SR1_hist
    else
        return num_iter,
            total_time,
            base_cost,
            surr_margin,
            termination_status,
            norm_grad,
            p_g_sol
    end
end