
using Ipopt, JuMP, Printf
using SCACOPFSubproblems

using MAT
using LinearAlgebra
using SparseArrays
using Random
using Statistics   # for mean

jl_lib = string(path_to_tsslope,"/tsslope-pump-jl")
include(string(jl_lib,"/load_case.jl"))

const CACHE = Dict{UInt64, Any}()

function print_stats(name, x)
    println(
        rpad(name, 8), ": ",
        "min = ", minimum(x),
        ", max = ", maximum(x),
        ", mean = ", mean(x)
    )
end

function TSACOPF(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate)
	print("Reading instance from "*instance_dir*" ... ")
    psd = SCACOPFdata(instance_dir)
	print("done.\nCreating index lists for TSI constraint ...")
	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
		
	print("done.\nSolving basecase using sparse OPF ...")
	opt = optimizer_with_attributes(Ipopt.Optimizer,
		                            # "linear_solver" => "ma57",
		                            "sb" => "yes")
                  
    # get primal starting point
    x0 = get_primal_starting_point(psd)

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    if Surrogate["model_type"] != nothing
        N_gen = st_args["numb_active_gen"]
        function tsif(args...)
            pg_vec = collect(args[1:N_gen])
            qg_vec = collect(args[N_gen+1:2*N_gen])
    
            print("TSI constrint: ", TSIConstraint(psd, Surrogate, st_args, pg_vec, qg_vec),"\n")
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
            # st_args["PL"] = -st_args["PL"]
            st_args["QL"] = -st_args["QL"]
            @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= 0.0 )
        else
            @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= 0.80 ) # change to 0.75
        end
        
        if !ispath(solution_dir)
            mkpath(solution_dir)
        end
    end

    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir="output")

    if !ispath(solution_dir)
        mkpath(solution_dir)
    end    
    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir = solution_dir)

    # print("done. Objective value: \$", round(solution.base_cost, digits=1),
	# 	".\nWriting solution to "*solution_dir*" ... \n")
end

function TSACOPF_Loop(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate; 
                 sample_id::Int = 1, total_samples::Int = 1)
	print("Reading instance from "*instance_dir*" ... ")
    psd = SCACOPFdata(instance_dir)
    loads_ori = deepcopy(psd.loads)

    # Number of loads (rows)
    nloads = size(psd.loads, 1)

    st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
    N_gen = st_args["numb_active_gen"]    
    
    function tsif(args...)
        pg_vec = collect(args[1:N_gen])
        qg_vec = collect(args[N_gen+1:2*N_gen])

        # print("TSI constrint: ", TSIConstraint(psd, Surrogate, st_args, pg_vec, qg_vec),"\n")
        return TSIConstraint(psd, Surrogate, st_args, pg_vec, qg_vec)
        # return 1.
    end

    function tsig(g::AbstractVector, args...)
        pg_vec = collect(args[1:N_gen])
        qg_vec = collect(args[N_gen+1:2*N_gen])
        grad = TSIConstraintPrime(psd, Surrogate, st_args, pg_vec, qg_vec)
        # print("TSI gradient constrint: ", maximum(abs.(grad)), "\n")
        g[1:2*N_gen] .= grad
        # g[1:2*N_gen] .= 0. .*grad
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
            for i = 1:2*N_gen
                for j = 1:i
                    h[i, j] = hess[i,j]
                    # h[i, j] = 0.
                end
            end
            CACHE[hsh] = h
        end     
    end 

    # ====================
    # one randomized sample
    # ====================

    case_id = sample_id

    # reset to original
    psd.loads[!, :PL] .= loads_ori[!, :PL]
    psd.loads[!, :QL] .= loads_ori[!, :QL]

    ### change pl, ql with some random numbers
    # random multipliers
    rng = MersenneTwister(sample_id)   # unique RNG for this sample
    mult_PL = 0.85 .+ 0.30 .* rand(rng, nloads)
    mult_QL = 0.85 .+ 0.30 .* rand(rng, nloads)

    psd.loads[!, :PL] .*= mult_PL
    psd.loads[!, :QL] .*= mult_QL
                
    if size(psd.loads, 1) > 0
        BusLoad = indexin(psd.loads[!,:I], psd.N[!,:Bus])
        for l = 1:size(psd.loads, 1)
            if psd.loads[l,:STATUS] == 1
                psd.N[BusLoad[l],:Pd] = 0.0
                psd.N[BusLoad[l],:Qd] = 0.0
            end
        end
        for l = 1:size(psd.loads, 1)
            if BusLoad[l] == nothing
                error("bus ", psd.loads[l,:I], " of load ", l, " not found.")
            end
            if psd.loads[l,:STATUS] == 1
                psd.N[BusLoad[l],:Pd] += psd.loads[l,:PL]/psd.MVAbase
                psd.N[BusLoad[l],:Qd] += psd.loads[l,:QL]/psd.MVAbase
            end
        end
    end        
        
    st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
    N_gen = st_args["numb_active_gen"]
        
    #print("done.\nCreating index lists for TSI constraint ...")
        
   	println("done.\nSolving basecase using sparse OPF $case_id / $total_samples ...")
    opt = optimizer_with_attributes(Ipopt.Optimizer,
                                    # "linear_solver" => "ma57",
                                    "print_level" => 0,
                                    "sb" => "yes")                  

    # get primal starting point
    x0 = get_primal_starting_point(psd)

    # create model
    m, model_data = create_basecase_model(psd, opt, x0)

    if Surrogate["model_type"] != "None"            
        register(m, :tsicon, 2*N_gen, tsif, tsig, tsih)

        print_stats("PL",  st_args["PL"])
        print_stats("QL",  st_args["QL"])
        print_stats("Pd",  psd.N[:, :Pd])
        print_stats("Qd",  psd.N[:, :Qd])
        
        
        if Surrogate["model_type"] == "CNF"
            st_args["PL"] = st_args["PL"]
            st_args["QL"] = -st_args["QL"]
            @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= 0.0 )
        else
            @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= 0.5 )
        end
    end

    output_dir = joinpath(solution_dir, "folder_$(case_id)")
    if !ispath(output_dir)
        mkpath(output_dir)
    end    
    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir = output_dir)

    print("done. Objective value: \$", round(solution.base_cost, digits=1),
	 	".\nWriting solution to "*output_dir*" ... \n")
end