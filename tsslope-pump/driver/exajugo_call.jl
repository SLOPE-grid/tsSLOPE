
using Ipopt, JuMP, Printf
using SCACOPFSubproblems

using MAT
using LinearAlgebra
using SparseArrays

jl_lib = string(path_to_tsslope,"/tsslope-pump-jl")
include(string(jl_lib,"/load_case.jl"))

function TSACOPF(instance_dir::String, solution_dir::String, pf_limit_file::String, Surrogate)
	print("Reading instance from "*instance_dir*" ... ")
    psd = SCACOPFdata(instance_dir)
	print("done.\nCreating index lists for TSI constraint ...")
	st_args = load_case(psd, pf_limit_file, Surrogate["model_type"])
		
	print("done.\nSolving basecase using sparse OPF ...")
	opt = optimizer_with_attributes(Ipopt.Optimizer,
		                            #"linear_solver" => "ma57",
		                            "sb" => "yes")
                                
    # get primal starting point
    x0 = get_primal_starting_point(psd)

    N_gen = st_args["numb_active_gen"]
    function tsif(args...)
        pg_vec = collect(args[1:N_gen])
        qg_vec = collect(args[N_gen+1:2*N_gen])

        gen_idx = st_args["gen_idx"] .+1
        PG_full = zeros(st_args["numb_gen"])
        QG_full = zeros(st_args["numb_gen"])
        
        PG_full[gen_idx] = pg_vec
        QG_full[gen_idx] = qg_vec

        # println("Pg = [", join(PG_full, ", "), "]")
        # println("Qg = [", join(QG_full, ", "), "]")

        # print("TSI constrint: ", TSIConstraint(psd, Surrogate, st_args, pg_vec, qg_vec),"\n")
        # return TSIConstraint(psd, Surrogate, st_args, pg_vec, qg_vec)
        return 1.
    end

    function tsig(g::AbstractVector, args...)
        pg_vec = collect(args[1:N_gen])
        qg_vec = collect(args[N_gen+1:2*N_gen])
        grad = TSIConstraintPrime(psd, Surrogate, st_args, pg_vec, qg_vec)
        # print("TSI gradient constrint: ", maximum(abs.(grad)), "\n")
        # g[1:2*N_gen] .= grad
        g[1:2*N_gen] .= 0. .*grad
        
    end

    function tsih(h::AbstractMatrix, args...)
        pg_vec = collect(args[1:N_gen])
        qg_vec = collect(args[N_gen+1:2*N_gen])
        hess = TSIConstraintPrimePrime(psd, Surrogate, st_args, pg_vec, qg_vec)
        # print("TSI hess constrint: ", maximum(abs.(hess)), "\n")
        for i = 1:2*N_gen
            for j = 1:i
                # h[i, j] = hess[i,j]
                h[i, j] = 0.
            end
        end
        
    end

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    Pl = st_args["PL"]
    Ql = st_args["QL"]
    println("Pl = [", join(Pl, ", "), "]")
    println("Ql = [", join(Ql, ", "), "]")

    register(m, :tsicon, 2*N_gen, tsif, tsig, tsih)

    @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= 0.5 )
    
    if !ispath(solution_dir)
		mkpath(solution_dir)
	end
    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir="nyTempTSI")
    
    # Pg_val = value.(m[:p_g])
    # Qg_val = value.(m[:q_g])


    # gen_idx = st_args["gen_idx"] .+1
    # PG_full = zeros(st_args["numb_gen"])
    # QG_full = zeros(st_args["numb_gen"])
    
    # PG_full[gen_idx] = Pg_val
    # QG_full[gen_idx] = Qg_val

    # println("Pg = [", join(Pg_val, ", "), "]")
    # println("Qg = [", join(Qg_val, ", "), "]")

	print("done. Objective value: \$", round(solution.base_cost, digits=1),
		".\nWriting solution to "*solution_dir*" ... \n")

end
