
using Ipopt, JuMP, Printf
using SCACOPFSubproblems

using MAT
using LinearAlgebra
using SparseArrays

jl_lib = string(path_to_tsslope,"/tsslope-pump-jl")
include(string(jl_lib,"/load_case.jl"))

const CACHE = Dict{UInt64, Any}()

function finite_difference_hessian_check(tsif, tsig, tsih, pg, qg; eps = 1e-6)

    n = length(pg)

    # ============================================================
    # 1) Gradient check
    # ============================================================
    g_exact = tsig(pg, qg)

    # println("Analytic Gradient:")
    # println(g_exact)

    println("\nGradient FD Check:")
    println("Index     g_exact         FD1 (1st)       Error1         FD2 (2nd)        Error2")
    println("--------------------------------------------------------------------------------")

    for i = 1:n
        pg_plus  = copy(pg)
        pg_minus = copy(pg)

        pg_plus[i]  += eps
        pg_minus[i] -= eps

        f0 = tsif(pg, qg)
        fp = tsif(pg_plus, qg)
        fm = tsif(pg_minus, qg)

        fd1 = (fp - f0) / eps
        fd2 = (fp - fm) / (2 * eps)

        err1 = abs(g_exact[i] - fd1)
        err2 = abs(g_exact[i] - fd2)

        @printf("%4d   %12.6e   %12.6e   %12.6e   %12.6e   %12.6e\n",
                i, g_exact[i], fd1, err1, fd2, err2)
    end

    # ============================================================
    # 2) Hessian check using gradient differences
    # ============================================================
    H_exact = tsih(pg, qg)

    max_err = 0.0

    for i = 1:n
        pg_plus  = copy(pg)
        pg_minus = copy(pg)

        pg_plus[i]  += eps
        pg_minus[i] -= eps

        gp = tsig(pg_plus,  qg)
        gm = tsig(pg_minus, qg)

        # FD Hessian column i
        H_fd_i = (gp - gm) / (2 * eps)

        # compute element-wise error
        for j = 1:n
            err = abs(H_exact[j,i] - H_fd_i[j])

            # update running max error (your request)
            if err < max_err
                # do nothing
            else
                max_err = err
            end
        end
    end

    println("\nMaximum Hessian absolute error = ", @sprintf("%.6e", max_err))
end

function finite_difference_check(tsif, tsig, pg, qg; eps = 1e-6)
    n = length(pg)

    # analytic gradient
    g_exact = tsig(pg, qg)

    # println("Analytic Gradient:")
    # println(g_exact)

    println("\nFinite Difference Check:")
    println("Index     g_exact         FD1 (1st)       Error1         FD2 (2nd)        Error2")
    println("--------------------------------------------------------------------------------")

    for i = 1:n
        # copy input
        pg_plus  = copy(pg)
        pg_minus = copy(pg)

        # perturb
        pg_plus[i]  += eps
        pg_minus[i] -= eps

        # evaluate function
        f0 = tsif(pg, qg)
        fp = tsif(pg_plus, qg)
        fm = tsif(pg_minus, qg)

        # println(f0, fp, fm)

        # finite differences
        fd1 = (fp - f0) / eps                       # first order
        fd2 = (fp - fm) / (2 * eps)                 # second order

        # errors
        err1 = abs(g_exact[i] - fd1)
        err2 = abs(g_exact[i] - fd2)

        @printf("%4d   %12.6e   %12.6e   %12.6e   %12.6e   %12.6e\n",
                i, g_exact[i], fd1, err1, fd2, err2)
    end
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
                        # h[i, j] = 0.
                    end
                end
                CACHE[hsh] = h
            end     
        end

        register(m, :tsicon, 2*N_gen, tsif, tsig, tsih)

        if Surrogate["model_type"] == "CNF"
            st_args["PL"] = -st_args["PL"]
            st_args["QL"] = -st_args["QL"]
            @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= 0.0 )
        elseif Surrogate["model_type"] == "CNN_silu_no_sig" || Surrogate["model_type"] == "UQ_CNN_SiLU_no_sig"
            @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= 0.0 )
        else
            @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= 0.6 )
        end
        
        if !ispath(solution_dir)
            mkpath(solution_dir)
        end
    end

    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir="output")
    
	# print("done. Objective value: \$", round(solution.base_cost, digits=1),
	# 	".\nWriting solution to "*solution_dir*" ... \n")

end
