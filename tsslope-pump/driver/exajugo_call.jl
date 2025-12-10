
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
    # opt = optimizer_with_attributes(Ipopt.Optimizer,
    #                         # "linear_solver" => "ma57",
    #                         "sb" => "yes",
    #                         "tol" =>  1e-6,
    #                         "mu_superlinear_decrease_power" =>  1.25,
    #                         "mu_linear_decrease_factor" =>  0.4,
    #                         "max_iter" =>  500,
    #                         "print_user_options"  =>  "yes"
    #                         )
                  
    # get primal starting point
    x0 = get_primal_starting_point(psd)

    # function tsif(pg, qg)
    #     # print("TSI constrint: ", TSIConstraint(psd, Surrogate, st_args, pg, qg),"\n")
    #     return TSIConstraint(psd, Surrogate, st_args, pg, qg)
    # end

    # function tsig(pg, qg)
    #     grad = TSIConstraintPrime(psd, Surrogate, st_args, pg, qg)
    #     return grad        
    # end

    # function tsih(pg, qg)
    #     hess = TSIConstraintPrimePrime(psd, Surrogate, st_args, pg, qg)
    #     return hess
    # end
               
    # pg = [7.663979921630508, 4.413387597729772, 8.826778511511918, 1.564960257252561, 0.01287966023356796, 0.02579230227999361, 0.038281502730900877, 0.01823504422016482, 0.012198996583396553, 3.595316326544632, 1.6091324478034001, 1.6091751191586243, 5.983325777022009, 3.0054804788673364, 1.9221090569353885, 1.9221079289671803, 3.843879232897778, 2.9154475966108775, 2.0584859030754763, 0.04999381561800943, 0.1775775667013186, 5.078945604010334, 5.1146985375204705, 0.028771655794996027, 0.010891652850466508, 0.028771605881902446, 0.0287716060618411, 0.008148807341717943, 0.022221061693687517, 0.41961202590446206, 0.14805737380506173, 0.06018179784327167, 0.1421000177631754, 0.7355061874258805, 0.7796497708295823, 0.7942894024203592, 0.7942894023969989, 0.6025593464221594, 0.008079879724726017, 0.031026627923287184, 0.010076452984810592, 0.06494281160282457, 0.04100028913771343, 1.9569189481316076, 1.9569248902949368, 1.9569196387027081, 0.6783439734415687, 1.3501357052635536, 1.0886586887036307, 0.5561656670271722, 0.4467285752066622, 0.4467175374929713, 0.2376234955407589, 0.2775115262950042, 0.5958185222816899, 0.1778049095411438]
    # qg = [7.663979921630508, 4.413387597729772, 8.826778511511918, 1.564960257252561, 0.01287966023356796, 0.02579230227999361, 0.038281502730900877, 0.01823504422016482, 0.012198996583396553, 3.595316326544632, 1.6091324478034001, 1.6091751191586243, 5.983325777022009, 3.0054804788673364, 1.9221090569353885, 1.9221079289671803, 3.843879232897778, 2.9154475966108775, 2.0584859030754763, 0.04999381561800943, 0.1775775667013186, 5.078945604010334, 5.1146985375204705, 0.028771655794996027, 0.010891652850466508, 0.028771605881902446, 0.0287716060618411, 0.008148807341717943, 0.022221061693687517, 0.41961202590446206, 0.14805737380506173, 0.06018179784327167, 0.1421000177631754, 0.7355061874258805, 0.7796497708295823, 0.7942894024203592, 0.7942894023969989, 0.6025593464221594, 0.008079879724726017, 0.031026627923287184, 0.010076452984810592, 0.06494281160282457, 0.04100028913771343, 1.9569189481316076, 1.9569248902949368, 1.9569196387027081, 0.6783439734415687, 1.3501357052635536, 1.0886586887036307, 0.5561656670271722, 0.4467285752066622, 0.4467175374929713, 0.2376234955407589, 0.2775115262950042, 0.5958185222816899, 0.1778049095411438]

    # finite_difference_check(tsif, tsig, pg, qg; eps = 1e-6)
    # finite_difference_hessian_check(tsif, tsig, tsih, pg, qg; eps=1e-6)

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

	# create model
    m, model_data = create_basecase_model(psd, opt, x0)

    # set_optimizer_attribute(m, "derivative_test", "first-order")   # or "first-order"
    # set_optimizer_attribute(m, "derivative_test_first_index", 3404) 

    register(m, :tsicon, 2*N_gen, tsif, tsig, tsih)

    if surrogate["model_type"] == "CNF"
        st_args["PL"] = -st_args["PL"]
        st_args["QL"] = -st_args["QL"]
        @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= 0.0 )
    else
        @NLconstraint(m, tsicon( m[:p_g]..., m[:q_g]...) >= 0.5 )
    end
    
    if !ispath(solution_dir)
		mkpath(solution_dir)
	end
    solution, m = solve_basecase_from_model(m, psd, model_data, output_dir="nyTempTSI")
    
	# print("done. Objective value: \$", round(solution.base_cost, digits=1),
	# 	".\nWriting solution to "*solution_dir*" ... \n")

end
