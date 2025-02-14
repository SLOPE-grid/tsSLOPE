
include("load_env_values.jl")
include("exajugo_call.jl") 

include(string(path_to_tsslope, "/init.jl"))

GPmodel, data, TSI = tsslope_lib.load_model(model_path, data_record)


using Pkg;

Pkg.activate((path_to_exajugo))

push!(LOAD_PATH, string(path_to_exajugo, "/modules"))

using Ipopt, JuMP, Printf
using SCACOPFSubproblems

m, psd, st_args = TSACOPF(case_path, case_sol_path, pf_limit_file, GPmodel);

tsicon = TSIConstraint(psd, GPmodel, st_args)
tsicon_prime = TSIConstraintPrime(psd, GPmodel, st_args)
tsicon_prime_prime = TSIConstraintPrimePrime(psd, GPmodel, st_args)
register(m, :tsicon, 1, (pg,qg) -> tsicon(pg,qg), (pg,qg) -> tsicon_prime(pg,qg), (pg,qg) -> tsicon_prime_prime(pg,qg))

@constraint(m, tsicon( m[:p_g], m[:q_g]) <= 0 )

JuMP.optimize!(m)



