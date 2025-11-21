# method to generate a TSI constraint
#
# Inputs: - Surrogate
using PyCall

function ret_tsilib()
  return pyimport("tsslope-pump-py")
end

function get_values(tsi_f, pg, qg)
  if JuMP.has_values(tsi_f)
    # Use values if optimization has been performed
    Pg_values = JuMP.value.(pg)
    Qg_values = JuMP.value.(qg)
  else
    # Use start values if optimization has not been performed yet
    Pg_values = JuMP.start_value.(pg)
    Qg_values = JuMP.start_value.(qg)
  end
  return Pg_values, Qg_values
end

### define TSI constraint

function TSIConstraint2(psd::SCACOPFdata, Surrogate::Dict, st_args::Dict, pg, qg)
  gen_idx = st_args["gen_idx"] .+1
  PG_full = zeros(st_args["numb_gen"])
  QG_full = zeros(st_args["numb_gen"])
  
  PG_full[gen_idx] = pg
  QG_full[gen_idx] = qg

  # Call the Python function
  tsilib = ret_tsilib() 
  TSI_f = tsilib.eval_tsi_f(Surrogate, PG_full, QG_full, st_args)

  return Float64(TSI_f[1])
end

function TSIConstraintPrime2(psd::SCACOPFdata, Surrogate::Dict, st_args::Dict, pg, qg)
  nb = 500
  ng = 90
  # Load information
  PL = st_args["PL"]
  QL = st_args["QL"]

  gen_idx = st_args["gen_idx"] .+1
  PG_full = zeros(st_args["numb_gen"])
  QG_full = zeros(st_args["numb_gen"])
  PG_full[gen_idx] = pg
  QG_full[gen_idx] = qg

  # Call Python function via PyCall
  tsilib = ret_tsilib()
  dTSI = tsilib.eval_tsi_g(Surrogate, PG_full, QG_full, PL, QL, nb, ng, st_args)

  grad = zeros(2 * length(st_args["gen_idx"]))
  # check if the gradient is with respect to just pg or both pg qg
  if length(dTSI) == st_args["numb_gen"]
    grad[1:length(pg)] = dTSI[gen_idx]
  else
    grad[1:length(pg)] = dTSI[gen_idx]
    grad[1+length(pg):2*length(pg)] = dTSI[gen_idx.+length(pg)]
  end

  display("Taking derivative")

  return Float64.(grad)
end

function TSIConstraintPrimePrime2(psd::SCACOPFdata, Surrogate::Dict, st_args::Dict, pg, qg)
  nb = 500
  ng = 90

  # Load information
  PL = st_args["PL"]
  QL = st_args["QL"]

  muTSI = 1.0

  gen_idx = st_args["gen_idx"] .+1
  PG_full = zeros(st_args["numb_gen"])
  QG_full = zeros(st_args["numb_gen"])
  PG_full[gen_idx] = pg
  QG_full[gen_idx] = qg

  # Call Python function via PyCall
  tsilib = ret_tsilib()
  dTSI = tsilib.eval_tsi_h(Surrogate, PG_full, QG_full, PL, QL, nb, ng, muTSI, st_args)

  display("Taking 2nd derivative")

  return Float64.(dTSI)
end

# struct TSIConstraint
#     psd_::SCACOPFdata
#     Surrogate_::Dict
#     st_args_::Dict
#     function TSIConstraint(psd::SCACOPFdata, Surrogate::Dict, st_args::Dict)
#         psd_tmp = psd
#         Surrogate_tmp = Surrogate
#         st_args_tmp = st_args
#         return new(psd_tmp,Surrogate_tmp,st_args_tmp)
#     end
#   end

#   function (tsi_f::TSIConstraint)(pg,qg)  
#     gen_idx = tsi_f.st_args_["gen_idx"] .+ 1

#     display("length(gen_idx): $(length(gen_idx)))")

#     PG_full = zeros(tsi_f.st_args_["numb_gen"])
#     QG_full = zeros(tsi_f.st_args_["numb_gen"])
    
#     PG_full[gen_idx] = pg
#     QG_full[gen_idx] = qg
        
#     # Call the Python function
#     tsilib = ret_tsilib() 
#     TSI_f = tsilib.eval_tsi_f(tsi_f.Surrogate_, PG_full, QG_full, tsi_f.st_args_)

#     display(TSI_f)

#     return Float32(TSI_f[1])

#   end
  
#   struct TSIConstraintPrime
#     psd_::SCACOPFdata
#     Surrogate_::Dict
#     st_args_::Dict
#     function TSIConstraintPrime(psd::SCACOPFdata, Surrogate::Dict, st_args::Dict)
#         psd_tmp = psd
#         Surrogate_tmp = Surrogate
#         st_args_tmp = st_args
#         return new(psd_tmp, Surrogate_tmp, st_args_tmp)
#     end
#   end
  
#   function (tsi_g::TSIConstraintPrime)(pg, qg)
#     if tsi_g.Surrogate_["model_type"] == "CNN"
#         return TSIConstraintPrime_CNN(tsi_g, pg, qg)
#     elseif tsi_g.Surrogate_["model_type"] == "DSPP"
#         return TSIConstraintPrime_GP(tsi_g, pg, qg)
#     end
#   end

#   function TSIConstraintPrime_CNN(tsi_g::TSIConstraintPrime, pg, qg)
#     nb = 500
#     ng = 90

#     # Get values from model
#     # Pg, Qg = get_values(tsi_g.m_, pg, qg)

#     # Load information
#     PL = tsi_g.st_args_["PL"]
#     QL = tsi_g.st_args_["QL"]

#     # Call Python function via PyCall
#     tsilib = ret_tsilib()
#     dTSI = tsilib.eval_tsi_g(tsi_g.Surrogate_, pg, qg, Pl, Ql, nb, ng, tsi_g.st_args_)

#     display("Taking derivative")

#     return Float32(dTSI)
#   end

#   function TSIConstraintPrime_GP(tsi_g::TSIConstraintPrime, pg, qg)
#     disp_load = tsi_g.st_args_["disp_load"]
#     pgen_ls = tsi_g.st_args_["pgen_ls"]
#     gen_idx = tsi_g.st_args_["gen_idx"]

#     nb = 500
#     ng = 90

#     # Get values from model
#     # Pg_GP, Qg_GP = get_values(tsi_g.m_, pg, qg)

#     # Load information
#     condition = tsi_g.psd_.N[:, :Pd] .> 0
#     load_bus_indices = findall(condition)
#     Pl_GP = tsi_g.psd_.N[load_bus_indices, :Pd]
#     Ql_GP = tsi_g.psd_.N[load_bus_indices, :Qd]

#     # Call Python function via PyCall
#     tsilib = ret_tsilib()
#     dTSI = tsilib.eval_tsi_g(tsi_g.Surrogate_, Pg_GP, Qg_GP, Pl_GP, Ql_GP, nb, ng, tsi_g.st_args_)

#     display("Taking derivative")

#     return Float32(dTSI)
#   end

  
#   struct TSIConstraintPrimePrime
#     psd_::SCACOPFdata
#     Surrogate_::Dict
#     st_args_::Dict
#     function TSIConstraintPrimePrime(psd::SCACOPFdata, Surrogate::Dict, st_args::Dict)
#         psd_tmp = psd
#         Surrogate_tmp = Surrogate
#         st_args_tmp = st_args
#         return new(psd_tmp,Surrogate_tmp,st_args_tmp)
#     end
#   end

#   function (tsi_h::TSIConstraintPrimePrime)(pg, qg)
#     if tsi_h.Surrogate_["model_type"] == "CNN"
#         return TSIConstraintPrimePrime_CNN(tsi_h, pg, qg)
#     elseif tsi_h.Surrogate_["model_type"] == "DSPP"
#         return TSIConstraintPrimePrime_GP(tsi_h, pg, qg)
#     end
#   end
  
#   function TSIConstraintPrimePrime_CNN(tsi_h::TSIConstraintPrimePrime, pg, qg)
#     nb = 500
#     ng = 90

#     # Get values from model
#     # Pg, Qg = get_values(tsi_h.m_, pg, qg)

#     # Load information
#     PL = tsi_h.st_args_["PL"]
#     QL = tsi_h.st_args_["QL"]

#     muTSI = 1.0

#     # Call Python function via PyCall
#     tsilib = ret_tsilib()
#     dTSI = tsilib.eval_tsi_h(tsi_h.Surrogate_, pg, qg, Pl, Ql, nb, ng, muTSI, tsi_h.st_args_)

#     display("Taking 2nd derivative")

#     return Float32(dTSI)
#   end

#   function TSIConstraintPrimePrime_GP(tsi_h::TSIConstraintPrimePrime, pg, qg)
#     disp_load = tsi_h.st_args_["disp_load"]
#     pgen_ls = tsi_h.st_args_["pgen_ls"]
#     gen_idx = tsi_h.st_args_["gen_idx"]
    
#     nb = 500
#     ng = 90
    
#     Pg_GP, Qg_GP = get_values(tsi_h.m_, pg, qg)
    
#     condition = tsi_h.psd_.N[:, :Pd] .> 0
#     load_bus_indices = findall(condition)
  
#     Pl_GP = tsi_h.psd_.N[load_bus_indices, :Pd]
#     Ql_GP = tsi_h.psd_.N[load_bus_indices, :Qd]
  
#     muTSI = 1.0
    
#     # Call the Python function
#     tsilib = ret_tsilib()
#     dTSI = tsilib.eval_tsi_h(tsi_g.Surrogate_, Pg_GP, Qg_GP, Pl_GP, Ql_GP, nb, ng, muTSI, tsi_g.st_args_)
  
#     display("Taking 2nd derivative")

#     return dTSI
#   end
  
