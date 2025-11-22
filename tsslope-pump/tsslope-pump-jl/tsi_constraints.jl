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

### define first derivative for the TSI constraint

function TSIConstraintPrime2(psd::SCACOPFdata, Surrogate::Dict, st_args::Dict, pg, qg)

  # Load information
  PL = st_args["PL"]
  QL = st_args["QL"]

  total_num_gen = st_args["numb_gen"]
  num_active_gen = st_args["numb_active_gen"]

  gen_idx = st_args["gen_idx"] .+1
  PG_full = zeros(total_num_gen)
  QG_full = zeros(total_num_gen)
  PG_full[gen_idx] = pg
  QG_full[gen_idx] = qg

  # Call Python function via PyCall
  tsilib = ret_tsilib()
  dTSI = tsilib.eval_tsi_g(Surrogate, PG_full, QG_full, PL, QL, st_args)

  # extract gradient infomation just for active generators
  grad = zeros(2 * num_active_gen)

  # check if the gradient is with respect to just pg or both pg qg
  if length(dTSI) == total_num_gen
    grad[1:num_active_gen] = dTSI[gen_idx]
  else
    grad[1:num_active_gen] = dTSI[gen_idx]
    grad[1+num_active_gen:2*num_active_gen] = dTSI[gen_idx.+num_active_gen]
  end

  return Float64.(grad)
end

### define second derivative for the TSI constraint

function TSIConstraintPrimePrime2(psd::SCACOPFdata, Surrogate::Dict, st_args::Dict, pg, qg)

  # Load information
  PL = st_args["PL"]
  QL = st_args["QL"]

  muTSI = 1.0

  total_num_gen = st_args["numb_gen"]
  num_active_gen = st_args["numb_active_gen"]

  gen_idx = st_args["gen_idx"] .+1
  PG_full = zeros(st_args["numb_gen"])
  QG_full = zeros(st_args["numb_gen"])
  PG_full[gen_idx] = pg
  QG_full[gen_idx] = qg

  # Call Python function via PyCall
  tsilib = ret_tsilib()
  dTSI2 = tsilib.eval_tsi_h(Surrogate, PG_full, QG_full, PL, QL, muTSI, st_args)

  # extract Hessian infomation just for active generators
  hess = zeros(2 * num_active_gen, 2 * num_active_gen)
  if size(dTSI2)[1] == total_num_gen
    hess[1:num_active_gen, 1:num_active_gen] = dTSI2[gen_idx, gen_idx]
  else
    gen_idx_full = vcat(gen_idx, gen_idx .+ num_active_gen)
    hess = dTSI2[gen_idx_full, gen_idx_full]
  end

  return Float64.(hess)
end
