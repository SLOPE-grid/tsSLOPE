# method to generate a TSI constraint
#
# Inputs: - Surrogate
using PyCall

function ret_tsilib()
  return pyimport("tsslope-pump-py")
end

### define TSI constraint

function TSIConstraint(psd::SCACOPFdata, Surrogate::Dict, st_args::Dict, pg, qg)
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

function TSIConstraintPrime(psd::SCACOPFdata, Surrogate::Dict, st_args::Dict, pg, qg)

  # Load information
  PL = st_args["PL"]
  QL = st_args["QL"]

  total_num_gen = st_args["numb_gen"]
  num_active_gen = st_args["numb_active_gen"]
  nl = st_args["numb_loads"]

  gen_idx = st_args["gen_idx"] .+1
  PG_full = zeros(total_num_gen)
  QG_full = zeros(total_num_gen)
  PG_full[gen_idx] = pg
  QG_full[gen_idx] = qg

  # Call Python function via PyCall
  # println("1")
  tsilib = ret_tsilib()
  dTSI = tsilib.eval_tsi_g(Surrogate, PG_full, QG_full, PL, QL, st_args)

  # println("Grad TSI size:", size(dTSI))

  # extract gradient infomation just for active generators
  grad = zeros(2 * num_active_gen)
  # println("In 1")

  # check if the gradient is with respect to just pg or both pg qg
  if length(dTSI) == total_num_gen
    # println("In if 1")
    # gradient wrt [pg]
    grad[1:num_active_gen] = dTSI[gen_idx]
  elseif length(dTSI) == 2*total_num_gen
    # println("In if 2")
    # gradient wrt [pg qg]
    grad[1:num_active_gen] = dTSI[gen_idx]
    grad[1+num_active_gen:2*num_active_gen] = dTSI[gen_idx.+num_active_gen]
  else
    # println("In if 3")
    # gradient wrt [pg pl qg ql]
    numb_gen_loads = total_num_gen + nl 
    grad[1:num_active_gen] = dTSI[gen_idx]
    grad[1+num_active_gen:2*num_active_gen] = dTSI[gen_idx.+ numb_gen_loads]
  end
  # println("In 2")

  return Float64.(grad)
end

### define second derivative for the TSI constraint

function TSIConstraintPrimePrime(psd::SCACOPFdata, Surrogate::Dict, st_args::Dict, pg, qg)

  # Load information
  PL = st_args["PL"]
  QL = st_args["QL"]

  muTSI = 1.0

  total_num_gen = st_args["numb_gen"]
  num_active_gen = st_args["numb_active_gen"]
  nl = st_args["numb_loads"]

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
    # Hessian wrt [pg]
    hess[1:num_active_gen, 1:num_active_gen] = dTSI2[gen_idx, gen_idx]
  elseif size(dTSI2)[1] == 2*total_num_gen
    # Hessian wrt [pg qg]
    gen_idx_full = vcat(gen_idx, gen_idx .+ num_active_gen)
    hess = dTSI2[gen_idx_full, gen_idx_full]
  else
    # Hessian wrt [pg pl qg ql]
    numb_gen_loads = total_num_gen + nl 
    gen_idx_full = vcat(gen_idx, gen_idx .+ numb_gen_loads)
    hess = dTSI2[gen_idx_full, gen_idx_full]
  end

  return Float64.(hess)
end
