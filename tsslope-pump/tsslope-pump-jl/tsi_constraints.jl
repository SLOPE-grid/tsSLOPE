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
  total_num_gen = st_args["numb_gen"]

  PG_full = zeros(total_num_gen)
  QG_full = zeros(total_num_gen)  
  PG_full[gen_idx] = pg
  QG_full[gen_idx] = qg

  # Call the Python function
  tsilib = ret_tsilib() 
  TSI_f = tsilib.eval_tsi_f(Surrogate, PG_full, QG_full, st_args)

  return Float64(TSI_f[1])
end

### define first derivative for the TSI constraint

function TSIConstraintPrime(psd::SCACOPFdata, Surrogate::Dict, st_args::Dict, pg, qg, approx_type = "None")

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
  tsilib = ret_tsilib()
  dTSI = tsilib.eval_tsi_g(Surrogate, PG_full, QG_full, PL, QL, st_args)

  # extract gradient infomation just for active generators
  grad = zeros(2 * num_active_gen)

  # check if the gradient is with respect to just pg or both pg qg
  if length(dTSI) == total_num_gen 
    # gradient wrt [pg]
    grad[1:num_active_gen] = dTSI[gen_idx]
  elseif length(dTSI) == num_active_gen 
    # gradient wrt [pg_active]
    grad[1:num_active_gen] = dTSI
  elseif length(dTSI) == 2*total_num_gen
    # gradient wrt [pg qg]
    grad[1:num_active_gen] = dTSI[gen_idx]
    grad[1+num_active_gen:2*num_active_gen] = dTSI[gen_idx.+num_active_gen]
  elseif length(dTSI) == 2*num_active_gen
    # gradient wrt [pg_active qg_active]
    grad[1:num_active_gen] = dTSI[1:num_active_gen]
    grad[1+num_active_gen:2*num_active_gen] = dTSI[(1:num_active_gen).+num_active_gen]
  elseif length(dTSI) == 2 * total_num_gen + 2 * nl 
    # gradient wrt [pg pl qg ql]
    numb_gen_loads = total_num_gen + nl 
    grad[1:num_active_gen] = dTSI[gen_idx]
    grad[1+num_active_gen:2*num_active_gen] = dTSI[gen_idx.+ numb_gen_loads]
  else
    # gradient wrt [pg_active pl qg_active ql]
    numb_gen_loads = num_active_gen + nl 
    grad[1:num_active_gen] = dTSI[1:num_active_gen]
    grad[1+num_active_gen:2*num_active_gen] = dTSI[(1:num_active_gen).+ numb_gen_loads]
  end

  if approx_type == "Sparse"
    idx = findall(!iszero, grad)
    return Int.(idx), Float64.(grad[idx]), Float64.(grad)
  else
    return Float64.(grad)
  end
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
  elseif size(dTSI2)[1] == num_active_gen
    # Hessian wrt [pg_active]
    hess[1:num_active_gen, 1:num_active_gen] = dTSI2
  elseif size(dTSI2)[1] == 2*total_num_gen
    # Hessian wrt [pg qg]
    gen_idx_full = vcat(gen_idx, gen_idx .+ total_num_gen)
    hess = dTSI2[gen_idx_full, gen_idx_full]
  elseif size(dTSI2)[1] == 2*num_active_gen
    # Hessian wrt [pg_active qg_active]
    gen_idx_full = vcat(1:num_active_gen, (1:num_active_gen) .+ num_active_gen)
    hess = dTSI2[gen_idx_full, gen_idx_full]
  elseif size(dTSI2)[1] == 2 * total_num_gen + 2 * nl 
    # Hessian wrt [pg pl qg ql]
    numb_gen_loads = total_num_gen + nl 
    gen_idx_full = vcat(gen_idx, gen_idx .+ numb_gen_loads)
    hess = dTSI2[gen_idx_full, gen_idx_full]
  else
    # Hessian wrt [pg_active pl qg_active ql]
    numb_gen_loads = num_active_gen + nl 
    gen_idx_full = vcat(1:num_active_gen, (1:num_active_gen) .+ numb_gen_loads)
    hess = dTSI2[gen_idx_full, gen_idx_full]
  end

  return Float64.(hess)
end

function TSIConstraintHessApprox(st_args::Dict, B, S, Y, approx_type="Sparse")

  # Call Python function via PyCall
  tsilib = ret_tsilib()

  if approx_type == "Sparse"

    # println("In Sparse")
    num_active_gen = st_args["numb_active_gen"]
    dTSI2 = tsilib.eval_tsi_h_approx(B, S, Y, approx_type, st_args["top_idx"])

    gen_idx_full = vcat(1:num_active_gen, (1:num_active_gen) .+ num_active_gen)
    hess = dTSI2[gen_idx_full, gen_idx_full]

    # println("Created Approximation")

    return Float64.(hess)
  elseif approx_type == "Sparse_pattern"
    # println("In Sparse Pattern")
    I, J, top = tsilib.eval_tsi_h_approx(B, S, Y, approx_type)

    return Int.(I), Int.(J), Int.(top)
  else

    num_active_gen = st_args["numb_active_gen"]

    dTSI2 = tsilib.eval_tsi_h_approx(B, S, Y, approx_type)

    gen_idx_full = vcat(1:num_active_gen, (1:num_active_gen) .+ num_active_gen)
    hess = dTSI2[gen_idx_full, gen_idx_full]
  
    return Float64.(hess)
  end
end

function h_analysis(
  H::AbstractMatrix{<:Real};
  sparsity_tol::Float64 = 1e-3,
  verbose::Bool = true,
  save_Hess::Bool = false,
  ev_nonzero_tol::Float64 = 1e-10,
)

  tsilib = ret_tsilib()

  # Call Python with keyword arguments
  pyres = tsilib.analyze_hessian(
      H;
      sparsity_tol = sparsity_tol,
      verbose = verbose,
      save_Hess = save_Hess,
      ev_nonzero_tol = 1e-10
  )

  # ---- Convert to Julia-native structures ----
  results = Dict{String,Any}()

  if save_Hess
    results["H"]        = Matrix{Float64}(pyres["H"])
  end

  results["eigenvalues"] = Vector{Float64}(pyres["eigenvalues"])
  results["hessian_density"] = Float64(pyres["hessian_density"])

  results["nonzero_ev_indx"] =
      Vector{Int}(pyres["nonzero_ev_indx"]) 
  
  results["nonzero_evecs"] =
      Matrix{Float64}(pyres["nonzero_evecs"])

  return results
end