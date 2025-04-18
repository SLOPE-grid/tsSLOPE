# method to generate a TSI constraint
#
# Inputs: - GPmodel
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

struct TSIConstraint
    m_::Model
    psd_::SCACOPFdata
    GPmodel_::Dict
    st_args_::Dict
    function TSIConstraint(m::Model, psd::SCACOPFdata, GPmodel::Dict, st_args::Dict)
        psd_tmp = psd
        GPmodel_tmp = GPmodel
        st_args_tmp = st_args
        return new(m, psd_tmp,GPmodel_tmp,st_args_tmp)
    end
  end
  
  function (tsi_f::TSIConstraint)(pg,qg)
    Pg_values, Qg_values = get_values(tsi_f.m_, pg, qg)
  
    pgen_ls = tsi_f.st_args_["pgen_ls"] .+ 1
    disp_load = tsi_f.st_args_["disp_load"] .+ 1
    PG_full = zeros(length(pgen_ls)+length(disp_load))
    QG_full = zeros(length(pgen_ls)+length(disp_load))
    
    PG_full[pgen_ls] = Pg_values
    QG_full[pgen_ls] = Qg_values
    
    condition = tsi_f.psd_.N[:, :Pd] .> 0
    load_bus_indices = findall(condition)
  
    PG_full[disp_load] = tsi_f.psd_.N[load_bus_indices, :Pd]
    QG_full[disp_load] = tsi_f.psd_.N[load_bus_indices, :Qd]
    
    # Call the Python function
    tsilib = ret_tsilib() 
    TSI_f = tsilib.eval_tsi_f(tsi_f.GPmodel_, PG_full, QG_full, tsi_f.st_args_)

    return Float32(TSI_f[1])

  end
  
  struct TSIConstraintPrime
    m_::Model
    psd_::SCACOPFdata
    GPmodel_::Dict
    st_args_::Dict
    function TSIConstraintPrime(m::Model, psd::SCACOPFdata, GPmodel::Dict, st_args::Dict)
        psd_tmp = psd
        GPmodel_tmp = GPmodel
        st_args_tmp = st_args
        return new(m, psd_tmp,GPmodel_tmp,st_args_tmp)
    end
  end
  
  function (tsi_g::TSIConstraintPrime)(pg,qg)
    disp_load = tsi_g.st_args_["disp_load"]
    pgen_ls = tsi_g.st_args_["pgen_ls"]
    gen_idx = tsi_g.st_args_["gen_idx"]
    
    nb = 500
    ng = 90
    
    Pg_GP, Qg_GP = get_values(tsi_g.m_, tsi_f, pg, qg)
    
    condition = tsi_g.psd_.N[:, :Pd] .> 0
    load_bus_indices = findall(condition)
  
    Pl_GP = tsi_g.psd_.N[load_bus_indices, :Pd]
    Ql_GP = tsi_g.psd_.N[load_bus_indices, :Qd]
  
    # Call the Python function
    tsilib = ret_tsilib()
    dTSI = tsilib.eval_tsi_g(tsi_g.GPmodel_, Pg_GP, Qg_GP, Pl_GP, Ql_GP, nb, ng, tsi_g.st_args_)
  
    return Float32(dTSI)
  end
  
  struct TSIConstraintPrimePrime
    m_::Model
    psd_::SCACOPFdata
    GPmodel_::Dict
    st_args_::Dict
    function TSIConstraintPrimePrime(m::Model, psd::SCACOPFdata, GPmodel::Dict, st_args::Dict)
        psd_tmp = psd
        GPmodel_tmp = GPmodel
        st_args_tmp = st_args
        return new(m, psd_tmp,GPmodel_tmp,st_args_tmp)
    end
  end
  
  function (tsi_h::TSIConstraintPrimePrime)(pg,qg)
    disp_load = tsi_h.st_args_["disp_load"]
    pgen_ls = tsi_h.st_args_["pgen_ls"]
    gen_idx = tsi_h.st_args_["gen_idx"]
    
    nb = 500
    ng = 90
    
    Pg_GP, Qg_GP = get_values(tsi_h.m_, pg, qg)
    
    condition = tsi_h.psd_.N[:, :Pd] .> 0
    load_bus_indices = findall(condition)
  
    Pl_GP = tsi_h.psd_.N[load_bus_indices, :Pd]
    Ql_GP = tsi_h.psd_.N[load_bus_indices, :Qd]
  
    muTSI = 1.0
    
    # Call the Python function
    tsilib = ret_tsilib()
    dTSI = tsilib.eval_tsi_h(tsi_g.GPmodel_, Pg_GP, Qg_GP, Pl_GP, Ql_GP, nb, ng, muTSI, tsi_g.st_args_)
  
    return dTSI
  end
  
