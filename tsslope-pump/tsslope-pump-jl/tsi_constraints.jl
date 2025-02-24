# method to generate a TSI constraint
#
# Inputs: - GPmodel
using PyCall

function ret_tsilib()

       return pyimport("tsslope-pump-py")
       end


### define TSI constraint

struct TSIConstraint
    psd_::SCACOPFdata
    GPmodel_::Dict
    st_args_::Dict
    function TSIConstraint(psd::SCACOPFdata, GPmodel::Dict, st_args::Dict)
        psd_tmp = psd
        GPmodel_tmp = GPmodel
        st_args_tmp = st_args
        
        return new(psd_tmp,GPmodel_tmp,st_args_tmp)
    end
  end
  
  function (tsi_f::TSIConstraint)(pg,qg)

    tsilib = ret_tsilib() 
    
    Pg = pg
    Qg = qg
    Pg_values = JuMP.value.(Pg)
    Qg_values = JuMP.value.(Qg)

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
    TSI_f = tsilib.eval_tsi_f(tsi_f.GPmodel_, PG_full, QG_full, tsi_f.st_args_)

    return Float32(TSI_f[1])

  end
  
  struct TSIConstraintPrime
    psd_::SCACOPFdata
    GPmodel_::Dict
    st_args_::Dict
    function TSIConstraintPrime(psd::SCACOPFdata, GPmodel::Dict, st_args::Dict)
        psd_tmp = psd
        GPmodel_tmp = GPmodel
        st_args_tmp = st_args
        return new(psd_tmp,GPmodel_tmp,st_args_tmp)
    end
  end
  
  function (tsi_g::TSIConstraintPrime)(pg,qg)

    tsilib = ret_tsilib()
    
    disp_load = tsi_g.st_args_["disp_load"]
    pgen_ls = tsi_g.st_args_["pgen_ls"]
    gen_idx = tsi_g.st_args_["gen_idx"]
    
    nb = 500
    ng = 90
    
    Pg = pg
    Qg = qg
    Pg_GP = JuMP.value.(Pg)
    Qg_GP = JuMP.value.(Qg)
    
    condition = tsi_g.psd_.N[:, :Pd] .> 0
    load_bus_indices = findall(condition)
  
    Pl_GP = tsi_g.psd_.N[load_bus_indices, :Pd]
    Ql_GP = tsi_g.psd_.N[load_bus_indices, :Qd]
  
    # Call the Python function
    dTSI = tsilib.eval_tsi_g(tsi_g.GPmodel_, Pg_GP, Qg_GP, Pl_GP, Ql_GP, nb, ng, tsi_g.st_args_)
  
    return Float32(dTSI)
  end
  
  struct TSIConstraintPrimePrime
    psd_::SCACOPFdata
    GPmodel_::Dict
    st_args_::Dict
    function TSIConstraintPrimePrime(psd::SCACOPFdata, GPmodel::Dict, st_args::Dict)
        psd_tmp = psd
        GPmodel_tmp = GPmodel
        st_args_tmp = st_args
        return new(psd_tmp,GPmodel_tmp,st_args_tmp)
    end
  end
  
  function (tsi_h::TSIConstraintPrimePrime)(pg,qg)

    tsilib = ret_tsilib()
    
    disp_load = tsi_h.st_args_["disp_load"]
    pgen_ls = tsi_h.st_args_["pgen_ls"]
    gen_idx = tsi_h.st_args_["gen_idx"]
    
    nb = 500
    ng = 90
    
    Pg = pg
    Qg = qg
    Pg_GP = JuMP.value.(Pg)
    Qg_GP = JuMP.value.(Qg)
    
    condition = tsi_h.psd_.N[:, :Pd] .> 0
    load_bus_indices = findall(condition)
  
    Pl_GP = tsi_h.psd_.N[load_bus_indices, :Pd]
    Ql_GP = tsi_h.psd_.N[load_bus_indices, :Qd]
  
    muTSI = 1.0
    
    # Call the Python function
    dTSI = tsilib.eval_tsi_h(tsi_g.GPmodel_, Pg_GP, Qg_GP, Pl_GP, Ql_GP, nb, ng, muTSI, tsi_g.st_args_)
  
    return dTSI
  end
  
