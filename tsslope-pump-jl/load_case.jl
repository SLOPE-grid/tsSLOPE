using MAT
using LinearAlgebra
using SparseArrays


function Mul_confi_get(confi_level)
    if confi_level == 0  # 0
        Mul_confi = 0.0
    elseif confi_level == 1  # 1*std 68.268949%
        Mul_confi = 1.0
    elseif confi_level == 2  # 2*std 95.449974%
        Mul_confi = 2.0
    elseif confi_level == 3  # 2*std 3*std 99.730020%
        Mul_confi = 3.0
    elseif confi_level == 6  # 60% - 0.8416
        Mul_confi = 0.8416
    elseif confi_level == 7  # 70% - 1.0364
        Mul_confi = 1.0364
    elseif confi_level == 8  # 80% - 1.2815
        Mul_confi = 1.2815
    elseif confi_level == 9  # 90% - 1.6448
        Mul_confi = 1.6448
    end

    return Mul_confi
end

function load_case(psd::SCACOPFdata, pf_file::String)

  baseMVA = psd.MVAbase
  bus = psd.N
  gen = psd.G
  branch = psd.L
  transformer = psd.T
  gencost_slope = psd.G_epicost_slope
  gencost_intercept = psd.G_epicost_intercept
  nb = size(bus, 1)
  ng = size(gen, 1)
  nl = size(branch, 1)
  nw = 9

  # Load power flow data
  pf = matread(pf_file)
  
  # use python index
  ref_py = psd.RefBus - 1
  #pv_Nidx = psd.N[N[!,:Type] .== :PV, :Bus]
  #pq_Nidx= psd.N[N[!,:Type] .== :PQ, :Bus]
  pv_py = findall(row -> (row[:Bus] in psd.G[:, :Bus]) && (row[:Type] == :PV), eachrow(bus)) .- 1 # PV bus indices
  pq_py = findall(row -> (row[:Bus] in psd.G[:, :Bus]) && (row[:Type] == :PQ), eachrow(bus)) .- 1 # PQ bus indices
  pvref_py = vcat(pv_py,ref_py)
  pvref_py = sort(unique(pvref_py))   
  
  #=
  pql = [i for i in 1:nb if bus[i, :Pd] != 0]

  Qg_max = pf["Qg_max"]
  Qg_min = pf["Qg_min"]
  Pflow_max_all = pf["Pflow_max"]
  branch_pw_all = pf["branch"]

#    branch_pw = branch_pw_all[branch_pw_all[:, :TAP] .== 1, :]
#    transformer_pw = branch_pw_all[branch_pw_all[:, :TAP] .!= 1, :]
  Pflow_max_b = Pflow_max_all[branch_pw_all[:, :TAP] .== 1, :]
  Pflow_max_t = Pflow_max_all[branch_pw_all[:, :TAP] .!= 1, :]
  
  V_max = pf["V_max"]
  V_min = pf["V_min"]

#    branch[:, [From, To]] .= branch_pw[:, 1:2] .- 1
#    branch[:, [BR_R, BR_X, BR_B]] .= branch_pw[:, 3:end-1]
  branch[:, RateBase] .= Pflow_max_b .+ 200
  transformer[:, RateBase] .= Pflow_max_t .+ 200

  =#

  load_idx_py = pf["load_idx"][1, :] .- 1

  gen_syn_idx_py = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53, 54, 55, 56, 57, 58, 59]
  gen_rew_ide_py = [3, 9, 16, 17, 22, 33, 34, 51, 52] 
  gen_idx_py = vcat(gen_rew_ide_py, gen_syn_idx_py)
  ref_gen_idx_py = findfirst(==(ref_py), gen_idx_py)
  load_bus_py = unique(load_idx_py)
  genload_bus_py = vcat(pvref_py, load_bus_py)            
  genload_bus_py = Int.(genload_bus_py)                
  genload_bus_sort_py = sort(unique(genload_bus_py))   
  
  disp_load_py = [findfirst(==(i), genload_bus_sort_py) for i in load_bus_py] .- 1
  pgen_ls_py = [findfirst(==(i), genload_bus_sort_py) for i in pvref_py] .- 1

  slack_gen = 3
  n_unstable = 0
  confi_level = 2
  Ql_tol_min = 0.01

  Mul_confi = Mul_confi_get(confi_level)

  st_args = Dict(
      "gen_idx" => gen_idx_py,
      "disp_load" => disp_load_py,
      "pgen_ls" => pgen_ls_py,
      "num_J_H" => 0,
      "Mul_confi" => Mul_confi,
      "load_bus" => load_bus_py,
      "genload_bus_sort" => genload_bus_sort_py,
      "pvref" => pvref_py,
      "pv" => pv_py
  )

  return st_args
end
