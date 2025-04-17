

import scipy.io as scio
import numpy as np


from pypower.ext2int import ext2int
from pypower.bustypes import bustypes
from pypower.Mul_confi_get import Mul_confi_get #, F_BUS, T_BUS, BR_R, BR_X, BR_B
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A
from pypower.idx_bus import PD


'''
def load_case_psd(psd):

   baseMVA = 100
   bus = psd['B'] 
   gen = psd['G']
   branch = psd['T']
   pf = psd['']


   branch[:, [L_Nidx, T_Nidx]] = branch_pw[:, 0:2] - 1


   [ref, pv, pq, pvref] = new_bustypes(bus, gen)
'''




def load_case(case_file, pf_file):

#mpc = scio.loadmat('case_ACTIVSg500.mat')
   mpc = scio.loadmat(case_file)
   mpc = ext2int(mpc)

   baseMVA = 100
   bus = mpc['bus'] 
   gen = mpc['gen']
   branch = mpc['branch']
   gencost = mpc['gencost']
   nb = bus.shape[0]
   ng = gen.shape[0]
   nl = branch.shape[0]
   nw = 9

   [ref, pv, pq, pvref] = bustypes(bus, gen)
   pql = np.array([i for i in range(nb) if bus[i, PD] != 0])

   pf = scio.loadmat(pf_file)
   
   Qg_max = pf['Qg_max']
   Qg_min = pf['Qg_min']
   Pflow_max = pf['Pflow_max']
   branch_pw = pf['branch']
   V_max = pf['V_max']
   V_min = pf['V_min']
   
   branch[:, [F_BUS, T_BUS]] = branch_pw[:, 0:2] - 1
   branch[:, [BR_R, BR_X, BR_B]] = branch_pw[:, 2:-1]
   branch[:, RATE_A] = Pflow_max+200
   load_idx = pf['load_idx'][0, :]-1

   gen_syn_idx = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
               35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53, 54, 55, 56, 57, 58, 59]
   gen_rew_ide = [3, 9, 16, 17, 22, 33, 34, 51, 52]
   gen_idx = gen_rew_ide + gen_syn_idx
   ref_gen_idx = gen_idx.index(ref)
   load_bus = list(set(load_idx))
   genload_bus = pvref.tolist() + load_bus
   genload_bus = list(map(int, genload_bus))
   genload_bus_sort = list(set(genload_bus))
   disp_load = [genload_bus_sort.index(i) for i in load_bus]
   pgen_ls = [genload_bus_sort.index(i) for i in pvref]

   slack_gen = 3
   n_unstable = 0
   confi_level = 2
   Ql_tol_min = 0.01

   Mul_confi = Mul_confi_get(confi_level)

   st_args = {'gen_idx': gen_idx, 'disp_load': disp_load, 'pgen_ls': pgen_ls, 'num_J_H': 0,\
              'Mul_confi': Mul_confi, 'load_bus': load_bus, 'load_idx': load_idx, 'genload_bus_sort': genload_bus_sort, 'pvref': pvref}


   return st_args
    
   [ref, pv, pq, pvref] = bustypes(bus, gen)
   print([ref, pv, pq, pvref])



