import andes
from andes.utils.paths import get_case
import numpy as np
import os
from pyDOE import lhs
import scipy.io as scio
import shutil


# Latin Hypercube Sampling
def LHS(samnum, range, dim):
    samples = lhs(dim, samnum)  # shape = (samnum, dim)
    samples = range[0] + (range[1] - range[0]) * samples
    return samples


# Load data
PATHcwd = os.getcwd()
andes.config_logger(stream_level=20)
case = get_case(PATHcwd + '/wecc_2VSM_Fault.xlsx')
ss = andes.run(case, default_config=True)

# Step 2: Set sample number and perturbation range of generators and loads
num_sample = 20000
range_gen = [0.5, 0.7]  # [0.85, 1.05]  # [0.55, 0.95]  # 0.6-1.4, time_cut = 0.17
range_rew = [0.0, 0.7]

# Step 3: Generate load samples using LHS
load_number = len(ss.PQ.p0.v)
load_LoadMW = ss.PQ.p0.v
load_LoadMVR = ss.PQ.q0.v
pf_load = np.arctan(load_LoadMVR/load_LoadMW)
p_load_sample = LHS(num_sample, range_gen, load_number) * load_LoadMW
p_load_sample_sum = np.sum(p_load_sample, axis=1)

# Set the index of renewable and synchronous generator
gen_number = len(ss.PV.p0.v)+1
rew_idx = (np.array([16, 21])-1).tolist()
syn_idx = [i for i in range(gen_number) if i not in rew_idx]

# Store the GenMW of generators at the slack bus and PV buses in the same matrix, arranged in the order of their corresponding bus indices.
slack_idx = 17-1
slack_GenMW = ss.Slack.p0.v
PV_GenMW = ss.PV.p0.v
gen_GenMW = np.insert(PV_GenMW, slack_idx, slack_GenMW)

# Generate renewable samples using LHS
rew_number = len(rew_idx)
rew_GenMW = gen_GenMW[rew_idx]
p_rew_sample = LHS(num_sample, range_rew, rew_number) * rew_GenMW
p_rew_sample_max = np.max(p_rew_sample, axis=0)
p_rew_sample_sum = np.sum(p_rew_sample, axis=1)

# Generate synchronous generator samples using LHS
syn_number = len(syn_idx)
syn_GenMW = gen_GenMW[syn_idx]
p_syn_sample = LHS(num_sample, range_gen, syn_number) * syn_GenMW
p_syn_sample_max = np.max(p_syn_sample, axis=0)
p_syn_sample_sum = np.sum(p_syn_sample, axis=1)

# Store the samples of renewable and synchronous generators in a single matrix, ordered according to their corresponding bus indices.
p_gen_sample = np.zeros((num_sample, gen_number))
p_gen_sample[:, rew_idx] = p_rew_sample
p_gen_sample[:, syn_idx] = p_syn_sample

# Step 4: Balance the active power of generation and load demand
multiple = p_load_sample_sum / (p_syn_sample_sum + p_rew_sample_sum)
p_load_sample = p_load_sample / multiple.reshape(-1, 1)
p_load_sample_sum = np.sum(p_load_sample, axis=1)
q_load_sample = p_load_sample * (np.tan(pf_load))
q_load_sample_sum = np.sum(q_load_sample, axis=1)

# Check balance
p_gen_sample_sum = p_syn_sample_sum + p_rew_sample_sum
p_balance_sum = p_gen_sample_sum - p_load_sample_sum

# Obtain the indices of PV bus generators excluding the slack bus
p_gen_idx = [i for i in range(1, gen_number+1) if i != slack_idx+1]

# PF
ss.PFlow.config.report = 0  # write output report  0/1
PF = ss.PFlow.run()
if PF != True:
    print('Power flow error')

# Create the folder to save samples
shutil.rmtree('D:/ANDES/wecc/batch_cases')
os.makedirs('D:/ANDES/wecc/batch_cases')

# Generate the different scenarios
digit_len = len(str(num_sample))-1
for i in range(num_sample):
    for j in range(load_number):
        ss.PQ.alter('p0', f'PQ_{j+1}', p_load_sample[i, j])  # Change the p0 of loads
        ss.PQ.alter('q0', f'PQ_{j+1}', q_load_sample[i, j])  # Change the q0 of loads
    for j in p_gen_idx:
        ss.PV.alter('p0', j, p_gen_sample[i, j-1])    # Change the p0 of generators

    # Generate the xlsx files
    file_name = f'D:/ANDES/wecc/batch_cases/wecc_n{str(i).zfill(digit_len)}.xlsx'
    andes.io.dump(ss, 'xlsx', file_name, overwrite=True)

# Save the data
Path_Save = PATHcwd + r'\record_' + str(num_sample) + 'n'
if not os.path.exists(Path_Save):
    os.makedirs(Path_Save)

Data = np.hstack((p_rew_sample, p_syn_sample, p_load_sample, q_load_sample))
scio.savemat(Path_Save + r'\record_data.mat', {'Data': Data})
