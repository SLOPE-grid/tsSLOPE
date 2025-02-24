# renewable energy type3 1372.2MW /6456.0MW = 21.3%
# esa can be downloaded from: https://github.com/mzy2240/ESA

from esa import SAW
import numpy as np
import pandas as pd
from pyDOE import lhs
import matplotlib.pyplot as plt
import concurrent.futures

import os
import scipy.io as scio

def run_pw_simulation(num, CASE_PATH,
                      gen_key_fields, load_key_fields,
                      gen_params, load_params, gen_closed_idx,
                      gen_syn, p_syn_sample, syn_number,
                      gen_rew, p_rew_sample, rew_number,
                      load, p_load_sample, q_load_sample, load_number,
                      Fault_number, t0, te, stepsize):
    
    saw = SAW(CASE_PATH, CreateIfNotFound=True)#, UIVisible=True)

    # Change parameters of PowerWorld including active power of generators, and active and reactive power of loads
    gen_syn_values = [[gen_syn.loc[i, gen_key_fields[0]], gen_syn.loc[i, gen_key_fields[1]], p_syn_sample[num, i]] for i in range(syn_number)]
    gen_rew_values = [[gen_rew.loc[i, gen_key_fields[0]], gen_rew.loc[i, gen_key_fields[1]], p_rew_sample[num, i]] for i in range(rew_number)]
    load_values = [[load.loc[i, load_key_fields[0]], load.loc[i, load_key_fields[1]], p_load_sample[num, i], q_load_sample[num, i]] for i in range(load_number)]
    saw.ChangeParametersMultipleElement(ObjectType='gen', ParamList=gen_params, ValueList=gen_syn_values)
    saw.ChangeParametersMultipleElement(ObjectType='gen', ParamList=gen_params, ValueList=gen_rew_values)  # Bus7031: 3106
    saw.ChangeParametersMultipleElement(ObjectType='load', ParamList=load_params, ValueList=load_values)
    try:
        saw.SolvePowerFlow() # Run power flow
    except:
       print("Error!")
       

    # Read power flow results
    bus_pf = saw.get_power_flow_results('bus')
    gen_pf = saw.get_power_flow_results('gen')
    branch_pf = saw.get_power_flow_results('branch')
    V_row = bus_pf['BusPUVolt'].values
    print(np.min(V_row))
    Pflow_row = branch_pf['LineMW'].values
    Qflow_row = branch_pf['LineMVR'].values
    Pg_row = gen_pf.loc[gen_closed_idx, :]['GenMW'].values
    Qg_row = gen_pf.loc[gen_closed_idx, :]['GenMVR'].values
    TSI_row = np.zeros((1,Fault_number))

    # Run different faults for the same sample
    for fault_idx in range(Fault_number):
        print('++++++++++++++++++++++++ ' + str(num) + '_' + str(fault_idx) + ' +++++++++++++++++++++++++++++++')

        # Run time-domain simulation for fault with name ctg_name
        ctg_name = 'My Transient Contingency' + str(fault_idx+4)  # 'My Transient Contingency'  ctg['Name'][0]
        cmd = 'TSSolve("{}",[{},{},{},NO])'.format(ctg_name, t0, te, stepsize)
        res = saw.RunScriptCommand(cmd)

        # Read time-domain results
        objFieldList = ["Plot 'Gen_Rotor Angle'"]  # Gen_Rotor Angle  'Plot ''Gen_Rotor Angle 2'''
        result = saw.TSGetContingencyResults(ctg_name, objFieldList)  # "My Transient Contingency" is the contingency name
        angle = result[1]  # result[0] is meta data
        # angle = pd.DataFrame(angle.loc[:, 0:].values, index=angle['time'])
        # angle.plot(legend=None)
        # plt.show(block=True)
        # break
        angle = angle.T.values
        angle = angle[1:, :]
        angle_zeros = [angle[i, :].any() == 0 for i in range(angle.shape[0])]
        angle = np.delete(angle, angle_zeros, 0)

        # Calculate the TSI according to the rotor angles
        max_delta_angle_trajectory = np.zeros(angle.shape[1])
        for i in range(angle.shape[1]):
            max_delta_angle_trajectory[i] = np.max(angle[:, i]) - np.min(angle[:, i])

        max_delta_angle = np.max(max_delta_angle_trajectory)
        TSI_row[:,fault_idx] = 100 * (360 - max_delta_angle) / (360 + max_delta_angle)
        print('TSI:', TSI_row[:, fault_idx])

    saw.exit()

    return num, V_row, Pflow_row, Qflow_row, Pg_row, Qg_row, TSI_row

def LHS(samnum, range, dim):
    samples = lhs(dim, samnum)  # shape = (samnum, dim)
    samples = range[0] + (range[1] - range[0]) * samples
    return samples


if __name__ == "__main__":

    # some user defined parameters
    print(os.cpu_count())
    max_workers = min(2, os.cpu_count())  # Use 2 or the available CPU cores, whichever is smaller

    # number of samples
    num_sample = 20
    # Range of generators
    range_gen = [1.0, 1.4]  # [0.85, 1.05]  # [0.55, 0.95]  # 0.6-1.4, time_cut = 0.17
    range_rew = [0, 1.0]

    t0 = 0.0 # Time-domain simulation start time
    tf = 1.0 # Fault start time
    td = 0.1 # Fault duration
    te = 10.0 # Fault clear time
    stepsize = 0.005 # Time-domain simulation step size
    Fault_number = 1 # Fault number: need preset in the PowerWorld




    # Read PowerWorld project
    PATHcwd = os.getcwd()
    CASE_PATH = PATHcwd + r"\example\ACTIVSg500Rew.pwb"
    print(f"CASE_PATH  - {CASE_PATH}")
    saw = SAW(CASE_PATH, CreateIfNotFound=True)#, UIVisible=True)
    nb = 500
    nl = 599
    ng = 60

    # Read branch data
    branch_key_fields = saw.get_key_field_list('branch')
    branch_fields = saw.GetFieldList('branch')
    branch = saw.GetParametersMultipleElement('branch', branch_key_fields + ['LineR', 'LineX', 'LineC', 'BranchDeviceType'])
    branch['BranchDeviceType'] = ((branch.loc[:, 'BranchDeviceType'] == 'Transformer').values).astype(int)
    branch = branch[['BusNum', 'BusNum:1', 'LineR', 'LineX', 'LineC', 'BranchDeviceType']].values

    # Read gen data
    gen_key_fields = saw.get_key_field_list('gen')
    gen_fields = saw.GetFieldList('gen')
    gen_params = gen_key_fields + ['GenMW']
    gen_params_max = gen_params + ['GenMWMax', 'GenMWMin', 'GenUnitType']
    gen = saw.GetParametersMultipleElement('gen', gen_params_max + ['GenStatus'])
    gen_closed_idx = gen['GenStatus'] == 'Closed'
    gen = gen.loc[gen_closed_idx, gen_params_max]
    gen.reset_index(drop=True, inplace=True)  # Fix Index
    gen_GenMWMax = np.array(gen.GenMWMax)
    gen_number = len(gen)

    # Find synchronous generators
    gen_syn_idx = gen['GenUnitType'] != 'W3 (Wind Turbine, Type 3)'
    gen_syn = gen.loc[gen_syn_idx, gen_params_max]
    gen_syn.reset_index(drop=True, inplace=True)  # Fix Index

    # Find renewable energy generators
    gen_rew_idx = gen['GenUnitType'] == 'W3 (Wind Turbine, Type 3)'
    gen_rew = gen.loc[gen_rew_idx, gen_params_max]
    gen_rew.reset_index(drop=True, inplace=True)  # Fix Index

    # Read load data
    load_key_fields = saw.get_key_field_list('load')
    load_fields = saw.GetFieldList('load')
    load_params = load_key_fields + ['LoadMW', 'LoadMVR']
    load = saw.GetParametersMultipleElement('load', load_params + ['LoadStatus'])
    load_closed_idx = load['LoadStatus'] == 'Closed'
    load = load.loc[load_closed_idx, load_params]
    load.reset_index(drop=True, inplace=True)  # Fix Index
    load_number = len(load)
    load_bus = load['BusNum'].values

    syn_number = len(gen_syn)
    rew_number = len(gen_rew)
    p_rew_max = np.array([])
    p_rew_min = np.array([])


    # Calculate power factor of loads
    pf_load = np.arctan(load['LoadMVR'].values/load['LoadMW'].values)
    # Sample active power of loads using LHS
    p_load_sample = LHS(num_sample, range_gen, load_number) * load['LoadMW'].values
    p_load_sample_sum = np.sum(p_load_sample, axis=1)

    # Sample active power of renewable energy generators using LHS
    p_rew_sample = LHS(num_sample, range_rew, rew_number) * (gen_rew['GenMWMax'].values - gen_rew['GenMWMin'].values) + gen_rew['GenMWMin'].values
    p_rew_sample_max = np.max(p_rew_sample, axis=0)
    p_rew_sample_sum = np.sum(p_rew_sample, axis=1)


    # Sample active power of synchronous generators using LHS
    p_syn_sample = LHS(num_sample, range_gen, syn_number) * gen_syn['GenMW'].values
    p_syn_sample_max = np.max(p_syn_sample, axis=0)
    p_syn_sample_sum = np.sum(p_syn_sample, axis=1)

    # Put data of all generators in a matrix
    gen_GenMW_LHS = np.zeros(ng)
    gen_GenMW_LHS[gen_syn_idx] = p_syn_sample_max
    gen_GenMW_LHS[gen_rew_idx] = p_rew_sample_max
    gen_GenError = gen_GenMWMax - gen_GenMW_LHS

    # Balance the active power between generators and loads for each sample
    multiple = p_load_sample_sum / (p_syn_sample_sum + p_rew_sample_sum)
    p_load_sample = p_load_sample / multiple.reshape(-1, 1)
    p_load_sample_sum = np.sum(p_load_sample, axis=1)
    q_load_sample = p_load_sample * (np.tan(pf_load))
    q_load_sample_sum = np.sum(q_load_sample, axis=1)

    # Check power blance
    p_gen_sample_sum = p_syn_sample_sum + p_rew_sample_sum
    p_balance_sum = p_gen_sample_sum - p_load_sample_sum

    saw.exit()

    # Record power flow results
    V = np.zeros((num_sample, nb))
    Pflow = np.zeros((num_sample, nl))
    Qflow = np.zeros((num_sample, nl))
    Pg = np.zeros((num_sample, ng))
    Qg = np.zeros((num_sample, ng))
    TSI = np.zeros((num_sample, Fault_number))


    # Run in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(run_pw_simulation, range(num_sample), [CASE_PATH] * num_sample,
                            [gen_key_fields] * num_sample, [load_key_fields] * num_sample,
                            [gen_params] * num_sample, [load_params] * num_sample, [gen_closed_idx] * num_sample,
                            [gen_syn] * num_sample, [p_syn_sample] * num_sample, [syn_number] * num_sample,
                            [gen_rew] * num_sample, [p_rew_sample] * num_sample, [rew_number] * num_sample,
                            [load] * num_sample, [p_load_sample] * num_sample, [q_load_sample] * num_sample, [load_number]* num_sample, 
                            [Fault_number] * num_sample, [t0] * num_sample, [te] * num_sample, [stepsize] * num_sample) 

    for num, V_row, Pflow_row, Qflow_row, Pg_row, Qg_row, TSI_row in results:
        V[num, :] = V_row
        Pflow[num, :] = Pflow_row
        Qflow[num, :] = Qflow_row
        Pg[num, :] = Pg_row
        Qg[num, :] = Qg_row
        TSI[num, :] = TSI_row  # Assigning all results properly


    # Save results
    Path_Save = PATHcwd + r'\record_(' + str(tf) + 's_' + str(td) + 's_' + str(te) + 's)_' + str(range_gen[0]) + \
                '_' + str(range_gen[1]) + '_' + str(num_sample) + 'n'
    if not os.path.exists(Path_Save):
        os.makedirs(Path_Save)

    TSI_min = np.min(TSI, 1).reshape(-1, 1)
    Data = np.hstack((p_rew_sample, p_syn_sample, p_load_sample, q_load_sample, TSI_min))
    scio.savemat(Path_Save + r'\data_record.mat', {'Data': Data, 'TSI': TSI})

    scio.savemat(Path_Save + r'\pf.mat', {'V': V, 'Pflow': Pflow, 'Qflow': Qflow, 'Pg': Pg, 'Qg': Qg, 'branch': branch,
                                        'load_bus': load_bus})
