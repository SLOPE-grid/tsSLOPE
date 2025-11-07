import andes
from andes.utils.paths import get_case
import numpy as np
import os
import time
import scipy.io as scio
import multiprocessing


def dynamic_simulation(n):
    faults = [50, 60]
    slack_idx = 17 - 1

    # Find all the scenarios
    os.chdir('D:/ANDES/wecc/batch_cases')
    matching_files = [f for f in os.listdir('D:/ANDES/wecc/batch_cases') if f.endswith('.xlsx')]

    # For the same scenario, perform faults at different buses
    TSI = np.zeros(len(faults))
    for m, f in enumerate(faults):
        print('+++++++++++++++++++++' + str(n) + '_' + str(f) + '+++++++++++++++++++++')

        andes.config_logger(stream_level=20)
        case = get_case('D:/ANDES/wecc/batch_cases/' + matching_files[n])
        ss = andes.run(case, default_config=True)

        # Step 5-8: PF
        ss.PFlow.config.report = 0  # write output report  0/1
        PF = ss.PFlow.run()
        if PF != True:
            print('Power flow error')

        # Get PF results
        V = ss.Bus.v.v
        Pflow = np.abs(ss.Line.a1.v)
        Pg = np.insert(ss.PV.p.v, slack_idx, ss.Slack.p.v)
        Qg = np.insert(ss.PV.q.v, slack_idx, ss.Slack.q.v)

        # If PF not converge, directly return system is unstable
        if PF == 0:
            return n, PF, V, Pflow, Pg, Qg, -100

        # Step 10-13: TDS
        tc = 1.01
        ss.Fault.alter('tc', 1, tc)
        ss.Fault.alter('bus', 1, f)
        ss.TDS.config.tf = 5  # simulate for 10 seconds
        ss.TDS.config.tstep = 0.01  # integration step size
        ss.TDS.config.criteria = 0  # use criteria to stop simulation if unstable
        ss.TDS.config.no_tqdm = 1  # disable progres bar printing
        ss.TDS.config.fixt = 0  # use fixed step size (1) or variable (0)
        ss.TDS.config.save_mode = 'manual'
        ss.TDS.run()
        FinalT = ss.dae.ts.t[-1]

        # Step 14-15: Check whether the TDS successfully runs until the specified time; otherwise, consider the system unstable.
        if ss.exit_code != 0 or FinalT < ss.TDS.config.tf-0.0001:
            TSI[m] = -100
            print('TDS error ', ss.exit_code, FinalT, ' ', TSI[m])
            continue

        # Step 17: Calculate the TSI
        RotorAngle = np.hstack([ss.dae.ts.x[:, ss.GENCLS.delta.a], ss.dae.ts.x[:, ss.GENROU.delta.a]]) / 3.14159 * 180
        data_len = RotorAngle.shape[0]
        max_dRotorAngle = np.zeros(data_len)
        for i in range(data_len):
            max_dRotorAngle[i] = np.max(RotorAngle[i, :]) - np.min(RotorAngle[i, :])
        TSI[m] = 100 * (360 - np.max(max_dRotorAngle)) / (360 + np.max(max_dRotorAngle))

        if ss.exit_code != 0:
            print('TDS error ', FinalT, ' ', TSI[m])
    return n, PF, V, Pflow, Pg, Qg, TSI


if __name__ == '__main__':
    # Set the faulted bus and create blank variable matrices
    faults = [50, 60]  # [30, 40, 50, 60] # 70 is totally unstable
    num_sample = len([f for f in os.listdir('D:/ANDES/wecc/batch_cases') if f.endswith('.xlsx')])
    n_bus = 179
    n_branch = 263
    n_gen = 29
    slack_idx = 17 - 1
    V = np.zeros((num_sample, n_bus))
    Pflow = np.zeros((num_sample, n_branch))
    Pg = np.zeros((num_sample, n_gen))
    Qg = np.zeros((num_sample, n_gen))
    PF = np.zeros((num_sample, 1))
    TSI = np.zeros((num_sample, len(faults)))

    # Run time-domain simulations in parallel and return the results
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    res = pool.map(dynamic_simulation, range(num_sample))
    for i in range(num_sample):
        PF[i, :], V[i, :], Pflow[i, :], Pg[i, :], Qg[i, :], TSI[i, :] = res[i][1], res[i][2], res[i][3], res[i][4], res[i][5], res[i][6]
    pool.close()
    pool.join()

    # Save the results
    PATHcwd = os.getcwd()
    Path_Save = PATHcwd + r'\record_' + str(num_sample) + 'n'
    if not os.path.exists(Path_Save):
        os.makedirs(Path_Save)

    scio.savemat(Path_Save + r'\record_pf.mat', {'PF': PF, 'V': V, 'Pflow': Pflow, 'Pg': Pg, 'Qg': Qg})

    TSI_min = np.min(TSI, 1).reshape(-1, 1)
    scio.savemat(Path_Save + r'\record_tds.mat', {'TSI': TSI, 'TSI_min': TSI_min})
