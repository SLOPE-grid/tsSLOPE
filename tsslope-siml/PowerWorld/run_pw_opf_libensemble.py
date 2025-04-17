import os
import random
import numpy as np

from libensemble import Ensemble
from libensemble.libE import libE
from libensemble.tools import parse_args, add_unique_random_streams
from libensemble.gen_funcs.sampling import uniform_random_sample
from libensemble.specs import ExitCriteria, GenSpecs, LibeSpecs, SimSpecs

from run_pw_opf import run_pw_opf  

def run_opf(H, persis_info, sim_specs, libE_info):
    worker_id = libE_info["workerID"] 

    batch = len(H["x"])  # Num evaluations each sim_f call.
    H_o = np.zeros(batch, dtype=sim_specs["out"])  # Define output array H
#    print(f"batch = {batch}, W={worker_id} --- {H["x"]}")

    for i, caseidin in enumerate(H["x"]):
#        print(f"i={i}, W {worker_id} --- {caseidin}")
        caseid = int(caseidin)
        case_file  = sim_specs["user"]["casefile_pool"][caseid]      
        success = run_pw_opf(case_file, worker_id=worker_id)  # Function evaluations placed into H
        H_o["success"][i]  = success
        H_o["case_file"][i]  = case_file

    return H_o, persis_info


def pick_case_from_pool(_, persis_info, gen_specs):
    ub = gen_specs["user"]["ub"]
    lb = gen_specs["user"]["lb"]

    n = len(lb)
    b = gen_specs["user"]["gen_batch_size"]

    H_o = np.zeros(b, dtype=gen_specs["out"])

    H_o["x"] = persis_info["rand_stream"].integers(lb, ub, (b, n))
    print(f"batch_size = {b}, n_var = {n} --- {H_o["x"] }")

    return H_o, persis_info


if __name__ == "__main__":
    nworkers = 2

    casefile_pool = [
        "C:/Users/chiang7/OneDrive - LLNL/SHARE/Scidac/workflow/example/ACTIVSg200.pwb",
        "C:/Users/chiang7/OneDrive - LLNL/SHARE/Scidac/workflow/example/ACTIVSg200_copy.pwb",
        "C:/Users/chiang7/OneDrive - LLNL/SHARE/Scidac/workflow/example/ACTIVSg500.pwb"
    ]

    libE_specs = LibeSpecs(nworkers=nworkers, comms="local", workflow_dir_path="./example",use_workflow_dir=True)

    sim_specs = SimSpecs(
        sim_f=run_opf,
        inputs=["x"],
        outputs=[("success", bool), ("case_file", "U250")],
        user={
            "casefile_pool": casefile_pool,  
        },
    )

    gen_specs = GenSpecs(
        gen_f=pick_case_from_pool,
        outputs=[("x", int, 1)],
        user={
            "gen_batch_size": 3,
            "lb": np.array([0]),
            "ub": np.array([2]), #Upper bound is exclusive
        },
    )

    exit_criteria = ExitCriteria(sim_max=6)

    sampling = Ensemble(
            libE_specs=libE_specs,
            sim_specs=sim_specs,
            gen_specs=gen_specs,
            exit_criteria=exit_criteria,
    )

    sampling.add_random_streams()
    H, persis_info, flag = sampling.run()

    if sampling.is_manager:
        print("Done!")