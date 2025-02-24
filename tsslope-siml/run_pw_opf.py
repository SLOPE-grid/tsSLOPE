import os
import pythoncom
import win32com.client
import numpy as np

def run_pw_opf(case_file, worker_id=0):

    retval = []
    pythoncom.CoInitialize()
    simauto = win32com.client.Dispatch("pwrworld.SimulatorAuto")

    # Open a case file
    result = simauto.OpenCase(case_file)
        
    if result[0] != '':  # Check for errors
        print(f"Error opening case: {result[0]}")

    # Solve PF/OPF
    #result = simauto.RunScriptCommand("SolvePowerFlow;")
    result = simauto.RunScriptCommand("SolvePrimalLP")
    if result[0] != '':
        print(f"Workerid {worker_id} --- Error solving power flow: {result[0]}")
        success = False
    else:
        print(f"Workerid {worker_id} --- Successful to solve OPF: {case_file}")

        
        results_bus = simauto.GetParametersMultipleElement(
            "bus", ["BusNum", "BusPUVolt"], ""
        )
    #    print("OPF Results (Bus Voltage):", results_bus)

        results_gen = simauto.GetParametersMultipleElement(
        "gen", ["BusNum", "GenMW", "GenMVR"], ""
        )
    #    print("OPF Results (Generator Dispatch):", results_gen)
    #    retval.append({"success": True, "opf_result": results_gen, "case_file": case_file})
        success = True
        

    simauto.CloseCase()
    pythoncom.CoUninitialize()
    
    return np.array(np.array([(success, case_file)], dtype=[("success", bool), ("case_file", str)]))
