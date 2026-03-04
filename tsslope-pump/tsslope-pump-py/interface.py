
import os
import numpy as np
import torch

from .d2TSI_dV2dP2      import d2TSI_dV2dP2
from .dTSI_dVdP         import dTSI_dVdP
from .TSI_constraint    import TSI_constraint
from .SR1_approx        import hess_approx


from .spectral_analysis    import analyze_hessian

#from .load_case         import load_case
from .load_Surrogate    import load_surrogate


def load_model(model_path, data_record, model_type, active_gen_only = True):
   return load_surrogate(model_path, data_record, model_type, active_gen_only = active_gen_only)


#def load_config(*_):

#    return (*_)


def eval_tsi_g(*_):

    return dTSI_dVdP(*_)
 

def eval_tsi_h(*_):

    return d2TSI_dV2dP2(*_)

def eval_tsi_h_approx(*_):

    return hess_approx(*_)

def analy_h(*_):

    return analyze_hessian(*_)

def eval_tsi_f(*_):
    return TSI_constraint(*_)

def eval_tsi_f2(*_):
    return 1

def load():

   src_dir = os.path.dirname(os.path.abspath(__file__))
   c_dir = os.path.join(src_dir, "data")

   model_path = os.path.join(c_dir, "model_state_CNN1D_acc_1.00000_rmse_0.003557_mae_0.000904_epoch_10_bs_32_lr_0.001_time_14.796.pth")
   data_record = os.path.join(c_dir,"combined_llnl_data.mat")
#    model_path = os.path.join(c_dir, "model_state_DSPP_500_0.990_0.884_28.9_9.0_4.2_20000_318.0_300_6_0.021_0.00075.pth")
#    data_record = os.path.join(c_dir,"data_record.mat")

   model_type = "CNN"

   GPmodel, data, TSI = load_model(model_path, data_record, model_type)

   st_args = load_config(os.path.join(c_dir, "case_ACTIVSg500.mat"), os.path.join(c_dir, "pf_new.mat"))

   return GPmodel, st_args

