
import os
import numpy as np
import torch

from .d2TSI_dV2dP2      import d2TSI_dV2dP2
from .dTSI_dVdP         import dTSI_dVdP
from .TSI_constraint    import TSI_constraint

#from .load_case         import load_case
from .load_GPmodel      import load_GPmodel


def load_model(model_path, data_record):
   return load_GPmodel(model_path, data_record)


#def load_config(*_):

#    return load_case(*_)


def eval_tsi_g(*_):

    return dTSI_dVdP(*_)
 

def eval_tsi_h(*_):

    return d2TSI_dV2dP2(*_)

def eval_tsi_f(*_):
    return TSI_constraint(*_)

def eval_tsi_f2(*_):
    return 1

def load():

   src_dir = os.path.dirname(os.path.abspath(__file__))
   c_dir = os.path.join(src_dir, "data")

   model_path = os.path.join(c_dir, "model_state_DSPP_500_0.990_0.884_28.9_9.0_4.2_20000_318.0_300_6_0.021_0.00075.pth")
   data_record = os.path.join(c_dir,"data_record.mat")

   GPmodel, data, TSI = load_model(model_path, data_record)

   st_args = load_config(os.path.join(c_dir, "case_ACTIVSg500.mat"), os.path.join(c_dir, "pf_new.mat"))

   return GPmodel, st_args

