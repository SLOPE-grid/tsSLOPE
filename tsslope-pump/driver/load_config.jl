
path_to_exajugo = "/p/lustre1/chiang7/tsi_simulation/exajugo"
path_to_tsslope = "/p/lustre1/chiang7/tsi_simulation/tsSLOPE/tsslope-pump"

model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/example/model_state_DSPP_500_0.990_0.884_28.9_9.0_4.2_20000_318.0_300_6_0.021_0.00075.pth"
data_record =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/example/data_record.mat"
pf_limit_file = "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/example/pf_new.mat"

CNN_model_path =  "/p/lustre1/chiang7/tsi_simulation/tsSLOPE/tsslope-ml/CNN1D_SiLU_acc_0.92378.pth"
CNF_model_path =  "/p/lustre1/chiang7/tsi_simulation/tsSLOPE/tsslope-ml/cdf_flow_checkpoint.pt"
LLNL_data_record =  "/p/lustre1/chiang7/tsi_simulation/tsSLOPE/tsslope-ml/combined_llnl_data.mat"

case_path = "/p/lustre1/chiang7/tsi_simulation/tsSLOPE/example/ACTIVSg500"
#case_path = "/p/lustre1/chiang7/tsi_simulation/exajugo/examples/uqgrid_9bus"

case_sol_path = "/p/lustre1/chiang7/tsi_simulation/data_generated/500bus/acopf_sol_load_pert_uniform_0dot15_20260205"
#case_sol_path = "/p/lustre1/chiang7/tsi_simulation/data_generated/9bus_1sample_test/acopf"

#ENV["PYTHON"] = "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-pump/pyenv/bin/python3"
ENV["PYTHON"] = "/usr/WS2/hiop/dane/venv/hiopbbpy-ci/bin/python3"
