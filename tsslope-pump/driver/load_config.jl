
path_to_exajugo = "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/exajugo"
path_to_tsslope = "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-pump"

model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/example/model_state_DSPP_500_0.990_0.884_28.9_9.0_4.2_20000_318.0_300_6_0.021_0.00075.pth"
data_record =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/example/data_record.mat"
pf_limit_file = "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/example/pf_new.mat"

# CNN_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/model_state_CNN1D_Float_64_acc_0.91017.pth"
CNN_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/model_state_CNN1D_Float_64_acc_0.92344.pth"
CNN_silu_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/CNN1D_SiLU_acc_0.92378.pth"
CNN_silu_no_sig_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/CNN1D_SiLU_No_Sig_acc_0.91664.pth"
UQ_CNN_silu_no_sig_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/UQ_CNN1D_SiLU_No_Sig_acc_0.91936.pth"
UQ_CNN_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/model_state_UQ_CNN1D_Float_64_acc_0.91834_rmse_0.261615_mae_0.104813.pth"
# UQ_CNN_SiLU_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/UQ_CNN1D_SiLU_acc_0.90949.pth"
# UQ_CNN_SiLU_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/UQ_CNN1D_SiLU_diff_Loss_acc_0.90575.pth"
UQ_CNN_SiLU_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/UQ_CNN1D_SiLU_diff_Loss_acc_0.89758.pth"
# UQ_CNN_STD_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/UQ_CNN1D_STD_acc_0.90643.pth"
UQ_CNN_STD_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/UQ_CNN1D_STD_acc_0.89929.pth"
# UQ_CNN_STD_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/UQ_CNN1D_STD_NLL_acc_0.90915.pth"
# UQ_CNN_SiLU_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/UQ_CNN1D_Adpt_LR_acc_0.90881.pth"
# UQ_CNN_SiLU_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/UQ_CNN1D_acc_0.91664.pth"
CNF_model_path =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/cdf_flow_checkpoint.pt"
LLNL_data_record =  "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-ml/combined_llnl_data.mat"

case_path = "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/example/ACTIVSg500"

case_sol_path = "./output"

ENV["PYTHON"] = "/usr/workspace/hiop/dane/project/scidac_2025/irabiel/WF_ACOPF/tsSLOPE/tsslope-pump/pyenv/bin/python3"
