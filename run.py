from Mylib.myfuncs import read_yaml, sub_param_for_yaml_file
from classifier.constants import *
from pathlib import Path
import sys
import os
import yaml

params = read_yaml(PARAMS_FILE_PATH)

C = params.data_correction.name

CP = params.data_transformation.data_correction_name
P = params.data_transformation.number

PT = params.model_trainer.data_transformation_name
T = params.model_trainer.model_name

PV = params.model_evaluation_on_train_val.data_transformation_number
MV = params.model_evaluation_on_train_val.model_name
MIV = str(params.model_evaluation_on_train_val.model_index)

CE = params.model_evaluation.data_correction_name
PE = params.model_evaluation.data_transformation_no
E = params.model_evaluation.model_name

replace_dict = {
    "${C}": C,
    "${CP}": CP,
    "${P}": P,
    "${PT}": PT,
    "${T}": T,
    "${PV}": PV,
    "${MV}": MV,
    "${MIV}": MIV,
    "${CE}": CE,
    "${PE}": PE,
    "${E}": E,
}

sub_param_for_yaml_file("config_p.yaml", "config.yaml", replace_dict)
sub_param_for_yaml_file("dvc_p.yaml", "dvc.yaml", replace_dict)


stage_name = sys.argv[1]
do_compel_run = sys.argv[2]
do_compel_run = "--force" if do_compel_run == "y" else ""


os.system(f"dvc repro {do_compel_run} {stage_name}")
