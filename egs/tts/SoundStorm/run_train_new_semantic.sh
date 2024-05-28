# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
 
######## Set Experiment Configuration ###########
# exp_config="$exp_dir/exp_config_base.json"
# exp_name="soundstorm_24k_12layers_debug"
exp_config="$exp_dir/exp_config_16k_emilia_llama_new_semantic.json"
exp_name="soundstorm_16k_kmeans_2048_emilia_50k_llama_new_semantic"

######## Train Model ###########
CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --main_process_port 16033 \
    "${work_dir}"/bins/tts/train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level debug \
    --dataloader_seed 52349 \


CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --main_process_port 16033 \
    "${work_dir}"/bins/tts/train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --resume \
    --log_level debug \
    --dataloader_seed 5219 \


CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --main_process_port 16033 \
    "${work_dir}"/bins/tts/train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --resume \
    --log_level debug \
    --dataloader_seed 53219 \

CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --main_process_port 16033 \
    "${work_dir}"/bins/tts/train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --resume \
    --log_level debug \
    --dataloader_seed 536291 \

CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch --main_process_port 16033 \
    "${work_dir}"/bins/tts/train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --resume \
    --log_level debug \
    --dataloader_seed 543219 \