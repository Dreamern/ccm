gpu_num=$1
ckpt_path=$2
task=$3
dataset=$4
torchrun --nproc-per-node=$gpu_num test_model.py $ckpt_path $task $dataset