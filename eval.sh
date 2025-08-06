ckpt_path=$1
python compute_fid.py --path ${ckpt_path}  --integration_steps 1 --integration_method euler