# export PYTHONPATH=$PYTHONPATH:/home/chenxi/projects/Upt
gpu_id=0
output_dir='res/target/visda-c/'
is_save=False
dataset='VISDA-C'

CUDA_VISIBLE_DEVICES=$gpu_id python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 --k_val 4 --sel_ratio 0.8 --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} --output_src 'res/ckps/source';
