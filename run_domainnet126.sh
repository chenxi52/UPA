# export PYTHONPATH=$PYTHONPATH:/home/chenxi/projects/Upt
dataset='domainnet126'
gpu_id=1
output_dir='res/targets/domainnet126'
is_save=True

sel_ratio=0.8
CUDA_VISIBLE_DEVICES=0 python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 \
    --k_val 4 --sel_ratio $sel_ratio --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} \
    --output_src 'res/ckps/source' --run_all False --cos_t 0.1 --net resnet50 --lr 1e-3 --max_epoch 15 --s 2 --t 0 & 
# CUDA_VISIBLE_DEVICES=1 python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 \
#     --k_val 4 --sel_ratio $sel_ratio --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} \
#     --output_src 'res/ckps/source' --run_all False --cos_t 0.1 --net resnet50 --lr 1e-3 --max_epoch 15 --s 2 --t 1 \
#     --warmup_epochs 0&
# CUDA_VISIBLE_DEVICES=7 python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 \
#     --k_val 4 --sel_ratio $sel_ratio --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} \
#     --output_src 'res/ckps/source' --run_all False --cos_t 0.1 --net resnet50 --lr 1e-3 --max_epoch 15 --s 2 --t 3 \
#     --warmup_epochs 0&
# CUDA_VISIBLE_DEVICES=2 python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 \
#     --k_val 4 --sel_ratio $sel_ratio --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} \
#     --output_src 'res/ckps/source' --run_all False --cos_t 0.1 --net resnet50 --lr 1e-3 --max_epoch 15 --s 1 --t 0 \
#     --warmup_epochs 0&
# CUDA_VISIBLE_DEVICES=3 python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 \
#     --k_val 4 --sel_ratio $sel_ratio --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} \
#     --output_src 'res/ckps/source' --run_all False --cos_t 0.1 --net resnet50 --lr 1e-3 --max_epoch 15 --s 0 --t 3 \
#     --warmup_epochs 0&
# CUDA_VISIBLE_DEVICES=6 python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 \
#     --k_val 4 --sel_ratio $sel_ratio --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} \
#     --output_src 'res/ckps/source' --run_all False --cos_t 0.1 --net resnet50 --lr 1e-3 --max_epoch 15 --s 3 --t 1 \
#     --warmup_epochs 0&