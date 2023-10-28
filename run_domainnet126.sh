# export PYTHONPATH=$PYTHONPATH:/home/chenxi/projects/Upt
dataset='domainnet126'
gpu_id=1
output_dir='res/ckps/'
is_save=True
echo $output_dir
# for par_ent_i in 1
# do
#     for tau_2_i in 0.1
#     do
#         for k_val_i in 2 6 8 10 
#         do
#             for ratio_i in 0.6
#             do
#                 for knn_times_i in 2 
#                 do 
#                     for par_su_cl in 1
#                     do 
#                     CUDA_VISIBLE_DEVICES=$gpu_id python3 upa_v2_amp.py --output_src res/ckps/source --par_noisy_ent $par_ent_i --par_su_cl $par_su_cl --su_cl_t $tau_2_i --k_val $k_val_i --sel_ratio $ratio_i --knn_times $knn_times_i --issave ${is_save} --output ${output_dir} --dset ${dataset}
#                     done
#                 done
#             done
#         done
#     done
# done
# for pair in 20 21 10 03 31 23 12
sel_ratio=0.8
CUDA_VISIBLE_DEVICES=0 python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 \
    --k_val 4 --sel_ratio $sel_ratio --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} \
    --output_src 'res/ckps/source' --run_all False --cos_t 0.1 --net resnet50 --lr 1e-3 --max_epoch 15 --s 2 --t 0 \
    --warmup_epochs 0& 
CUDA_VISIBLE_DEVICES=1 python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 \
    --k_val 4 --sel_ratio $sel_ratio --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} \
    --output_src 'res/ckps/source' --run_all False --cos_t 0.1 --net resnet50 --lr 1e-3 --max_epoch 15 --s 2 --t 1 \
    --warmup_epochs 0&
CUDA_VISIBLE_DEVICES=7 python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 \
    --k_val 4 --sel_ratio $sel_ratio --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} \
    --output_src 'res/ckps/source' --run_all False --cos_t 0.1 --net resnet50 --lr 1e-3 --max_epoch 15 --s 2 --t 3 \
    --warmup_epochs 0&
CUDA_VISIBLE_DEVICES=2 python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 \
    --k_val 4 --sel_ratio $sel_ratio --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} \
    --output_src 'res/ckps/source' --run_all False --cos_t 0.1 --net resnet50 --lr 1e-3 --max_epoch 15 --s 1 --t 0 \
    --warmup_epochs 0&
CUDA_VISIBLE_DEVICES=3 python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 \
    --k_val 4 --sel_ratio $sel_ratio --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} \
    --output_src 'res/ckps/source' --run_all False --cos_t 0.1 --net resnet50 --lr 1e-3 --max_epoch 15 --s 0 --t 3 \
    --warmup_epochs 0&
CUDA_VISIBLE_DEVICES=6 python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 \
    --k_val 4 --sel_ratio $sel_ratio --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} \
    --output_src 'res/ckps/source' --run_all False --cos_t 0.1 --net resnet50 --lr 1e-3 --max_epoch 15 --s 3 --t 1 \
    --warmup_epochs 0&

# CUDA_VISIBLE_DEVICES=1  python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 1 --par_noisy_cls 0.3 --su_cl_t 0.1 \
#     --k_val 4 --sel_ratio $sel_ratio --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} \
#     --output_src 'res/ckps/source' --run_all False --cos_t 0.1 --net resnet50 --lr 1e-3 --max_epoch 15 --s 1 --t 2 \
#     --warmup_epochs 0&
