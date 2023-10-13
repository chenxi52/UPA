# export PYTHONPATH=$PYTHONPATH:/home/chenxi/projects/Upt
gpu_id=0
output_dir='res/ckps/loss_ablations'
is_save=False
echo $output_dir
dataset='VISDA-C'

# CUDA_VISIBLE_DEVICES=1 python3 upa_v2_amp.py --par_noisy_ent 0.1 --par_su_cl 0 --su_cl_t 0.06 --k_val 3 --sel_ratio 0.6 --knn_times 1 --issave False --output 'res/ckps/test' --dset 'VISDA-C' --output_src 'res/ckps/source-2022'

# for tau_2_i in 0.1
# do 
#     for par_ent_i in 1
#     do
#         for k_val_i in 4
#         do
#             for ratio_i in 0.8
#             do
#                 for knn_times_i in 2
#                 do 
#                     for par_su_cl in 0
#                     do
#                         for par_noisy_cls in 0 1
#                         do 
#                             for dataset in 'VISDA-C'
#                             do
#                             CUDA_VISIBLE_DEVICES=$gpu_id python3 upa_v2_amp.py --par_noisy_ent $par_ent_i --par_su_cl $par_su_cl --su_cl_t $tau_2_i --k_val $k_val_i --sel_ratio $ratio_i --knn_times $knn_times_i --issave ${is_save} --output 'res/ckps/hyper/knn' --dset ${dataset} --output_src 'res/ckps/source'
#                             done
#                         done
#                     done
#                 done
#             done
#         done
#     done
# done
CUDA_VISIBLE_DEVICES=$gpu_id python3 upa_v2_amp.py --par_noisy_ent 0 --par_su_cl 0 --par_noisy_cls 0.3 --su_cl_t 0.1 --k_val 4 --sel_ratio 0.8 --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} --output_src 'res/ckps/source';
CUDA_VISIBLE_DEVICES=$gpu_id python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 0 --par_noisy_cls 0 --su_cl_t 0.1 --k_val 4 --sel_ratio 0.8 --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} --output_src 'res/ckps/source';
CUDA_VISIBLE_DEVICES=$gpu_id python3 upa_v2_amp.py --par_noisy_ent 0 --par_su_cl 1 --par_noisy_cls 0 --su_cl_t 0.1 --k_val 4 --sel_ratio 0.8 --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} --output_src 'res/ckps/source'