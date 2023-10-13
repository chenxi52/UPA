dataset='office'
gpu_id=0
output_dir='res/ckps/loss_ablations'
is_save=False
echo $output_dir
# CUDA_VISIBLE_DEVICES=0 python3 upa_v2.py --par_noisy_ent 0.1 --su_cl_t 0.06 --k_val 3 --sel_ratio 0.6 --knn_times 1 --issave False --output 'res/ckps/fine_param' --dset 'office'


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
CUDA_VISIBLE_DEVICES=$gpu_id python3 upa_v2_amp.py --par_noisy_ent 0 --par_su_cl 0 --par_noisy_cls 0.3 --su_cl_t 0.1 --k_val 4 --sel_ratio 0.6 --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} --output_src 'res/ckps/source';
CUDA_VISIBLE_DEVICES=$gpu_id python3 upa_v2_amp.py --par_noisy_ent 1 --par_su_cl 0 --par_noisy_cls 0 --su_cl_t 0.1 --k_val 4 --sel_ratio 0.6 --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} --output_src 'res/ckps/source';
CUDA_VISIBLE_DEVICES=$gpu_id python3 upa_v2_amp.py --par_noisy_ent 0 --par_su_cl 1 --par_noisy_cls 0 --su_cl_t 0.1 --k_val 4 --sel_ratio 0.6 --knn_times 2 --issave ${is_save} --output $output_dir --dset ${dataset} --output_src 'res/ckps/source'
