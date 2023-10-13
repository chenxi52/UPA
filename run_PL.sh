dataset='VISDA-C'
gpu_id=0
output_dir='res/ckps/test'
is_save=False
echo $output_dir
# CUDA_VISIBLE_DEVICES=0 python3 upa_v2.py --par_noisy_ent 0.1 --su_cl_t 0.06 --k_val 3 --sel_ratio 0.6 --knn_times 1 --issave False --output 'res/ckps/fine_param' --dset 'office'


for par_ent_i in 1
do
    for tau_2_i in 0.1
    do
        for k_val_i in 4
        do
            for ratio_i in 0.8
            do
                for PL in 'SSPL_ent' 'SSPL_maxp' 'SSPL_cossim' 
                do 
                    for par_su_cl in 1
                    do 
                    CUDA_VISIBLE_DEVICES=$gpu_id python3 image_target_PL.py --output_src res/ckps/source --par_noisy_ent $par_ent_i --par_su_cl $par_su_cl --su_cl_t $tau_2_i --k_val $k_val_i --sel_ratio $ratio_i --PL $PL --issave ${is_save} --dset ${dataset} 
                    done
                done
            done
        done
    done
done


for par_ent_i in 1
do
    for tau_2_i in 0.1
    do
        for k_val_i in 4
        do
            for ratio_i in 0.8
            do
                for PL in 'GMM_JMDS'
                do 
                    for par_su_cl in 0
                    do 
                    CUDA_VISIBLE_DEVICES=$gpu_id python3 image_target_PL.py --run_all False --output_src res/ckps/source --par_noisy_ent $par_ent_i --par_su_cl $par_su_cl --su_cl_t $tau_2_i --k_val $k_val_i --sel_ratio $ratio_i --PL $PL --issave ${is_save} --dset ${dataset} 
                    done
                done
            done
        done
    done
done