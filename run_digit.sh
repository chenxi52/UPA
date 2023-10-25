
# python uda_digit.py --dset u2m --gpu_id 3 --par_cls 0.1 --output res/ckps_digits --k_val 4 --sel_ratio 0.6 --seed 2021&
# python uda_digit.py --dset s2m --gpu_id 2 --par_cls 0.1 --output res/ckps_digits --k_val 4 --sel_ratio 0.6 --seed 2021&

wait 4477 4478
python uda_digit.py --dset m2u --gpu_id 2 --par_cls 0.1 --output res/ckps_digits --k_val 4 --sel_ratio 0.6 --seed 2022&
python uda_digit.py --dset u2m --gpu_id 1 --par_cls 0.1 --output res/ckps_digits --k_val 4 --sel_ratio 0.6 --seed 2022;

python uda_digit.py --dset s2m --gpu_id 1 --par_cls 0.1 --output res/ckps_digits --k_val 4 --sel_ratio 0.6 --seed 2022&
python uda_digit.py --dset m2u --gpu_id 2 --par_cls 0.1 --output res/ckps_digits --k_val 4 --sel_ratio 0.6 --seed 2021;

python uda_digit.py --dset s2m --gpu_id 2 --par_cls 0.1 --output res/ckps_digits --k_val 4 --sel_ratio 0.6 --seed 2020&
python uda_digit.py --dset u2m --gpu_id 1 --par_cls 0.1 --output res/ckps_digits --k_val 4 --sel_ratio 0.6 --seed 2020
