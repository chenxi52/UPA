dataset='VISDA-C'
output_dir=res/ckps/source/
python3 train_source.py --output $output_dir --dset ${dataset} --trte val --net resnet101 --lr 1e-3 --max_epoch 10 --s 0;
