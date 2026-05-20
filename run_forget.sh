unlearn_type=RL
dataset_type=cifar10 #TinyImagenet  #cifar100 
model_arch=vgg16_bn_lth  # vgg16_bn_lth  #resnet18
ForgetType=LT
seed=4307
# NumIndexes=9000     # I changed from 9000 to 5000
NumIndexes=5000      # as I prev run code for 5000 so that masks,retraining,unlearning will mismatch
unlearn_lr=0.01
# gpu=6            # I changed from gpu 6 to 0
gpu=0         # bcz as I'm running in colab it has only single gpu
alpha=0.1
beta=1
epoch_num=10
mask_rate=0.5
other_info=
data_path=./data
python main_random.py --data ${data_path} --arch ${model_arch} --dataset ${dataset_type} --unlearn ${unlearn_type} --unlearn_epochs ${epoch_num} --unlearn_lr ${unlearn_lr} --num_indexes_to_replace ${NumIndexes} --model_path ./checkpoints/scratch/${seed}/${dataset_type}_${model_arch}/0checkpoint.pth.tar --save_dir ./checkpoints/unlearn/${unlearn_type}/${seed}/${dataset_type}_${model_arch}_${NumIndexes}_rl_${ForgetType}${other_info}/  --mask_path ./checkpoints/mask/${seed}/${dataset_type}_${model_arch}_${NumIndexes}_${ForgetType}/with_${mask_rate}.pt --seed ${seed} --gpu ${gpu} --batch_size 512 --sample_forget_type ${ForgetType} --alpha ${alpha}