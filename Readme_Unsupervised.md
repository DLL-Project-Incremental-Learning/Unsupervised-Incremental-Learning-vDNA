
#### Important things to run

1. Test_only setup for Cityscapes

python main2.py --data_root datasets/data/cityscapes --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth --model deeplabv3plus_resnet101 --test_only --crop_val --val_batch_size 2

2. Initial Weak labeling

# 003 data
python predict.py --input datasets/data/KITTI-360/data_2d_raw/2013_05_28_drive_0003_sync/image_00/data_rect/ --dataset cityscapes --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth --save_val_results_to test_results/2013_05_28_drive_0003_sync_labelid --model deeplabv3plus_resnet101

# 007
python predict.py --input datasets/data/KITTI-360/data_2d_raw/2013_05_28_drive_0007_sync/image_00/data_rect/ --dataset cityscapes --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth --save_val_results_to test_results/2013_05_28_drive_0007_sync_labelid --model deeplabv3plus_resnet101




3. FineTune script:
python finetuning.py --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth --model deeplabv3plus_resnet101 --batch_size 2 --val_batch_size 2


4. Weak labels

## On Train data for bucket_idx 0

python predict2.py --bucketidx 0 --dataset cityscapes --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth --save_val_results_to weak_labels --model deeplabv3plus_resnet101 --val_data False

## On Val data fpr bucket_idx 0

python predict2.py --bucketidx 0 --dataset cityscapes --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth --save_val_results_to weak_labels --model deeplabv3plus_resnet101 --val_data True


5. Bucket Fine Tuning
python finetuning2.py --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth --model deeplabv3plus_resnet101 --batch_size 2 --val_batch_size 2 --bucketidx 0


