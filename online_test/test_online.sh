# COCO val X101
CUDA_VISIBLE_DEVICES=0 python test_online.py --batch_size=10 --benchmark=COCO --task_type=val --features_type=X101 --features_path=/home/noonisy/local/X101_grid_feats_coco_trainval.hdf5 --annotation_folder=/home/noonisy/data/annotations/captions_val2014.json

# COCO test X101
#CUDA_VISIBLE_DEVICES=0 python test_online.py --batch_size=10 --benchmark=COCO --task_type=test --features_type=X101 --features_path=/home/noonisy/local/X101_grid_feats_coco_test.hdf5 --annotation_folder=/home/noonisy/data/annotations/image_info_test2014.json

# COCO val X152
#CUDA_VISIBLE_DEVICES=0 python test_online.py --batch_size=10 --benchmark=COCO --task_type=val --features_type=X152 --features_path=/home/noonisy/local/X152_grid_feats_coco_trainval.hdf5 --annotation_folder=/home/noonisy/data/annotations/captions_val2014.json

# COCO test X152
#CUDA_VISIBLE_DEVICES=0 python test_online.py --batch_size=10 --benchmark=COCO --task_type=test --features_type=X152 --features_path=/home/noonisy/local/X152_grid_feats_coco_test.hdf5 --annotation_folder=/home/noonisy/data/annotations/image_info_test2014.json

# nocaps val X101
#CUDA_VISIBLE_DEVICES=0 python test_online.py --batch_size=10 --benchmark=nocaps --task_type=val --features_type=X101 --features_path=/home/noonisy/local/X101_grid_feats_nocaps_val.hdf5 --annotation_folder=/home/noonisy/data/annotations/nocaps_val_image_info.json

# nocaps val X152
#CUDA_VISIBLE_DEVICES=0 python test_online.py --batch_size=10 --benchmark=nocaps --task_type=val --features_type=X152 --features_path=/home/noonisy/local/X152_grid_feats_nocaps_val.hdf5 --annotation_folder=/home/noonisy/data/annotations/nocaps_val_image_info.json
