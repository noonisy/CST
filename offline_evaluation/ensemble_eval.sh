# COCO val X101
CUDA_VISIBLE_DEVICES=0 python ensemble_eval.py --batch_size=10 --features_type=X101 --features_path=/home/noonisy/local/X101_grid_feats_coco_trainval.hdf5 --annotation_folder=/home/noonisy/data/annotations

# COCO val X152
#CUDA_VISIBLE_DEVICES=0 python ensemble_eval.py --batch_size=10 --features_type=X152 --features_path=/home/noonisy/local/X152_grid_feats_coco_trainval.hdf5 --annotation_folder=/home/noonisy/data/annotations
