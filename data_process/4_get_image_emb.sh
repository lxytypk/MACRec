
export CUDA_VISIBLE_DEVICES=1

python clip_feature.py \
    --image_root /home/liangxinyu/MACRec/data/amazon18/Images \
    --save_root /home/liangxinyu/MACRec/data \
    --model_cache_dir /home/liangxinyu/MACRec/cache_models/clip \
    --dataset Instruments


