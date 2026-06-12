python /xsky/yangxin/data_process_repo/filter/video_quality_pipeline.py \
    -in /xsky/yangnq/LM_data/zgtz_test \
    --dover-weight /xsky/yangxin/data_process_repo/Third_part/DOVER-master/pretrained_weights/DOVER.pth \
    --kvq-weight   /xsky/yangxin/data_process_repo/Third_part/KVQ/weights/KVQ.pth \
    --aes-frames   16 \
    -out /xsky/yangxin/data_process_repo/scores.csv
