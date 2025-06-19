python train-test/mlp_train.py \
    --lr 0.00001 \
    --epochs 200 \
    --commit-hashes-path data/commit_hashes/commit_hashes_hive.json \
    --train-triplets data/triplets_hive_train_consider_target_file.json \
    --test-triplets data/triplets_hive_test_consider_target_file.json \
    --db-path ../embedding_db/01_exp.db \
    --model-save-path parameters/triplet_intraproject_hive.pth \
    --loss-graph-path results/training_loss.png