# MLP train

python train-test/mlp_train.py \
    --lr 0.00001 \
    --epochs 200 \
    --commit-hashes-path data/commit_hashes/commit_hashes_hive.json \
    --train-triplets data/triplets_hive_train_consider_target_file.json \
    --test-triplets data/triplets_hive_test_consider_target_file.json \
    --db-path ../vector_chromadb/01_exp.db \
    --collection-name commit_embeddings \
    --model-save-path parameters/triplet_intraproject_hive.pth \
    --loss-graph-path results/training_loss.png

# MLP test
python train-test/mlp_test.py \
    --params parameters/triplet_intraproject_hive.pth \
    --commit-hashes-path data/commit_hashes/commit_hashes_hive.json \
    --initialdb ../vector_chromadb/01_exp.db \
    --initial-collection-name commit_embeddings \
    --processeddb ../vector_chromadb/02_exp.db \
    --processed-collection-name commit_embeddings \
    --batch-size 1000