# Return similar files
python recommender/return_similar_files.py \
            --query data/bic_bfc_pairs/hive/sid.json \
            --output-recommendations recommendation_results/hive_compressed_file_recommendation_sid.json \
            --output-commit-recommendations recommendation_results/hive_compressed_commit_recommendation_sid.json \
            --repo-name hive \
            --user apache \
            --db-path ../vector_chromadb/02_exp.db \
            --collection-name commit_embeddings