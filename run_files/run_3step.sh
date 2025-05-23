CUDA_VISIBLE_DEVICES=0 python tests/RetrievalAttention/run_3step.py \
--model_name llama-3.1-8b \
--B 1 \
--attn_type RetroInfer \
--gamma1 8 \
--gamma2 30 \
--budget1 0.01 \
--budget2 0.25 \
--dataset pg19 \
--prefix_len 32800
