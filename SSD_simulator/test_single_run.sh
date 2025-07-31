#!/bin/bash

python simulator.py \
--max_latency_calculate \
--csv_path test_output \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 16 \
--window_size 64 \
--layer_num 48 \
--profiling_dir /home/juchanlee/MagicDec/profile/data/ \
--num_replica 4 \
--budget_ratio 0.02 \
--hot_cluster_ratio 0.01 \
--planes_per_die 32 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name speculate_0_0 \
--prefix_len 8224
