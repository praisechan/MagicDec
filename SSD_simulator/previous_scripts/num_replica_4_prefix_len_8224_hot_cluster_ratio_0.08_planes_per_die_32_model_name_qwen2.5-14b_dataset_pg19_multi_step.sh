#!/bin/bash

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_0 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_1 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_2 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_3 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_4 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_5 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_6 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_7 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_8 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_9 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_10 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_11 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_12 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_13 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_14 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_15 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_16 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_17 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_18 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_19 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_20 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_21 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_22 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_23 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_24 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_25 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_26 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_27 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_0 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_1 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_2 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_3 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_4 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_5 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_6 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_7 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_8 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_9 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_10 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_11 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_12 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_13 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_14 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_15 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_16 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_17 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_18 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_19 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_20 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_0 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_1 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_2 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_3 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_4 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_5 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_6 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_7 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_8 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_9 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_10 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_11 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_12 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_13 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_14 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_15 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_16 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_17 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_18 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_19 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_20 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_21 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_0 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_1 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_2 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_3 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_4 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_5 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_6 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_7 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_8 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_9 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_10 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_11 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_12 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_13 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_14 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_15 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_16 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_17 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_18 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_19 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_20 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_21 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_0 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_1 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_2 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_3 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_4 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_5 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_6 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_7 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_8 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_9 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_10 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_11 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_12 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_13 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_14 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_15 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_16 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_17 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_18 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_19 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_20 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_0 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_1 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_2 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_3 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_4 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_5 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_6 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_7 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_8 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_9 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_10 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_11 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_12 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_13 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_14 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_15 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_16 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_17 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_18 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_19 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_20 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_0 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_1 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_2 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_3 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_4 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_5 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_6 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_7 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_8 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_9 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_10 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_11 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_12 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_13 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_14 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_15 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_16 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_17 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_18 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_19 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_20 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_21 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_0 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_1 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_2 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_3 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_4 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_5 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_6 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_7 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_8 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_9 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_10 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_11 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_12 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_13 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_14 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_15 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_16 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_17 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_18 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_19 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_20 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_21 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_22 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_23 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_24 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_25 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_26 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_27 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_0 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_1 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_2 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_3 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_4 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_5 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_6 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_7 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_8 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_9 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_10 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_11 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_12 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_13 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_14 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_15 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_16 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_17 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_18 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_19 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_20 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_0 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_1 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_2 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_3 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_4 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_5 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_6 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_7 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_8 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_9 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_10 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_11 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_12 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_13 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_14 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_15 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_16 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_17 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_18 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_19 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_20 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_21 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_22 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_23 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_24 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_25 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_26 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_27 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_28 \
--prefix_len 8224

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_0 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_1 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_2 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_3 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_4 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_5 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_6 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_7 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_8 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_9 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_10 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_11 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_12 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_13 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_14 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_15 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_16 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_17 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_18 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_19 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_20 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_21 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_22 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_23 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_24 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_25 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_26 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_0_27 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_0 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_1 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_2 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_3 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_4 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_5 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_6 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_7 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_8 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_9 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_10 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_11 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_12 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_13 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_14 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_15 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_16 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_17 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_18 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_19 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_1_20 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_0 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_1 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_2 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_3 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_4 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_5 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_6 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_7 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_8 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_9 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_10 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_11 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_12 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_13 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_14 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_15 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_16 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_17 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_18 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_19 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_20 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_2_21 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_0 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_1 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_2 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_3 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_4 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_5 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_6 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_7 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_8 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_9 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_10 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_11 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_12 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_13 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_14 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_15 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_16 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_17 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_18 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_19 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_20 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_3_21 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_0 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_1 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_2 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_3 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_4 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_5 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_6 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_7 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_8 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_9 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_10 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_11 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_12 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_13 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_14 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_15 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_16 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_17 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_18 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_19 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_4_20 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_0 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_1 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_2 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_3 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_4 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_5 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_6 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_7 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_8 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_9 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_10 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_11 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_12 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_13 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_14 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_15 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_16 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_17 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_18 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_19 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_5_20 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_0 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_1 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_2 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_3 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_4 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_5 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_6 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_7 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_8 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_9 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_10 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_11 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_12 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_13 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_14 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_15 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_16 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_17 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_18 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_19 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_20 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_6_21 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_0 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_1 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_2 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_3 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_4 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_5 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_6 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_7 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_8 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_9 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_10 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_11 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_12 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_13 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_14 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_15 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_16 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_17 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_18 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_19 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_20 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_21 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_22 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_23 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_24 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_25 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_26 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_7_27 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_0 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_1 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_2 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_3 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_4 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_5 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_6 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_7 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_8 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_9 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_10 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_11 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_12 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_13 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_14 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_15 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_16 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_17 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_18 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_19 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_8_20 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_0 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_1 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_2 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_3 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_4 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_5 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_6 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_7 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_8 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_9 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_10 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_11 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_12 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_13 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_14 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_15 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_16 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_17 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_18 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_19 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_20 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_21 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_22 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_23 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_24 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_25 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_26 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_27 \
--prefix_len 8224 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_8224_hot_cluster_ratio_0.08_planes_per_die_32_model_name_qwen2.5-14b_dataset_pg19 \
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
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name qwen2.5-14b \
--dataset pg19 \
--generate_name verify_9_28 \
--prefix_len 8224 \
--hot_cluster_duplicate