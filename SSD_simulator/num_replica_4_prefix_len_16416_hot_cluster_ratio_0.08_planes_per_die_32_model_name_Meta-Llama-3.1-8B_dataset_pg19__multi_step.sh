#!/bin/bash

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_28 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_32 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_36 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_40 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_44 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_48 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_52 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_56 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_60 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_64 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_68 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_72 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_76 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_80 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_28 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_32 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_36 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_40 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_44 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_48 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_52 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_56 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_60 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_64 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_68 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_72 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_76 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_80 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_28 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_32 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_36 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_40 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_44 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_48 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_52 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_56 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_60 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_64 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_68 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_72 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_76 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_80 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_84 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_88 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_92 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_96 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_100 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_104 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_108 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_28 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_32 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_36 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_40 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_44 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_48 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_52 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_56 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_60 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_64 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_68 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_72 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_76 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_80 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_28 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_32 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_36 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_40 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_44 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_48 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_52 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_56 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_60 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_64 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_68 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_72 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_76 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_80 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_28 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_32 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_36 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_40 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_44 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_48 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_52 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_56 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_60 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_64 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_68 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_72 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_76 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_80 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_28 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_32 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_36 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_40 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_44 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_48 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_52 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_56 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_60 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_64 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_68 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_72 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_76 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_80 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_84 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_28 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_32 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_36 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_40 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_44 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_48 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_52 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_56 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_60 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_64 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_68 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_72 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_76 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_80 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_84 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_88 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_92 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_96 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_100 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_104 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_108 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_28 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_32 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_36 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_40 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_44 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_48 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_52 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_56 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_60 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_64 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_68 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_72 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_76 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_80 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_84 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_88 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_92 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_96 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_100 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_104 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_108 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_112 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_116 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_120 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_124 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_128 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_132 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_136 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_140 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_144 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_148 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_152 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_156 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_160 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_164 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_168 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_172 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_176 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_180 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_184 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_188 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_192 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_196 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_200 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_204 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_208 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_212 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_216 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_220 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_224 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_28 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_32 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_36 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_40 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_44 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_48 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_52 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_56 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_60 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_64 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_68 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_72 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_76 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_80 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_84 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_88 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_92 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_96 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_100 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_104 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_108 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_112 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_116 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_120 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_124 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_128 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_132 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_136 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_140 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_144 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_148 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_152 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_156 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_160 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_164 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_168 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_1 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_2 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_3 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_5 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_6 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_7 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_9 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_10 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_11 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_13 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_14 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_15 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_17 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_18 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_19 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_1 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_2 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_3 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_5 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_6 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_7 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_9 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_10 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_11 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_13 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_14 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_15 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_17 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_18 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_19 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_1 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_2 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_3 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_5 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_6 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_7 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_9 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_10 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_11 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_13 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_14 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_15 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_17 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_18 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_19 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_21 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_22 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_23 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_25 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_26 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_27 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_1 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_2 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_3 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_5 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_6 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_7 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_9 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_10 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_11 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_13 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_14 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_15 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_17 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_18 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_19 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_1 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_2 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_3 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_5 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_6 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_7 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_9 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_10 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_11 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_13 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_14 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_15 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_17 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_18 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_19 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_1 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_2 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_3 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_5 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_6 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_7 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_9 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_10 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_11 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_13 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_14 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_15 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_17 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_18 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_19 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_1 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_2 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_3 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_5 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_6 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_7 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_9 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_10 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_11 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_13 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_14 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_15 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_17 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_18 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_19 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_21 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_1 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_2 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_3 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_5 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_6 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_7 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_9 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_10 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_11 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_13 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_14 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_15 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_17 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_18 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_19 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_21 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_22 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_23 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_25 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_26 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_27 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_1 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_2 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_3 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_5 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_6 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_7 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_9 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_10 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_11 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_13 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_14 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_15 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_17 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_18 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_19 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_21 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_22 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_23 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_25 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_26 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_27 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_28 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_29 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_30 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_31 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_32 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_33 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_34 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_35 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_36 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_37 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_38 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_39 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_40 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_41 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_42 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_43 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_44 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_45 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_46 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_47 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_48 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_49 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_50 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_51 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_52 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_53 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_54 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_55 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_56 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_0 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_1 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_2 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_3 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_4 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_5 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_6 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_7 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_8 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_9 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_10 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_11 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_12 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_13 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_14 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_15 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_16 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_17 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_18 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_19 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_20 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_21 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_22 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_23 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_24 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_25 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_26 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_27 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_28 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_29 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_30 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_31 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_32 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_33 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_34 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_35 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_36 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_37 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_38 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_39 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_40 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_41 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_42 \
--prefix_len 16416

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_28 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_32 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_36 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_40 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_44 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_48 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_52 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_56 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_60 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_64 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_68 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_72 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_76 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_80 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_28 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_32 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_36 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_40 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_44 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_48 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_52 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_56 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_60 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_64 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_68 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_72 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_76 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_80 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_28 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_32 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_36 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_40 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_44 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_48 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_52 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_56 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_60 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_64 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_68 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_72 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_76 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_80 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_84 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_88 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_92 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_96 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_100 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_104 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_108 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_28 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_32 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_36 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_40 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_44 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_48 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_52 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_56 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_60 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_64 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_68 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_72 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_76 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_80 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_28 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_32 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_36 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_40 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_44 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_48 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_52 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_56 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_60 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_64 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_68 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_72 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_76 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_80 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_28 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_32 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_36 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_40 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_44 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_48 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_52 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_56 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_60 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_64 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_68 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_72 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_76 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_80 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_28 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_32 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_36 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_40 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_44 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_48 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_52 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_56 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_60 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_64 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_68 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_72 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_76 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_80 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_84 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_28 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_32 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_36 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_40 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_44 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_48 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_52 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_56 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_60 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_64 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_68 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_72 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_76 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_80 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_84 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_88 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_92 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_96 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_100 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_104 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_108 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_28 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_32 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_36 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_40 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_44 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_48 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_52 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_56 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_60 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_64 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_68 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_72 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_76 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_80 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_84 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_88 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_92 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_96 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_100 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_104 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_108 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_112 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_116 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_120 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_124 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_128 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_132 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_136 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_140 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_144 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_148 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_152 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_156 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_160 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_164 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_168 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_172 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_176 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_180 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_184 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_188 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_192 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_196 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_200 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_204 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_208 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_212 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_216 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_220 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_224 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_28 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_32 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_36 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_40 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_44 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_48 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_52 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_56 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_60 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_64 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_68 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_72 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_76 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_80 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_84 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_88 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_92 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_96 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_100 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_104 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_108 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_112 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_116 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_120 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_124 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_128 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_132 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_136 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_140 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_144 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_148 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_152 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_156 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_160 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_164 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_168 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_1 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_2 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_3 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_5 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_6 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_7 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_9 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_10 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_11 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_13 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_14 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_15 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_17 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_18 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_19 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_0_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_1 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_2 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_3 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_5 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_6 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_7 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_9 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_10 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_11 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_13 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_14 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_15 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_17 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_18 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_19 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_1_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_1 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_2 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_3 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_5 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_6 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_7 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_9 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_10 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_11 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_13 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_14 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_15 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_17 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_18 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_19 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_21 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_22 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_23 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_25 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_26 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_2_27 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_1 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_2 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_3 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_5 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_6 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_7 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_9 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_10 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_11 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_13 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_14 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_15 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_17 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_18 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_19 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_3_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_1 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_2 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_3 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_5 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_6 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_7 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_9 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_10 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_11 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_13 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_14 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_15 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_17 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_18 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_19 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_4_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_1 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_2 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_3 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_5 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_6 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_7 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_9 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_10 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_11 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_13 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_14 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_15 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_17 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_18 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_19 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_5_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_1 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_2 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_3 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_5 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_6 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_7 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_9 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_10 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_11 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_13 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_14 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_15 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_17 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_18 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_19 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_6_21 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_1 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_2 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_3 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_5 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_6 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_7 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_9 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_10 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_11 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_13 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_14 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_15 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_17 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_18 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_19 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_21 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_22 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_23 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_25 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_26 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_7_27 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_1 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_2 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_3 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_5 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_6 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_7 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_9 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_10 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_11 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_13 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_14 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_15 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_17 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_18 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_19 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_21 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_22 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_23 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_25 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_26 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_27 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_28 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_29 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_30 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_31 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_32 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_33 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_34 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_35 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_36 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_37 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_38 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_39 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_40 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_41 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_42 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_43 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_44 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_45 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_46 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_47 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_48 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_49 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_50 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_51 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_52 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_53 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_54 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_55 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_8_56 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_0 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_1 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_2 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_3 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_4 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_5 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_6 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_7 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_8 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_9 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_10 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_11 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_12 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_13 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_14 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_15 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_16 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_17 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_18 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_19 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_20 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_21 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_22 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_23 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_24 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_25 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_26 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_27 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_28 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_29 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_30 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_31 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_32 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_33 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_34 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_35 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_36 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_37 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_38 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_39 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_40 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_41 \
--prefix_len 16416 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_16416_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_ \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_kl_no_optimized_cluster32_gamma32 \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.25 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name verify_9_42 \
--prefix_len 16416 \
--hot_cluster_duplicate