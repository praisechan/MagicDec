#!/bin/bash

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_0 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_4 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_8 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_12 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_16 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_20 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_24 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_28 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_32 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_36 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_40 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_44 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_48 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_52 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_56 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_60 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_64 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_68 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_72 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_76 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_0 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_4 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_8 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_12 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_16 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_20 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_24 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_28 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_32 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_36 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_40 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_44 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_48 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_52 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_56 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_60 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_64 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_68 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_72 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_76 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_0 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_4 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_8 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_12 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_16 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_20 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_24 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_28 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_32 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_36 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_40 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_44 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_48 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_52 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_56 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_60 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_64 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_68 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_72 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_76 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_0 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_4 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_8 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_12 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_16 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_20 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_24 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_28 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_32 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_36 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_40 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_44 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_48 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_52 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_56 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_60 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_64 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_68 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_72 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_76 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_80 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_84 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_0 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_4 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_8 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_12 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_16 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_20 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_24 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_28 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_32 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_36 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_40 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_44 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_48 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_52 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_56 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_60 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_64 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_68 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_72 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_76 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_80 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_0 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_4 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_8 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_12 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_16 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_20 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_24 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_28 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_32 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_36 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_40 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_44 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_48 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_52 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_56 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_60 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_64 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_68 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_72 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_76 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_80 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_84 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_88 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_0 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_4 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_8 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_12 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_16 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_20 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_24 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_28 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_32 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_36 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_40 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_44 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_48 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_52 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_56 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_60 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_64 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_68 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_72 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_76 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_80 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_84 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_88 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_0 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_4 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_8 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_12 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_16 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_20 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_24 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_28 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_32 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_36 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_40 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_44 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_48 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_52 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_56 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_60 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_64 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_68 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_72 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_76 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_80 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_0 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_4 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_8 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_12 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_16 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_20 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_24 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_28 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_32 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_36 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_40 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_44 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_48 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_52 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_56 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_60 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_64 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_68 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_72 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_76 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_80 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_84 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_0 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_4 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_8 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_12 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_16 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_20 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_24 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_28 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_32 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_36 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_40 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_44 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_48 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_52 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_56 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_60 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_64 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_68 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_72 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_76 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_80 \
--prefix_len 32800

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_0 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_4 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_8 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_12 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_16 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_20 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_24 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_28 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_32 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_36 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_40 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_44 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_48 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_52 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_56 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_60 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_64 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_68 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_72 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_0_76 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_0 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_4 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_8 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_12 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_16 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_20 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_24 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_28 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_32 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_36 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_40 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_44 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_48 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_52 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_56 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_60 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_64 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_68 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_72 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_1_76 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_0 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_4 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_8 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_12 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_16 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_20 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_24 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_28 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_32 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_36 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_40 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_44 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_48 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_52 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_56 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_60 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_64 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_68 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_72 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_2_76 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_0 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_4 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_8 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_12 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_16 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_20 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_24 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_28 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_32 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_36 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_40 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_44 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_48 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_52 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_56 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_60 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_64 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_68 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_72 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_76 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_80 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_3_84 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_0 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_4 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_8 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_12 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_16 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_20 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_24 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_28 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_32 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_36 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_40 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_44 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_48 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_52 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_56 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_60 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_64 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_68 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_72 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_76 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_4_80 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_0 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_4 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_8 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_12 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_16 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_20 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_24 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_28 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_32 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_36 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_40 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_44 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_48 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_52 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_56 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_60 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_64 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_68 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_72 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_76 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_80 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_84 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_5_88 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_0 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_4 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_8 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_12 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_16 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_20 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_24 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_28 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_32 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_36 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_40 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_44 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_48 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_52 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_56 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_60 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_64 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_68 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_72 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_76 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_80 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_84 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_6_88 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_0 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_4 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_8 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_12 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_16 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_20 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_24 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_28 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_32 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_36 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_40 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_44 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_48 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_52 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_56 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_60 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_64 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_68 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_72 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_76 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_7_80 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_0 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_4 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_8 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_12 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_16 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_20 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_24 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_28 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_32 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_36 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_40 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_44 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_48 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_52 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_56 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_60 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_64 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_68 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_72 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_76 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_80 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_8_84 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_0 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_4 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_8 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_12 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_16 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_20 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_24 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_28 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_32 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_36 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_40 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_44 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_48 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_52 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_56 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_60 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_64 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_68 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_72 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_76 \
--prefix_len 32800 \
--hot_cluster_duplicate

echo "Processing next configuration..."

python simulator.py \
--max_latency_calculate \
--csv_path num_replica_4_prefix_len_32800_hot_cluster_ratio_0.08_planes_per_die_32_model_name_Meta-Llama-3.1-8B_dataset_pg19_for_0.1 \
--num_channels 1 \
--chips_per_channel 1 \
--dies_per_chip 1 \
--page_size_bytes 16384 \
--vector_bytes 4 \
--flash_read_latency_us 50 \
--num_heads 8 \
--cluster_size 32 \
--window_size 64 \
--layer_num 32 \
--profiling_dir /home/juchanlee/MagicDec/profile/data_2step_for_0.1/ \
--num_replica 4 \
--hot_cluster_ratio 0.08 \
--planes_per_die 32 \
--budget_ratio 0.02 \
--model_name Meta-Llama-3.1-8B \
--dataset pg19 \
--generate_name speculate_9_80 \
--prefix_len 32800 \
--hot_cluster_duplicate