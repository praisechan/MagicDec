# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 8224 --max_len 8320 --draft_budget 257 
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 8224 --max_len 8320 --draft_budget 513 
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 8224 --max_len 8320 --draft_budget 1025 
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 8224 --max_len 8320 --draft_budget 2049 

# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 16416 --max_len 16512 --draft_budget 513 
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 16416 --max_len 16512 --draft_budget 1025 
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 16416 --max_len 16512 --draft_budget 2049 
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 16416 --max_len 16512 --draft_budget 4097 

# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 32800 --max_len 32896 --draft_budget 1025 
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 32800 --max_len 32896 --draft_budget 2049 
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 32800 --max_len 32896 --draft_budget 4097 
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 32800 --max_len 32896 --draft_budget 8193 

# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 65568 --max_len 65664 --draft_budget 2049 
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 65568 --max_len 65664 --draft_budget 4097 
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 65568 --max_len 65664 --draft_budget 8193 
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --prefix_len 65568 --max_len 65664 --draft_budget 16385 

python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --draft_budget 257 --chunk_size 16  --dataset longbenchv1 --task qmsum
python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --draft_budget 513 --chunk_size 16  --dataset longbenchv1 --task qmsum
python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --draft_budget 1025 --chunk_size 16  --dataset longbenchv1 --task qmsum
python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --draft_budget 2049 --chunk_size 16  --dataset longbenchv1 --task qmsum
python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --draft_budget 257 --chunk_size 16  --dataset longbenchv1 --task gov_report
python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --draft_budget 513 --chunk_size 16  --dataset longbenchv1 --task gov_report
python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --draft_budget 1025 --chunk_size 16  --dataset longbenchv1 --task gov_report
python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --draft_budget 2049 --chunk_size 16  --dataset longbenchv1 --task gov_report
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --draft_budget 129 --chunk_size 16  --dataset longbenchv1 --task multi_news
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --draft_budget 257 --chunk_size 16  --dataset longbenchv1 --task multi_news
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --draft_budget 513 --chunk_size 16  --dataset longbenchv1 --task multi_news
# python tests/Quest/selfspec_benchmark.py --model_name lmsys/longchat-7b-v1.5-32k --latest_k 128 --rank_group 0 --gamma 16 --B 1 --draft_budget 1025 --chunk_size 16  --dataset longbenchv1 --task multi_news