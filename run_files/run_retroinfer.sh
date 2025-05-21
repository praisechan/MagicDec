# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 8224 --max_len 8320 --attn_type RetroInfer --budget_ratio 257 
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 8224 --max_len 8320 --attn_type RetroInfer --budget_ratio 513 
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 8224 --max_len 8320 --attn_type RetroInfer --budget_ratio 1025 
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 8224 --max_len 8320 --attn_type RetroInfer --budget_ratio 2049 

# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 16416 --max_len 16512 --attn_type RetroInfer --budget_ratio 513 
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 16416 --max_len 16512 --attn_type RetroInfer --budget_ratio 1025 
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 16416 --max_len 16512 --attn_type RetroInfer --budget_ratio 2049 
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 16416 --max_len 16512 --attn_type RetroInfer --budget_ratio 4097 

# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 32800 --max_len 32896 --attn_type RetroInfer --budget_ratio 1025 
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 32800 --max_len 32896 --attn_type RetroInfer --budget_ratio 2049 
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 32800 --max_len 32896 --attn_type RetroInfer --budget_ratio 4097 
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 32800 --max_len 32896 --attn_type RetroInfer --budget_ratio 8193 

# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 65568 --max_len 65664 --attn_type RetroInfer --budget_ratio 2049 
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 65568 --max_len 65664 --attn_type RetroInfer --budget_ratio 4097 
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 65568 --max_len 65664 --attn_type RetroInfer --budget_ratio 8193 
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --prefix_len 65568 --max_len 65664 --attn_type RetroInfer --budget_ratio 16385 

python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 64 --B 1 --attn_type RetroInfer --budget_ratio 0.25 --dataset longbenchv1 --task qmsum
python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 64 --B 1 --attn_type RetroInfer --budget_ratio 0.12 --dataset longbenchv1 --task qmsum
python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 64 --B 1 --attn_type RetroInfer --budget_ratio 0.06 --dataset longbenchv1 --task qmsum
python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 64 --B 1 --attn_type RetroInfer --budget_ratio 0.03 --dataset longbenchv1 --task qmsum
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 64 --B 1 --attn_type RetroInfer --budget_ratio 0.25 --dataset longbenchv1 --task gov_report
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 64 --B 1 --attn_type RetroInfer --budget_ratio 0.12 --dataset longbenchv1 --task gov_report
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 64 --B 1 --attn_type RetroInfer --budget_ratio 0.06 --dataset longbenchv1 --task gov_report
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 64 --B 1 --attn_type RetroInfer --budget_ratio 0.03 --dataset longbenchv1 --task gov_report

python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --attn_type RetroInfer --budget_ratio 0.25 --dataset longbenchv1 --task qmsum
python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --attn_type RetroInfer --budget_ratio 0.12 --dataset longbenchv1 --task qmsum
python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --attn_type RetroInfer --budget_ratio 0.06 --dataset longbenchv1 --task qmsum
python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --attn_type RetroInfer --budget_ratio 0.03 --dataset longbenchv1 --task qmsum
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --attn_type RetroInfer --budget_ratio 0.25 --dataset longbenchv1 --task gov_report
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --attn_type RetroInfer --budget_ratio 0.12 --dataset longbenchv1 --task gov_report
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --attn_type RetroInfer --budget_ratio 0.06 --dataset longbenchv1 --task gov_report
# python tests/RetrievalAttention/selfspec_benchmark.py --model_name llama-3.1-8b  --gamma 16 --B 1 --attn_type RetroInfer --budget_ratio 0.03 --dataset longbenchv1 --task gov_report