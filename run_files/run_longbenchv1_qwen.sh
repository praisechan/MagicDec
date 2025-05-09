python tests/SnapKV/selfspec_benchmark.py --model checkpoints/Qwen/Qwen2.5-14B/model.pth --model_name Qwen/Qwen2.5-14B --rank_group 0 --gamma 16 --B 1 --draft_budget 257 --compile --dataset longbenchv1 --task qmsum
python tests/SnapKV/selfspec_benchmark.py --model checkpoints/Qwen/Qwen2.5-14B/model.pth --model_name Qwen/Qwen2.5-14B --rank_group 0 --gamma 16 --B 1 --draft_budget 513 --compile --dataset longbenchv1 --task qmsum
python tests/SnapKV/selfspec_benchmark.py --model checkpoints/Qwen/Qwen2.5-14B/model.pth --model_name Qwen/Qwen2.5-14B --rank_group 0 --gamma 16 --B 1 --draft_budget 1025 --compile --dataset longbenchv1 --task qmsum
python tests/SnapKV/selfspec_benchmark.py --model checkpoints/Qwen/Qwen2.5-14B/model.pth --model_name Qwen/Qwen2.5-14B --rank_group 0 --gamma 16 --B 1 --draft_budget 2049 --compile --dataset longbenchv1 --task qmsum

python tests/SnapKV/selfspec_benchmark.py --model checkpoints/Qwen/Qwen2.5-14B/model.pth --model_name Qwen/Qwen2.5-14B --rank_group 0 --gamma 16 --B 1 --draft_budget 257 --compile --dataset longbenchv1 --task gov_report
python tests/SnapKV/selfspec_benchmark.py --model checkpoints/Qwen/Qwen2.5-14B/model.pth --model_name Qwen/Qwen2.5-14B --rank_group 0 --gamma 16 --B 1 --draft_budget 513 --compile --dataset longbenchv1 --task gov_report
python tests/SnapKV/selfspec_benchmark.py --model checkpoints/Qwen/Qwen2.5-14B/model.pth --model_name Qwen/Qwen2.5-14B --rank_group 0 --gamma 16 --B 1 --draft_budget 1025 --compile --dataset longbenchv1 --task gov_report
python tests/SnapKV/selfspec_benchmark.py --model checkpoints/Qwen/Qwen2.5-14B/model.pth --model_name Qwen/Qwen2.5-14B --rank_group 0 --gamma 16 --B 1 --draft_budget 2049 --compile --dataset longbenchv1 --task gov_report

# python tests/SnapKV/selfspec_benchmark.py --model checkpoints/Qwen/Qwen2.5-14B/model.pth --model_name Qwen/Qwen2.5-14B --rank_group 0 --gamma 16 --B 1 --draft_budget 129 --compile --dataset longbenchv1 --task multi_news
# python tests/SnapKV/selfspec_benchmark.py --model checkpoints/Qwen/Qwen2.5-14B/model.pth --model_name Qwen/Qwen2.5-14B --rank_group 0 --gamma 16 --B 1 --draft_budget 257 --compile --dataset longbenchv1 --task multi_news
# python tests/SnapKV/selfspec_benchmark.py --model checkpoints/Qwen/Qwen2.5-14B/model.pth --model_name Qwen/Qwen2.5-14B --rank_group 0 --gamma 16 --B 1 --draft_budget 513 --compile --dataset longbenchv1 --task multi_news
# python tests/SnapKV/selfspec_benchmark.py --model checkpoints/Qwen/Qwen2.5-14B/model.pth --model_name Qwen/Qwen2.5-14B --rank_group 0 --gamma 16 --B 1 --draft_budget 1025 --compile --dataset longbenchv1 --task multi_news
