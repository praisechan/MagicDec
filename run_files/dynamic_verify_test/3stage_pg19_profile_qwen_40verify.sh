############PG19#############
# dynamic budget
export CLUSTER_SIZE=32
# python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --prefix_len 32800 --dataset pg19

python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --prefix_len 32800 --dataset pg19

# export CLUSTER_SIZE=32
# # python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 8224 --dataset pg19
# python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --prefix_len 8224 --dataset pg19
# # python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 16416 --dataset pg19
# python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --prefix_len 16416 --dataset pg19
# # python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 32800 --dataset pg19
# python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --prefix_len 32800 --dataset pg19
# # python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 65568 --dataset pg19
# python tests/dynamic_verify_test/run_3step_profile.py --model_name qwen2.5-14b --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.4 --prefix_len 65568 --dataset pg19