############PG19#############
# dynamic budget
python tests/dynamic_verify_test/run_3step_kl_confidence.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step_kl_confidence.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step_kl_confidence.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 32800 --dataset pg19
python tests/dynamic_verify_test/run_3step_kl_confidence.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 65568 --dataset pg19

python tests/dynamic_verify_test/run_3step_kl_confidence.py --model_name meta-llama/Llama-3.2-3B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step_kl_confidence.py --model_name meta-llama/Llama-3.2-3B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step_kl_confidence.py --model_name meta-llama/Llama-3.2-3B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 32800 --dataset pg19
python tests/dynamic_verify_test/run_3step_kl_confidence.py --model_name meta-llama/Llama-3.2-3B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 65568 --dataset pg19