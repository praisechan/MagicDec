############PG19#############
# dynamic budget
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 32800 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 32800 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 65568 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 5 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 65568 --dataset pg19

# gamma 4
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 32800 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 32800 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 65568 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 65568 --dataset pg19

# gamma 3
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 32800 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 32800 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.5 --prefix_len 65568 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 65568 --dataset pg19

# gamma 4
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.05 --budget2 0.35 --enable_dynamic_budget --budget2_low 0.15 --confidence_threshold 0.5 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.05 --budget2 0.35 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.05 --budget2 0.35 --enable_dynamic_budget --budget2_low 0.15 --confidence_threshold 0.5 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.05 --budget2 0.35 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.05 --budget2 0.35 --enable_dynamic_budget --budget2_low 0.15 --confidence_threshold 0.5 --prefix_len 32800 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.05 --budget2 0.35 --prefix_len 32800 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.05 --budget2 0.35 --enable_dynamic_budget --budget2_low 0.15 --confidence_threshold 0.5 --prefix_len 65568 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 4 --gamma2 32 --budget1 0.05 --budget2 0.35 --prefix_len 65568 --dataset pg19

# gamma 3
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.3 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 8224 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.3 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 16416 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.3 --prefix_len 32800 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 32800 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --enable_dynamic_budget --budget2_low 0.1 --confidence_threshold 0.3 --prefix_len 65568 --dataset pg19
python tests/dynamic_verify_test/run_3step.py --model_name meta-llama/Meta-Llama-3.1-8B --gamma1 3 --gamma2 32 --budget1 0.02 --budget2 0.25 --prefix_len 65568 --dataset pg19
