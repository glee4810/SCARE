dataset='mimic_iv,eicu,mimicsql'
python src/evaluate_sql_correction.py --model_name "gemini-2.5-flash" --dataset_name_list ${dataset} --corrector_name single_turn_two_stage
python src/evaluate_sql_correction.py --model_name "gemini-2.5-flash" --dataset_name_list ${dataset} --corrector_name single_turn_correction
python src/evaluate_sql_correction.py --model_name "gemini-2.5-flash" --dataset_name_list ${dataset} --corrector_name verifier_correction
python src/evaluate_sql_correction.py --model_name "gemini-2.5-flash" --dataset_name_list ${dataset} --corrector_name multi_turn_correction
python src/evaluate_sql_correction.py --model_name "gemini-2.5-flash" --dataset_name_list ${dataset} --corrector_name single_turn_correction_cls
python src/evaluate_sql_correction.py --model_name "gemini-2.5-flash" --dataset_name_list ${dataset} --corrector_name verifier_correction_cls
python src/evaluate_sql_correction.py --model_name "gemini-2.5-flash" --dataset_name_list ${dataset} --corrector_name multi_turn_correction_cls


# dataset='mimic_iv,eicu,mimicsql'
# python src/evaluate_sql_correction.py --model_name "gemini-2.0-flash" --dataset_name_list ${dataset} --corrector_name single_turn_two_stage
# python src/evaluate_sql_correction.py --model_name "gemini-2.0-flash" --dataset_name_list ${dataset} --corrector_name single_turn_correction
# python src/evaluate_sql_correction.py --model_name "gemini-2.0-flash" --dataset_name_list ${dataset} --corrector_name verifier_correction
# python src/evaluate_sql_correction.py --model_name "gemini-2.0-flash" --dataset_name_list ${dataset} --corrector_name multi_turn_correction
# python src/evaluate_sql_correction.py --model_name "gemini-2.0-flash" --dataset_name_list ${dataset} --corrector_name single_turn_correction_cls
# python src/evaluate_sql_correction.py --model_name "gemini-2.0-flash" --dataset_name_list ${dataset} --corrector_name verifier_correction_cls
# python src/evaluate_sql_correction.py --model_name "gemini-2.0-flash" --dataset_name_list ${dataset} --corrector_name multi_turn_correction_cls


# dataset='mimic_iv,eicu,mimicsql'
# python src/evaluate_sql_correction.py --model_name "gpt-5-mini" --dataset_name_list ${dataset} --corrector_name single_turn_two_stage
# python src/evaluate_sql_correction.py --model_name "gpt-5-mini" --dataset_name_list ${dataset} --corrector_name single_turn_correction
# python src/evaluate_sql_correction.py --model_name "gpt-5-mini" --dataset_name_list ${dataset} --corrector_name verifier_correction
# python src/evaluate_sql_correction.py --model_name "gpt-5-mini" --dataset_name_list ${dataset} --corrector_name multi_turn_correction
# python src/evaluate_sql_correction.py --model_name "gpt-5-mini" --dataset_name_list ${dataset} --corrector_name single_turn_correction_cls
# python src/evaluate_sql_correction.py --model_name "gpt-5-mini" --dataset_name_list ${dataset} --corrector_name verifier_correction_cls
# python src/evaluate_sql_correction.py --model_name "gpt-5-mini" --dataset_name_list ${dataset} --corrector_name multi_turn_correction_cls


# dataset='mimic_iv,eicu,mimicsql'
# python src/evaluate_sql_correction.py --model_name "Llama-3.3-70B-Instruct" --dataset_name_list ${dataset} --corrector_name single_turn_two_stage
# python src/evaluate_sql_correction.py --model_name "Llama-3.3-70B-Instruct" --dataset_name_list ${dataset} --corrector_name single_turn_correction
# python src/evaluate_sql_correction.py --model_name "Llama-3.3-70B-Instruct" --dataset_name_list ${dataset} --corrector_name verifier_correction
# python src/evaluate_sql_correction.py --model_name "Llama-3.3-70B-Instruct" --dataset_name_list ${dataset} --corrector_name multi_turn_correction
# python src/evaluate_sql_correction.py --model_name "Llama-3.3-70B-Instruct" --dataset_name_list ${dataset} --corrector_name single_turn_correction_cls
# python src/evaluate_sql_correction.py --model_name "Llama-3.3-70B-Instruct" --dataset_name_list ${dataset} --corrector_name verifier_correction_cls
# python src/evaluate_sql_correction.py --model_name "Llama-3.3-70B-Instruct" --dataset_name_list ${dataset} --corrector_name multi_turn_correction_cls


# dataset='mimic_iv,eicu,mimicsql'
# python src/evaluate_sql_correction.py --model_name "Qwen3-32B" --dataset_name_list ${dataset} --corrector_name single_turn_two_stage
# python src/evaluate_sql_correction.py --model_name "Qwen3-32B" --dataset_name_list ${dataset} --corrector_name single_turn_correction
# python src/evaluate_sql_correction.py --model_name "Qwen3-32B" --dataset_name_list ${dataset} --corrector_name verifier_correction
# python src/evaluate_sql_correction.py --model_name "Qwen3-32B" --dataset_name_list ${dataset} --corrector_name multi_turn_correction
# python src/evaluate_sql_correction.py --model_name "Qwen3-32B" --dataset_name_list ${dataset} --corrector_name single_turn_correction_cls
# python src/evaluate_sql_correction.py --model_name "Qwen3-32B" --dataset_name_list ${dataset} --corrector_name verifier_correction_cls
# python src/evaluate_sql_correction.py --model_name "Qwen3-32B" --dataset_name_list ${dataset} --corrector_name multi_turn_correction_cls

