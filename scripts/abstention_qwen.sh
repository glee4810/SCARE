# single_turn_two_stage
export db_id='mimic_iv'
export model='Qwen/Qwen3-32B'
python src/single_turn_two_stage.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='eicu'
export model='Qwen/Qwen3-32B'
python src/single_turn_two_stage.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='mimicsql'
export model='Qwen/Qwen3-32B'
python src/single_turn_two_stage.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"


# single_turn_correction
export db_id='mimic_iv'
export model='Qwen/Qwen3-32B'
python src/single_turn_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='eicu'
export model='Qwen/Qwen3-32B'
python src/single_turn_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='mimicsql'
export model='Qwen/Qwen3-32B'
python src/single_turn_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"


# verifier_correction
export db_id='mimic_iv'
export model='Qwen/Qwen3-32B'
python src/verifier_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='eicu'
export model='Qwen/Qwen3-32B'
python src/verifier_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='mimicsql'
export model='Qwen/Qwen3-32B'
python src/verifier_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"


# multi_turn_correction
export db_id='mimic_iv'
export model='Qwen/Qwen3-32B'
python src/multi_turn_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='eicu'
export model='Qwen/Qwen3-32B'
python src/multi_turn_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='mimicsql'
export model='Qwen/Qwen3-32B'
python src/multi_turn_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"


# single_turn_correction_cls
export db_id='mimic_iv'
export model='Qwen/Qwen3-32B'
python src/single_turn_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='eicu'
export model='Qwen/Qwen3-32B'
python src/single_turn_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='mimicsql'
export model='Qwen/Qwen3-32B'
python src/single_turn_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"


# verifier_correction_cls
export db_id='mimic_iv'
export model='Qwen/Qwen3-32B'
python src/verifier_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='eicu'
export model='Qwen/Qwen3-32B'
python src/verifier_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='mimicsql'
export model='Qwen/Qwen3-32B'
python src/verifier_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"


# multi_turn_correction_cls
export db_id='mimic_iv'
export model='Qwen/Qwen3-32B'
python src/multi_turn_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='eicu'
export model='Qwen/Qwen3-32B'
python src/multi_turn_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"

export db_id='mimicsql'
export model='Qwen/Qwen3-32B'
python src/multi_turn_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 50 \
    --base_url "http://localhost:8001/v1"


