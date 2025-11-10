# single_turn_two_stage
export db_id='mimic_iv'
export model='gpt-5-mini'
python src/single_turn_two_stage.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='eicu'
export model='gpt-5-mini'
python src/single_turn_two_stage.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='mimicsql'
export model='gpt-5-mini'
python src/single_turn_two_stage.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30


# single_turn_correction
export db_id='mimic_iv'
export model='gpt-5-mini'
python src/single_turn_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='eicu'
export model='gpt-5-mini'
python src/single_turn_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='mimicsql'
export model='gpt-5-mini'
python src/single_turn_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30


# verifier_correction
export db_id='mimic_iv'
export model='gpt-5-mini'
python src/verifier_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='eicu'
export model='gpt-5-mini'
python src/verifier_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='mimicsql'
export model='gpt-5-mini'
python src/verifier_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30


# multi_turn_correction
export db_id='mimic_iv'
export model='gpt-5-mini'
python src/multi_turn_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='eicu'
export model='gpt-5-mini'
python src/multi_turn_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='mimicsql'
export model='gpt-5-mini'
python src/multi_turn_correction.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30


# single_turn_correction_cls
export db_id='mimic_iv'
export model='gpt-5-mini'
python src/single_turn_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='eicu'
export model='gpt-5-mini'
python src/single_turn_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='mimicsql'
export model='gpt-5-mini'
python src/single_turn_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30


# verifier_correction_cls
export db_id='mimic_iv'
export model='gpt-5-mini'
python src/verifier_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='eicu'
export model='gpt-5-mini'
python src/verifier_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='mimicsql'
export model='gpt-5-mini'
python src/verifier_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30


# multi_turn_correction_cls
export db_id='mimic_iv'
export model='gpt-5-mini'
python src/multi_turn_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='eicu'
export model='gpt-5-mini'
python src/multi_turn_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

export db_id='mimicsql'
export model='gpt-5-mini'
python src/multi_turn_correction_cls.py \
    --db_id ${db_id} \
    --model_name "${model}" \
    --num_process 30

