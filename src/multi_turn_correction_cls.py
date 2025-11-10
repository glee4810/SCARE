import argparse
import json
import os
import multiprocessing as mp
import litellm
import warnings
from typing import Dict, List
from pydantic import BaseModel

warnings.filterwarnings("ignore")

from utils.utils import (
    SQLEvaluator,
    llm_call_with_retry,
    llm_call_with_retry_messages,
    save_results_safely,
    process_instances_single_core,
    process_instances_multi_core,
    setup_subset_data,
    create_metadata,
    open_file
)
import dotenv
dotenv.load_dotenv()

class FeedbackResponse(BaseModel):
    """Model for feedback responses."""
    feedback: str

# Pydantic models for structured output
class CorrectionResponse(BaseModel):
    """Model for correction responses."""
    reasoning: str
    answer: str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_id', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--tables_path', type=str, default='./databases/tables.json', help='Path to the tables file')
    parser.add_argument('--db_path', type=str, default='./databases', help='Path to database files directory')
    parser.add_argument('--base_url', type=str, default=None)
    parser.add_argument('--num_process', type=int, default=1, help='Number of processes to use. 1 for single core, >1 for multicore')
    parser.add_argument('--save_interval', type=int, default=1, help='Save results every N iterations')
    parser.add_argument('--subset', type=bool, default=False, help='Whether to subset the data')
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum number of iterations')
    return parser.parse_args()


def multi_turn_correction_cls(model: str, metadata: Dict, db_path: str,
                          few_shots: List[Dict] = None,
                          base_url: str = None,
                          max_iter: int = 3):

    # Extract data from metadata
    db_id = metadata['db_id']
    question = metadata['question']
    evidence = metadata['evidence']
    schema = metadata['schema']
    pred_sql = metadata['init_pred_sql']
    pred_answer = metadata['init_pred_sql_exec_result']
    
    evaluator = SQLEvaluator(data_dir=db_path, dataset=db_id)

    trace = []

    # Question Classifier
    system_prompt = open_file('./prompt/multi_turn_correction_cls/classification_system_prompt.txt')
    user_template = open_file('./prompt/multi_turn_correction_cls/classification_user_prompt.txt')
    user_prompt = user_template.format(
        database_schema=schema,
        question=question,
        evidence=evidence,
        sql=pred_sql,
        exec=pred_answer
    )

    try:
        output, usage_info = llm_call_with_retry(
            model=model, system_prompt=system_prompt, user_prompt=user_prompt, 
            response_format=CorrectionResponse, few_shots=few_shots, base_url=base_url
        )
    except Exception as e:
        output = {
            "reasoning": f"Error in question classifier: {e}",
            "answer": "error in correction"
        }
        usage_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cost': 0.0, 'llm_calls': 0}
    
    question_classifier_decision = f'{output["reasoning"]} So, the question is {output["answer"]}.'
    trace.append({**output, 'usage_info': usage_info})

    messages = []
    system_prompt = open_file('./prompt/multi_turn_correction_cls/system_prompt.txt')
    messages.append({"role": "system", "content": system_prompt})

    user_template = open_file('./prompt/multi_turn_correction_cls/user_prompt.txt')
    user_prompt = user_template.format(
        database_schema=schema,
        question=question,
        evidence=evidence,
        sql=pred_sql,
        exec=pred_answer,
        question_classifier_decision=question_classifier_decision
    )
    messages.append({"role": "user", "content": user_prompt})

    for itr in range(max_iter):

        if itr > 0:
            user_template = open_file('./prompt/multi_turn_correction_cls/next_user_prompt.txt')
            user_prompt = user_template.format(
                sql=pred_sql,
                exec=pred_answer
            )
            messages.append({"role": "user", "content": user_prompt})

        try:
            output, usage_info = llm_call_with_retry_messages(
                model=model, messages=messages, 
                response_format=CorrectionResponse, base_url=base_url
            )
        except Exception as e:
            output = {
                "reasoning": f"Error in multi-turn correction: {e}",
                "answer": "error in correction"
            }
            usage_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cost': 0.0, 'llm_calls': 0}

        trace.append({**output, 'usage_info': usage_info})

        # Execute the corrected SQL
        final_sql = output['answer']
        try:
            final_sql_exec_result = evaluator.execute(db_id=db_id, sql=final_sql, is_gold_sql=False)
        except:
            final_sql_exec_result = final_sql

        pred_sql = final_sql
        pred_answer = final_sql_exec_result
        
        messages.append({"role": "assistant", "content": f'Reasoning: {output["reasoning"]}\n\nCorrected SQL: {output["answer"]}\n\nExecution Result: {pred_answer}'})

        # Feedback
        feedback_system_prompt = open_file('./prompt/multi_turn_correction_cls/feedback_system_prompt.txt')
        feedback_user_template = open_file('./prompt/multi_turn_correction_cls/feedback_user_prompt.txt')
        feedback_user_prompt = feedback_user_template.format(
            database_schema=schema,
            question=question,
            evidence=evidence,
            sql=pred_sql,
            exec=pred_answer
        )

        try:
            output, usage_info = llm_call_with_retry(
                model=model, system_prompt=feedback_system_prompt, user_prompt=feedback_user_prompt, 
                response_format=FeedbackResponse, few_shots=few_shots, base_url=base_url
            )
        except Exception as e:
            output = {
                "feedback": f"Error in multi-turn correction: {e}",
            }
            usage_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cost': 0.0, 'llm_calls': 0}

        trace.append({**output, 'usage_info': usage_info})
        messages.append({"role": "assistant", "content": f'self-feedback: {output["feedback"]}'})

        if 'the predicted sql is correct' in output['feedback'].lower():
            break

    # Update metadata with results
    result = metadata.copy()
    result['final_sql'] = final_sql
    result['final_sql_exec_result'] = final_sql_exec_result
    result['trace'] = trace

    return result


def create_process_instance_func(args):

    def process_instance(instance):
        metadata = create_metadata(instance, args)
        instance_id = metadata['id']

        # Run multi-turn correction
        result = multi_turn_correction_cls(
            model=args.model_name,
            metadata=metadata, 
            db_path=args.db_path,
            few_shots=None,
            base_url=args.base_url,
            max_iter=args.max_iter
        )

        return instance_id, result
    
    return process_instance


def main():
    args = parse_args()
    
    # Load data
    with open(f'./correction-data/{args.db_id}_test_set_correction_data.json', 'r') as f:
        sql_results = json.load(f)

    # Create output directory if needed
    model_name = args.model_name.replace('meta-llama/', '').replace('Qwen/', '').replace('gemini/', '')
    if not args.subset:
        args.data_output_path = f"output/multi_turn_correction_cls_{args.db_id}_{model_name}.json"
    else:
        args.data_output_path = f"output/multi_turn_correction_cls_{args.db_id}_{model_name}_subset.json"

    if not os.path.exists(os.path.dirname(args.data_output_path)):
        os.makedirs(os.path.dirname(args.data_output_path), exist_ok=True)

    # Setup subset data if needed
    sql_results = setup_subset_data(sql_results, args)

    # Load existing results
    result_data = []
    if os.path.exists(args.data_output_path):
        with open(args.data_output_path, 'r') as f:
            result_data = json.load(f)

    # Filter out already processed instances to get actual instances to process
    processed_ids = {item['id'] for item in result_data if 'id' in item}
    instances_to_process = [
        instance for instance in sql_results 
        if f"{instance['generator_name']}_{instance['id']}" not in processed_ids
    ]

    num_process = min(min(args.num_process, mp.cpu_count()), len(instances_to_process))
    print(f"Total instances in sql_results: {len(sql_results)}")
    print(f"Already processed: {len(result_data)}")
    print(f"Instances to process: {len(instances_to_process)}")
    print(f"Processing mode: {'Single-core' if num_process == 1 else f'Multi-core ({num_process} processes)'}")
    print(f"Save interval: every {args.save_interval} instances")
    
    # Create process function
    process_func = create_process_instance_func(args)
        
    # Choose processing method based on num_process
    if num_process == 1:
        result_data = process_instances_single_core(
            instances_to_process, args, process_func, result_data,
            desc="Processing instances (multi-turn single-core)"
        )
    else:
        result_data = process_instances_multi_core(
            instances_to_process, args, process_func, result_data,
            desc="Processing instances (multi-turn multi-core)"
        )
    
    # Final save
    save_results_safely(result_data, args.data_output_path)
    print(f"Processing complete. Final results saved to {args.data_output_path}")

if __name__ == "__main__":
    main()
