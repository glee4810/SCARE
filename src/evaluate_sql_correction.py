import os
import json
import argparse
import pandas as pd
from utils.utils import SQLEvaluator
from tqdm import tqdm

def parse_comma_separated_list(value):
    """Parse a comma-separated string into a list of strings."""
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.split(',')]

parser = argparse.ArgumentParser(description="Evaluate SQL correction results for answerable, ambiguous, and unanswerable queries.")
parser.add_argument("--model_name", type=str, required=True, help="Name of the model used for SQL generation.")
parser.add_argument("--dataset_name_list", type=parse_comma_separated_list, required=True, help="Comma-separated list of dataset names.")
parser.add_argument("--corrector_name", type=str, required=True, help="Name of the SQL corrector.")
parser.add_argument("--correction_output_dir", type=str, default="./output", help="Directory for correction output JSON files.")
parser.add_argument("--granular_evaluation", action="store_true", help="Enable granular evaluation for sub-types.")
parser.add_argument("--data_dir", type=str, default="./databases", help="Directory for database files.")
args = parser.parse_args()

_evaluator = {}

# Main metrics
answerable_correct_execution_accuracy = {'total': 0, 'preserved': 0}
answerable_incorrect_execution_accuracy = {'total': 0, 'corrected': 0}
ambiguous_precision = {'total': 0, 'correct': 0}
ambiguous_recall = {'total': 0, 'correct': 0}
unanswerable_precision = {'total': 0, 'correct': 0}
unanswerable_recall = {'total': 0, 'correct': 0}
# Granular metrics for detailed analysis
answerable_correct_execution_accuracy_granular = {
    'easy': {'total': 0, 'preserved': 0},
    'medium': {'total': 0, 'preserved': 0},
    'hard': {'total': 0, 'preserved': 0}
}

answerable_incorrect_execution_accuracy_granular = {
    'easy': {'total': 0, 'corrected': 0},
    'medium': {'total': 0, 'corrected': 0},
    'hard': {'total': 0, 'corrected': 0}
}

precision_granular = {
    'vague-question': {'total': 0, 'correct': 0},
    'vague-word': {'total': 0, 'correct': 0},
    'ambiguous-reference': {'total': 0, 'correct': 0},
    'infeasible-faq': {'total': 0, 'correct': 0},
    'missing-column': {'total': 0, 'correct': 0},
    'small-talk': {'total': 0, 'correct': 0},
    'out-of-scope': {'total': 0, 'correct': 0}
}

recall_granular = {
    'vague-question': {'total': 0, 'correct': 0},
    'vague-word': {'total': 0, 'correct': 0},
    'ambiguous-reference': {'total': 0, 'correct': 0},
    'infeasible-faq': {'total': 0, 'correct': 0},
    'missing-column': {'total': 0, 'correct': 0},
    'small-talk': {'total': 0, 'correct': 0},
    'out-of-scope': {'total': 0, 'correct': 0}
}

coverage = {
    "answerable-correct": {'total': 0, 'attempted': 0},
    "answerable-incorrect": {'total': 0, 'attempted': 0}
}


# Process each dataset
corrections_all = []
for dataset in args.dataset_name_list:

    correction_path = os.path.join(args.correction_output_dir, f"{args.corrector_name}_{dataset}_{args.model_name}.json")    
    eval_correction_path = os.path.join(args.correction_output_dir+'_eval', f"{args.corrector_name}_{dataset}_{args.model_name}.json")

    def safe_json_load(file_path):
        """Safely load JSON file with error handling"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error in {file_path}: {e}")
            print("Attempting to fix the JSON file...")
            
            # Try to read and fix the JSON file
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Common JSON fixes
            # Remove trailing commas before closing brackets/braces
            import re
            content = re.sub(r',(\s*[}\]])', r'\1', content)
            
            try:
                return json.loads(content)
            except json.JSONDecodeError as e2:
                print(f"Error: Could not fix JSON file {file_path}: {e2}")
                print("Skipping this dataset...")
                return None
        except Exception as e:
            print(f"Error: Could not read file {file_path}: {e}")
            return None

    if os.path.exists(eval_correction_path):
        corrections = safe_json_load(eval_correction_path)
    else:
        corrections = safe_json_load(correction_path)
    
    if corrections is None:
        print(f"Skipping dataset {dataset} due to JSON loading error")
        continue

    if dataset not in _evaluator:
        _evaluator[dataset] = SQLEvaluator(data_dir=args.data_dir, dataset=dataset)
    


    for instance in tqdm(corrections):
        # Extract instance details
        question_id = instance['id']
        sample_type = instance['sample_type']
        init_pred_sql = instance['init_pred_sql']
        init_pred_sql_exec_result = instance['init_pred_sql_exec_result']
        init_exec_acc = instance['init_exec_acc']
        gold_sql = instance['gold_sql']
        gold_answer = instance['gold_answer']
        final_sql = instance['final_sql']
        final_sql_exec_result = instance['final_sql_exec_result']
        final_exec_acc = instance.get('final_exec_acc', None)

        if gold_answer == 'null':
            gold_answer = sample_type
                
        # Count precision based on predicted type
        if final_sql in ['ambiguous', 'vague-question', 'vague-word', 'ambiguous-reference']:
            ambiguous_precision['total'] += 1
            if args.granular_evaluation:
                for amb_type in ['vague-question', 'vague-word', 'ambiguous-reference']:
                    if sample_type == amb_type:
                        precision_granular[amb_type]['total'] += 1
                        break
        elif final_sql in ['unanswerable', 'infeasible-faq', 'missing-column', 'small-talk', 'out-of-scope']:
            unanswerable_precision['total'] += 1
            if args.granular_evaluation:
                for unans_type in ['infeasible-faq', 'missing-column', 'small-talk', 'out-of-scope']:
                    if sample_type == unans_type:
                        precision_granular[unans_type]['total'] += 1
                        break

        # Evaluate based on true sample type
        if sample_type in ['easy', 'medium', 'hard']:
            # Answerable queries
            if final_exec_acc is None:
                final_exec_acc = _evaluator[dataset](db_id=dataset, pred_sql=final_sql, gold_sql=gold_sql, gold_answer=gold_answer)["is_correct"]
            
            if init_exec_acc:
                
                # Initially correct
                coverage["answerable-correct"]['total'] += 1
                if final_sql not in ['ambiguous', 'unanswerable', 'vague-question', 'vague-word', 'ambiguous-reference', 'infeasible-faq', 'missing-column', 'small-talk', 'out-of-scope']:
                    coverage["answerable-correct"]['attempted'] += 1
                answerable_correct_execution_accuracy['total'] += 1
                if args.granular_evaluation:
                    answerable_correct_execution_accuracy_granular[sample_type]['total'] += 1
                if final_exec_acc:
                    answerable_correct_execution_accuracy['preserved'] += 1
                    if args.granular_evaluation:
                        answerable_correct_execution_accuracy_granular[sample_type]['preserved'] += 1

            else:
                # Initially incorrect
                coverage["answerable-incorrect"]['total'] += 1
                if final_sql not in ['ambiguous', 'unanswerable', 'vague-question', 'vague-word', 'ambiguous-reference', 'infeasible-faq', 'missing-column', 'small-talk', 'out-of-scope']:
                    coverage["answerable-incorrect"]['attempted'] += 1
                answerable_incorrect_execution_accuracy['total'] += 1
                if args.granular_evaluation:
                    answerable_incorrect_execution_accuracy_granular[sample_type]['total'] += 1
                if final_exec_acc:
                    answerable_incorrect_execution_accuracy['corrected'] += 1
                    if args.granular_evaluation:
                        answerable_incorrect_execution_accuracy_granular[sample_type]['corrected'] += 1

        elif sample_type in ['vague-question', 'vague-word', 'ambiguous-reference', 'infeasible-faq', 'missing-column', 'small-talk', 'out-of-scope']:

            if final_exec_acc is None:
                final_exec_acc = (sample_type in ['vague-question', 'vague-word', 'ambiguous-reference'] and final_sql_exec_result == 'ambiguous') or (sample_type in ['infeasible-faq', 'missing-column', 'small-talk', 'out-of-scope'] and final_sql_exec_result == 'unanswerable')

            if sample_type in ['vague-question', 'vague-word', 'ambiguous-reference']:

                # Ambiguous queries
                ambiguous_recall['total'] += 1
                if args.granular_evaluation:
                    recall_granular[sample_type]['total'] += 1
                    
                if final_sql in ['ambiguous', 'vague-question', 'vague-word', 'ambiguous-reference']:
                    ambiguous_recall['correct'] += 1
                    if args.granular_evaluation:
                        recall_granular[sample_type]['correct'] += 1
                    ambiguous_precision['correct'] += 1
                    if args.granular_evaluation:
                        precision_granular[sample_type]['correct'] += 1
                        
            elif sample_type in ['infeasible-faq', 'missing-column', 'small-talk', 'out-of-scope']:
                # Unanswerable queries
                unanswerable_recall['total'] += 1
                if args.granular_evaluation:
                    recall_granular[sample_type]['total'] += 1
                    
                if final_sql in ['unanswerable', 'infeasible-faq', 'missing-column', 'small-talk', 'out-of-scope']:
                    unanswerable_recall['correct'] += 1
                    if args.granular_evaluation:
                        recall_granular[sample_type]['correct'] += 1
                    unanswerable_precision['correct'] += 1
                    if args.granular_evaluation:
                        precision_granular[sample_type]['correct'] += 1

        else:
            raise ValueError(f"Unknown sample_type: {sample_type}")

        instance['gold_answer'] = gold_answer
        instance['final_exec_acc'] = final_exec_acc

        corrections_all.append(instance)

    os.makedirs(args.correction_output_dir+'_eval', exist_ok=True)
    with open(eval_correction_path, 'w') as f:
        json.dump(corrections, f, indent=4)

if len(args.dataset_name_list) == 3:
    eval_correction_path = os.path.join(args.correction_output_dir+'_eval', f"{args.corrector_name}_all_{args.model_name}.json")
    with open(eval_correction_path, 'w') as f:
        json.dump(corrections_all, f, indent=4)


# Helper function to format percentage
def format_metric(num, den, no_frac=False):
    if den > 0:
        if no_frac:
            return f"{round(num / den * 100, 1)}"
        else:
            return f"{round(num / den * 100, 1)} ({num}/{den})"
    return "NaN"

# Helper function to calculate F1 score
def calculate_f1(precision_correct, precision_total, recall_correct, recall_total):
    if precision_total > 0 and recall_total > 0:
        precision = precision_correct / precision_total
        recall = recall_correct / recall_total
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            return f"{round(f1 * 100, 1)}"
        return "0.00"
    return "NaN"

def save_results_to_excel(model_name, corrector_name, results_data, excel_file_path="evaluation_results.xlsx"):
    """Save evaluation results to Excel file with model name"""
    
    # Map corrector names to display names
    corrector_mapping = {
        'single_turn_two_stage': 'Two-Stage',
        'single_turn_correction': 'Single-Turn',
        'verifier_correction': 'Single-Turn-Veri',
        'multi_turn_correction': 'Multi-Turn-SelfRef',
        'single_turn_correction_cls': 'Single-Turn-Cls',
        'verifier_correction_cls': 'Single-Turn-Veri-Cls',
        'multi_turn_correction_cls': 'Multi-Turn-SelfRef-Cls'
    }
    
    display_corrector_name = corrector_mapping.get(corrector_name, corrector_name)
    
    # Check if file exists to load existing data
    try:
        existing_df = pd.read_excel(excel_file_path, sheet_name=None)
    except FileNotFoundError:
        existing_df = {}
    
    # Save each table to a separate sheet
    for sheet_name, data in results_data.items():
        # Add model and corrector information to the data
        for row in data:
            row['Model'] = model_name
            row['Corrector'] = display_corrector_name
        
        new_df = pd.DataFrame(data)
        
        if sheet_name in existing_df:
            # Append to existing sheet
            combined_df = pd.concat([existing_df[sheet_name], new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        existing_df[sheet_name] = combined_df
    
    # Save to Excel file
    with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='w') as writer:
        for sheet_name, df in existing_df.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Collect results data for Excel export
results_data = {}

# Table 5 - Main Results
table5_data = []

# Main results row
main_result = {
    'Metric': 'Main Results',
    'Answerable-Correct (Preserved %)': format_metric(answerable_correct_execution_accuracy['preserved'], answerable_correct_execution_accuracy['total'], no_frac=True),
    'Answerable-Correct (Coverage %)': format_metric(coverage["answerable-correct"]['attempted'], coverage["answerable-correct"]['total'], no_frac=True),
    'Answerable-Incorrect (Corrected %)': format_metric(answerable_incorrect_execution_accuracy['corrected'], answerable_incorrect_execution_accuracy['total'], no_frac=True),
    'Answerable-Incorrect (Coverage %)': format_metric(coverage["answerable-incorrect"]['attempted'], coverage["answerable-incorrect"]['total'], no_frac=True),
    'Ambiguous (Precision %)': format_metric(ambiguous_precision['correct'], ambiguous_precision['total'], no_frac=True),
    'Ambiguous (Recall %)': format_metric(ambiguous_recall['correct'], ambiguous_recall['total'], no_frac=True),
    'Ambiguous (F1 %)': calculate_f1(ambiguous_precision['correct'], ambiguous_precision['total'], ambiguous_recall['correct'], ambiguous_recall['total']),
    'Unanswerable (Precision %)': format_metric(unanswerable_precision['correct'], unanswerable_precision['total'], no_frac=True),
    'Unanswerable (Recall %)': format_metric(unanswerable_recall['correct'], unanswerable_recall['total'], no_frac=True),
    'Unanswerable (F1 %)': calculate_f1(unanswerable_precision['correct'], unanswerable_precision['total'], unanswerable_recall['correct'], unanswerable_recall['total'])
}
table5_data.append(main_result)

results_data['Table5_Main_Results'] = table5_data

# Table 6 - Granular Results (if granular evaluation is enabled)
if args.granular_evaluation:
    table6_data = []
    
    # Granular answerable-incorrect results
    for key, val in answerable_incorrect_execution_accuracy_granular.items():
        if val['total'] > 0:
            granular_result = {
                'Category': f"Answerable-Incorrect ({key})",
                'Metric': 'Execution Accuracy',
                'Value (%)': format_metric(val['corrected'], val['total'], no_frac=True),
                'Fraction': f"{val['corrected']}/{val['total']}"
            }
            table6_data.append(granular_result)
    
    # Granular not answerable results
    for key in recall_granular:
        if recall_granular[key]['total'] > 0:
            precision_val = format_metric(precision_granular[key]['correct'], precision_granular[key]['total'], no_frac=True)
            recall_val = format_metric(recall_granular[key]['correct'], recall_granular[key]['total'], no_frac=True)
            f1_val = calculate_f1(precision_granular[key]['correct'], precision_granular[key]['total'],
                                  recall_granular[key]['correct'], recall_granular[key]['total'])
            
            granular_result = {
                'Category': key,
                'Precision (%)': precision_val,
                'Recall (%)': recall_val,
                'F1 (%)': f1_val,
                'Recall Fraction': f"{recall_granular[key]['correct']}/{recall_granular[key]['total']}",
                'Precision Fraction': f"{precision_granular[key]['correct']}/{precision_granular[key]['total']}"
            }
            table6_data.append(granular_result)
    
    results_data['Table6_Granular_Results'] = table6_data

# Save results to Excel
save_results_to_excel(args.model_name, args.corrector_name, results_data)

# Print results (keeping original output format)
print("ANSWERABLE-CORRECT (Decrease in Execution Accuracy (↑)) & Coverage (↑)")
print(format_metric(answerable_correct_execution_accuracy['preserved'], answerable_correct_execution_accuracy['total']), end= " & ")
print(format_metric(coverage["answerable-correct"]['attempted'], coverage["answerable-correct"]['total']), end= " & ")
print()

print("ANSWERABLE-INCORRECT (Increase in Execution Accuracy (↑)) & Coverage (↑)")
print(format_metric(answerable_incorrect_execution_accuracy['corrected'], answerable_incorrect_execution_accuracy['total']), end= " & ")
print(format_metric(coverage["answerable-incorrect"]['attempted'], coverage["answerable-incorrect"]['total']), end= " & ")
print()

if args.granular_evaluation:
    print("GRANULAR - ANSWERABLE-INCORRECT (Execution Accuracy (↑))")
    for key, val in answerable_incorrect_execution_accuracy_granular.items():
        if val['total'] > 0:  # Only show categories with data
            print(f"{key}: {format_metric(val['corrected'], val['total'])}")
    print()
    
    print("GRANULAR - NOT ANSWERABLE Precision (↑) & Recall (↑) & F1 (↑))")
    for key in recall_granular:
        if recall_granular[key]['total'] > 0:  # Only show categories with data
            precision_str = format_metric(precision_granular[key]['correct'], precision_granular[key]['total'])
            recall_str = format_metric(recall_granular[key]['correct'], recall_granular[key]['total'])
            f1_str = calculate_f1(precision_granular[key]['correct'], precision_granular[key]['total'],
                                  recall_granular[key]['correct'], recall_granular[key]['total'])
            print(f"{key}: {precision_str} & {recall_str} & {f1_str}")
    print()

print("AMBIGUOUS (Precision (↑) & Recall (↑) & F1 (↑))")
precision_str = format_metric(ambiguous_precision['correct'], ambiguous_precision['total'])
recall_str = format_metric(ambiguous_recall['correct'], ambiguous_recall['total'])
f1_str = calculate_f1(ambiguous_precision['correct'], ambiguous_precision['total'], 
                      ambiguous_recall['correct'], ambiguous_recall['total'])
print(f"{precision_str} & {recall_str} & {f1_str}")
print()

print("UNANSWERABLE (Precision (↑) & Recall (↑) & F1 (↑))")
precision_str = format_metric(unanswerable_precision['correct'], unanswerable_precision['total'])
recall_str = format_metric(unanswerable_recall['correct'], unanswerable_recall['total'])
f1_str = calculate_f1(unanswerable_precision['correct'], unanswerable_precision['total'], 
                      unanswerable_recall['correct'], unanswerable_recall['total'])
print(f"{precision_str} & {recall_str} & {f1_str}")

print("=" * 20)

print(format_metric(answerable_correct_execution_accuracy['preserved'], answerable_correct_execution_accuracy['total'], no_frac=True), end= " & ")
print(format_metric(coverage["answerable-correct"]['attempted'], coverage["answerable-correct"]['total'], no_frac=True), end= " & ")
print(format_metric(answerable_incorrect_execution_accuracy['corrected'], answerable_incorrect_execution_accuracy['total'], no_frac=True), end= " & ")
print(format_metric(coverage["answerable-incorrect"]['attempted'], coverage["answerable-incorrect"]['total'], no_frac=True), end= " & ")
precision_str = format_metric(ambiguous_precision['correct'], ambiguous_precision['total'], no_frac=True)
recall_str = format_metric(ambiguous_recall['correct'], ambiguous_recall['total'], no_frac=True)
f1_str = calculate_f1(ambiguous_precision['correct'], ambiguous_precision['total'], 
                      ambiguous_recall['correct'], ambiguous_recall['total'])
print(f"{precision_str} & {recall_str} & {f1_str}", end= " & ")
precision_str = format_metric(unanswerable_precision['correct'], unanswerable_precision['total'], no_frac=True)
recall_str = format_metric(unanswerable_recall['correct'], unanswerable_recall['total'], no_frac=True)
f1_str = calculate_f1(unanswerable_precision['correct'], unanswerable_precision['total'], 
                      unanswerable_recall['correct'], unanswerable_recall['total'])
print(f"{precision_str} & {recall_str} & {f1_str}")

print("=" * 100)
print(f"Results saved to evaluation_results.xlsx with model: {args.model_name}, corrector: {args.corrector_name}")

