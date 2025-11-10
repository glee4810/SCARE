import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
import litellm
from litellm import completion
from litellm.exceptions import BadRequestError
from pydantic import BaseModel
import re
import sqlparse
import sqlite3
import random
from sqlglot import parse_one, exp
from func_timeout import func_timeout, FunctionTimedOut
from typing import Any, Optional, Union, List, Tuple, Dict, Callable

from multiprocessing.dummy import Pool


CURRENT_DATE = "2100-12-31"
CURRENT_TIME = "23:59:00"
NOW = f"{CURRENT_DATE} {CURRENT_TIME}"
PRECOMPUTED_DICT = {
    'temperature': (35.5, 38.1),
    'sao2': (95.0, 100.0),
    'heart rate': (60.0, 100.0),
    'respiration': (12.0, 18.0),
    'systolic bp': (90.0, 120.0),
    'diastolic bp': (60.0, 90.0),
    'mean bp': (60.0, 110.0)
}
TIME_PATTERN = r"(DATE_SUB|DATE_ADD)\((\w+\(\)|'[^']+')[, ]+ INTERVAL (\d+) (MONTH|YEAR|DAY)\)"
POSTPROCESS_VAL_DICT = {'advising': {'Organogenesis: Stem Cells to Regenerative Biology': 'Organogenesis:  Stem Cells to Regenerative Biology'}}


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def process_item(item, db_id):
    try:
        item = round(float(item),3)
    except:
        pass
    return str(item)

def process_answer(ans, db_id):
    if type(ans)==str: # null
        return ans
    else:
        return str(sorted([[process_item(c, db_id) for c in row] for row in ans])[:100]) # check only up to 100th record

def postprocess_gt(query, db_id):

    if 'select' not in query.lower(): # remove non-select queries
        return query

    if "current_time" in query: # strftime('%J',current_time) => strftime('%J','2100-12-31 23:59:00')
        query = query.replace("current_time", f"'{NOW}'")
    if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query): # systolic_bp_lower => 90.0
        vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
        vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
        vital_name_list = list(set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr) + re.findall('([a-zA-Z0-9_]+)_upper', vital_upper_expr)))
        if len(vital_name_list) == 1:
            processed_vital_name = vital_name_list[0].replace('_', ' ')
            if processed_vital_name in PRECOMPUTED_DICT:
                vital_range = PRECOMPUTED_DICT[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr, f"{vital_range[1]}")
    query = query.replace("%y", "%Y").replace('%j', '%J') # strftime('%y-%m',outputevents.charttime) => strftime('%Y-%m',outputevents.charttime)

    return query

def postprocess_pred(query, db_id):

    if 'select' not in query.lower(): # remove non-select queries
        return query
    
    query = query.replace('```sql', '').replace('```', '') # function calling filtering
    query = query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=') # tokenization adjustment for open-source models
    query = re.sub('[ ]+', ' ', query.replace('\n', ' ')).strip()

    pattern = r'"([^\']*)"'
    query = re.sub(pattern, r"'\1'", query)

    if db_id in POSTPROCESS_VAL_DICT:
        for before, after in POSTPROCESS_VAL_DICT[db_id].items():
            query = query.replace(before, after)

    if "current_time" in query: # strftime('%J',current_time) => strftime('%J','2100-12-31 23:59:00')
        query = query.replace("current_time", f"'{NOW}'")
    if "'now'" in query: # 'now' => '2100-12-31 23:59:00'
        query = query.replace("'now'", f"'{NOW}'")
    if "NOW()" in query: # NOW() => '2100-12-31 23:59:00'
        query = query.replace("NOW()", f"'{NOW}'")
    if "CURDATE()" in query: # CURDATE() => '2100-12-31'
        query = query.replace("CURDATE()", f"'{CURRENT_DATE}'")
    if "CURTIME()" in query: # CURTIME() => '23:59:00'
        query = query.replace("CURTIME()", f"'{CURRENT_TIME}'")

    if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query): # systolic_bp_lower => 90.0
        vital_lower_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
        vital_upper_expr = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
        vital_name_list = list(set(re.findall('([a-zA-Z0-9_]+)_lower', vital_lower_expr) + re.findall('([a-zA-Z0-9_]+)_upper', vital_upper_expr)))
        if len(vital_name_list) == 1:
            processed_vital_name = vital_name_list[0].replace('_', ' ')
            if processed_vital_name in PRECOMPUTED_DICT:
                vital_range = PRECOMPUTED_DICT[processed_vital_name]
                query = query.replace(vital_lower_expr, f"{vital_range[0]}").replace(vital_upper_expr, f"{vital_range[1]}")
    query = query.replace("%y", "%Y").replace('%j', '%J') # strftime('%y-%m',outputevents.charttime) => strftime('%Y-%m',outputevents.charttime)

    return query


def modify_distinct(pred, real):

    pred = pred.strip()
    
    if not isinstance(pred, str):
        pred = str(pred)

    # Early returns if conditions are not met
    if pred.lower() == 'null' or real.lower() == 'null' or 'select' not in pred.lower():
        return pred

    # Define regex patterns to check the presence of DISTINCT right after SELECT.
    # We'll ignore case differences and assume the top-level SELECT is at the start of the query.
    select_distinct_pattern = re.compile(r"(?i)\bSELECT\s+DISTINCT\b")
    select_pattern = re.compile(r"(?i)\bSELECT\b")

    # Check if gold_sql has DISTINCT
    gold_has_distinct = bool(select_distinct_pattern.search(real))
    pred_has_distinct = bool(select_distinct_pattern.search(pred))

    if gold_has_distinct and not pred_has_distinct:
        # Gold requires DISTINCT, but pred doesn't have it. Insert DISTINCT right after SELECT.
        # Replace the first occurrence of SELECT (not DISTINCT), ensuring we don't double insert.
        if select_distinct_pattern.search(pred):
            # Already has DISTINCT, do nothing
            pass
        else:
            # Insert DISTINCT after SELECT
            pred = select_pattern.sub("SELECT DISTINCT", pred, count=1)
    elif not gold_has_distinct and pred_has_distinct:
        # Gold does not have DISTINCT, but pred does. Remove DISTINCT.
        # Replace 'SELECT DISTINCT' with 'SELECT'
        pred = select_distinct_pattern.sub("SELECT", pred, count=1)

    return pred
def creating_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            if isinstance(primary_key, list):
                continue
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    pd_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    pd_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    pd_foreign = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return pd_schema, pd_primary, pd_foreign

def find_foreign_keys_MYSQL_like(foreign, db_name):
    df = foreign[foreign['Database name'] == db_name]
    output = "["
    for index, row in df.iterrows():
        output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ', '
    output = output.strip()
    output= output[:-1] + "]"
    if len(output)==1:
        output = '[]'
    return output

def find_fields_MYSQL_like(schema, db_name):
    df = schema[schema['Database name'] == db_name]
    df = df.groupby(' Table Name')
    output = ""
    for name, group in df:
        output += "Table " +name+ ', columns = ['
        for index, row in group.iterrows():
            output += row[" Field Name"]+', '
        output = output.strip()
        output = output[:-1]
        output += "]\n"
    return output

def find_primary_keys_MYSQL_like(primary, db_name):
    df = primary[primary['Database name'] == db_name]
    output = "["
    for index, row in df.iterrows():
        end = ", "
        output += row['Table Name'] + '.' + row['Primary Key'] + end
    output = output[:-2]
    output += "]"
    if len(output)==1:
        output = '[]'
    return output
def create_db_schema_prompt(schema:pd.DataFrame, foreign:pd.DataFrame, database:str):
    db_schema_prompt = f"[DB ID]: {database}\n"
    db_schema_prompt += find_fields_MYSQL_like(schema, database).strip()
    db_schema_prompt += "\n\nForeign_keys = " + find_foreign_keys_MYSQL_like(foreign, database).strip()
    return db_schema_prompt
def execute_sql(sql, db_path, timeout=60):
    def inner_execute():
        con = sqlite3.connect(db_path, check_same_thread=False)
        con.text_factory = lambda b: b.decode(errors="ignore")
        cur = con.cursor()
        cur.execute(sql)
        result = cur.fetchall()
        con.close()
        return result
    try:
        return func_timeout(timeout=timeout, func=inner_execute)
    except FunctionTimedOut as e:
        return str(e)
    except Exception as e:
        return str(e)
        
class SQLEvaluator:
    def __init__(self, data_dir: str, dataset: str):

        self.data_dir = data_dir
        self.dataset = dataset
        self.table_path = f"{data_dir}/tables.json"

        try:
            with open(self.table_path, "r") as f:
                self.tables = json.load(f)
        except Exception as e:
            raise ValueError(f"Error in loading tables.json: {e}")

        self.column_names = [column[1] for db in self.tables if db['db_id'] == self.dataset for column in db['column_names_original']]

    def get_gold_columns_only(self, sql: str):

        try:
            # Parse the SQL query
            parsed = parse_one(sql, read='sqlite', error_level='ignore')
            
            # Extract all column references
            columns = parsed.find_all(exp.Column)
            
            # Get unique column names, ignoring table names
            extracted_cols = list(set(col.name for col in columns))
            
            # Create lowercase versions of self.column_names for case-insensitive comparison
            column_names_lower = {col.lower(): col for col in self.column_names}
            
            # Filter columns - match case-insensitively but return original case from self.column_names
            column_list = []
            for col in extracted_cols:
                col_lower = col.lower()
                if (not col_lower.endswith('id')) and (col_lower in column_names_lower):
                    column_list.append(column_names_lower[col_lower])
            
            return column_list
        except Exception as e:
            #print(e)
            return []

    def check_answer(self, gold_answer, pred_answer, gt_sql, db_id):

        if str(gold_answer).startswith('[Error]') or str(pred_answer).startswith('[Error]'):
            return False

        try:
            gold_answer = eval(gold_answer)
        except:
            pass
        try:
            pred_answer = eval(pred_answer)
        except:
            pass

        # handling gold multi-column
        if db_id == 'mimicsql':
            try:
                exec_acc = True
                for i in range(len(gold_answer)):
                    if sorted(set(gold_answer[i])) != sorted(set(pred_answer[i])):
                        exec_acc = False
                        break
            except:
                exec_acc = False
            return exec_acc
            
        # handling gold single-column
        else:

            if len(pred_answer)==0 and len(gold_answer) > 0:
                exec_acc = False
                return exec_acc
            elif len(pred_answer)==0 and len(gold_answer)==0:
                exec_acc = True
                return exec_acc
                
            if len(pred_answer[0]) == 1:

                # is_count = 'count' in gt_sql.lower() # count( * )
                # if is_count and pred_answer=='[]':
                #     pred_answer = [['0.0']]
                # is_count = re.search(r'\bcount\s*\([^)]*\)\s*>\s*0\s*from\b', gt_sql.lower()) # count( * ) > 0 
                # if is_count:
                #     pred_answer_set = [list(t) for t in set(tuple(x) for x in pred_answer)]
                #     if pred_answer_set == [['None']]:
                #         pred_answer = [['0.0']]
                if ('AVG' in gt_sql and 'CASE' in gt_sql): # calculating survival rate
                    try: # 100.0 => 1.0
                        converted = float(pred_answer[0][0])
                        if converted > 1.0:
                            pred_answer = [[str(round(converted/100, 3))]]
                    except:
                        pass
                exec_acc = (gold_answer == pred_answer)

            elif len(pred_answer[0]) > 1:

                exec_acc = False
                flattened_pred_answer = list(zip(*pred_answer))
                for i in range(len(flattened_pred_answer)):
                    if sorted(set([r for r in flattened_pred_answer[i] if r != 'None'])) == sorted(set([el[0] for el in gold_answer])):
                        exec_acc = True
                        break

            else:
                exec_acc = False

        return exec_acc

    def execute(self, db_id:str, sql:str, is_gold_sql:bool, timeout:int=60):
        
        assert os.path.exists(f"{self.data_dir}/{self.dataset}/{db_id}.sqlite"), f"Database file does not exist: {self.data_dir}/{self.dataset}/{db_id}.sqlite"

        if is_gold_sql:
            processed_sql = postprocess_gt(sql, db_id=db_id)
        else:
            processed_sql = postprocess_pred(sql, db_id=db_id)

        if processed_sql in ['null', 'ambiguous', 'unanswerable', 
                            'ambiguous-reference', 'vague-word', 'vague-question', 
                            'infeasible-faq', 'missing-column', 'small-talk', 'out-of-scope']:
            return processed_sql
        
        execution_result = execute_sql(
            sql=processed_sql,
            db_path=f"{self.data_dir}/{self.dataset}/{db_id}.sqlite",
            timeout=timeout
        )
        
        # Only process the answer if execution was successful
        if not str(execution_result).startswith('[Error]'):
            execution_result = process_answer(execution_result, db_id=db_id)
        
        return execution_result

    def __call__(self, db_id:str, pred_sql:str, gold_sql:str, gold_answer:str=None):

        pred_sql = modify_distinct(pred_sql, gold_sql)
        pred_answer = self.execute(db_id, pred_sql, is_gold_sql=False)
        
        if gold_answer is None:
            gold_answer = self.execute(db_id, gold_sql, is_gold_sql=True)
        
        is_correct = self.check_answer(gold_answer, pred_answer, gold_sql, db_id)

        result = {
            "pred_answer": pred_answer,
            "gold_answer": gold_answer,
            "is_correct": is_correct
        }
        return result


def create_schema_prompt(dataset_json, db_id, db_path=None):
    """Create a schema prompt in the format of question_exp.txt with sample data from database"""
    schema_df = pd.read_json(dataset_json)
    prompt = ""
    
    # Connect to the database if path is provided
    conn = None
    if db_path:
        conn = sqlite3.connect(db_path)
        conn.text_factory = str
    
    for _, row in schema_df.iterrows():
        if db_id != row['db_id']:
            continue
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        
        # Process each table
        for table_idx, table_name in enumerate(tables):
            # Create CREATE TABLE statement
            prompt += f"CREATE TABLE {table_name} (\n"
            # Add columns
            columns = []
            table_columns = []  # Store column names for INSERT statement
            for (idx, col_name), col_type in zip(col_names, col_types):
                if idx == table_idx:
                    columns.append(f"{col_name} {col_type}")
                    table_columns.append(col_name)
            
            # Add primary keys
            pk_cols = []
            for pk in primary_keys:
                idx, col_name = col_names[pk]
                if idx == table_idx:
                    pk_cols.append(col_name)
            if pk_cols:
                columns.append(f"primary key ( {' , '.join(pk_cols)} )")
            
            # Add foreign keys
            for fk in foreign_keys:
                first, second = fk
                first_idx, first_col = col_names[first]
                second_idx, second_col = col_names[second]
                if first_idx == table_idx:
                    ref_table = tables[second_idx]
                    columns.append(f"foreign key ( {first_col} ) references {ref_table} ( {second_col} )")
            
            prompt += ' ,\n'.join(columns)
            prompt += "\n)\n"
            
            # Add sample insert statement from database
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                    row = cursor.fetchone()
                    if row:
                        # Format values properly based on their type
                        values = []
                        for val in row:
                            if val is None:
                                values.append('NULL')
                            elif isinstance(val, (int, float)):
                                values.append(str(val))
                            else:
                                # Escape single quotes and wrap in quotes
                                val_str = str(val).replace("'", "''")
                                values.append(f"'{val_str}'")
                        
                        prompt += '-- Sample insert (only first row shown, more data omitted)\n'
                        prompt += f"insert into {table_name} ({', '.join(table_columns)}) values ({', '.join(values)})\n"
                except sqlite3.OperationalError as e:
                    print(f"Warning: Could not read sample data from table {table_name}: {str(e)}")
                        
            prompt += "\n"
    
    # Close database connection
    if conn:
        conn.close()
    
    return prompt



def is_reasoning_llm(model: str):
    return model in ['gpt-5', 'gpt-5-mini', 'o4-mini']


def llm_call_with_retry(model: str, system_prompt: str, user_prompt: str, 
                       response_format: BaseModel, temperature: float = 0.0, few_shots: List[Dict] = None, base_url: str = None, max_retries: int = 5,
                       ):
    """Call LiteLLM complete with structured output and retry logic."""
    usage_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cost': 0.0, 'llm_calls': 0}

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add few-shot examples if provided
    if few_shots:
        for example in few_shots:
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})
        
    messages.append({"role": "user", "content": user_prompt})

    # for attempt in range(max_retries):
    while True:
        try:
            output = completion(
                model=model,
                messages=messages,
                temperature=None if is_reasoning_llm(model) else temperature,
                response_format=response_format,
                base_url=base_url,
                custom_llm_provider="openai" if base_url else None,
            )

            cost = 0.0
            if hasattr(output, '_hidden_params') and 'response_cost' in output._hidden_params and output._hidden_params["response_cost"]:
                cost += output._hidden_params["response_cost"]
            usage_info['prompt_tokens'] += output.usage.prompt_tokens
            usage_info['completion_tokens'] += output.usage.completion_tokens
            usage_info['total_tokens'] += output.usage.total_tokens
            usage_info['cost'] += round(cost, 4)
            usage_info['llm_calls'] += 1

            try:
                return eval(output.choices[0].message.content), usage_info
            except (SyntaxError, NameError, ValueError) as eval_error:
                try:
                    return json.loads(output.choices[0].message.content), usage_info
                except json.JSONDecodeError:
                    raise Exception(f"Failed to parse LLM response: {eval_error}")
        except BadRequestError as e:
            return {"error": "Context window exceeded"}, usage_info
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            # if attempt == max_retries - 1:
            #     raise e            
            time.sleep(3)
            continue


def llm_call_with_retry_messages(model: str, messages: List[Dict], 
                       response_format: BaseModel, temperature: float = 0.0, base_url: str = None, max_retries: int = 5,
                       ):
    """Call LiteLLM complete with structured output and retry logic."""
    usage_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'cost': 0.0, 'llm_calls': 0}
    
    # for attempt in range(max_retries):
    while True:    
        try:
            output = completion(
                model=model,
                messages=messages,
                temperature=None if is_reasoning_llm(model) else temperature,
                response_format=response_format,
                base_url=base_url,
                custom_llm_provider="openai" if base_url else None,
            )

            cost = 0.0
            if hasattr(output, '_hidden_params') and 'response_cost' in output._hidden_params and output._hidden_params["response_cost"]:
                cost += output._hidden_params["response_cost"]
            usage_info['prompt_tokens'] += output.usage.prompt_tokens
            usage_info['completion_tokens'] += output.usage.completion_tokens
            usage_info['total_tokens'] += output.usage.total_tokens
            usage_info['cost'] += round(cost, 4)
            usage_info['llm_calls'] += 1

            try:
                return eval(output.choices[0].message.content), usage_info
            except (SyntaxError, NameError, ValueError) as eval_error:
                try:
                    return json.loads(output.choices[0].message.content), usage_info
                except json.JSONDecodeError:
                    raise Exception(f"Failed to parse LLM response: {eval_error}")
        except BadRequestError as e:
            return {"error": "Context window exceeded"}, usage_info
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            # if attempt == max_retries - 1:
            #     raise e
            time.sleep(3)
            continue


def format_sql_result(result) -> str:
    """Format SQL execution result for display."""
    if result and len(result) > 0:
        formatted_rows = ['| ' + ' | '.join([str(col) for col in row]) + ' |' 
                        for row in result[:5]]
        return '\n' + '\n'.join(formatted_rows)
    return 'None'


def get_schema_and_evidence(instance, tables_path, db_path):
    """Get database schema and evidence for an instance."""
    db_id = instance['db_id']
    evidence = open_file(f"{db_path}/{db_id}_assumption.txt")
    
    schema = create_schema_prompt(tables_path, db_id=db_id, 
                                 db_path=f"{db_path}/{db_id}/{db_id}.sqlite")

    return db_id, schema, evidence


def save_results_safely(result_data, output_path):
    """Safely save results to file with error handling."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=4)
    except Exception as e:
        print(f"Error saving results: {e}")


def create_metadata(instance, args):

    instance_id = instance['id']
    question = instance['question']
    template = instance['template']
    gold_sql = instance['gt_sql']
    gold_answer = instance['gt_exe']
    pred_sql = instance['pred_sql']
    pred_answer = instance['pred_exe']
    exec_acc = instance['exec_acc']
    sample_type = instance['type']
    generator_name = instance['generator_name']
    db_id, schema, evidence = get_schema_and_evidence(instance, args.tables_path, args.db_path)

    # Initialize result
    result = {}
    result['id'] = f"{generator_name}_{instance_id}"
    result['db_id'] = db_id
    result['question'] = question
    result['template'] = template
    result['evidence'] = evidence
    result['schema'] = schema
    result['init_pred_sql'] = pred_sql
    result['init_pred_sql_exec_result'] = pred_answer
    result['init_exec_acc'] = exec_acc
    result['gold_sql'] = gold_sql
    result['gold_answer'] = gold_answer
    result['sample_type'] = sample_type

    return result


def process_instances_single_core(instances, args, process_func: Callable, result_data, desc="Processing instances (single-core)"):
    """Process instances using single core with tqdm progress bar."""
    processed_count = 0
    processed_ids = {item['id'] for item in result_data if 'id' in item}
    
    for instance in tqdm(instances, desc=desc):
        instance_id = f"{instance['generator_name']}_{instance['id']}"
        
        # Skip if already processed
        if instance_id in processed_ids:
            continue
            
        instance_id, result = process_func(instance)
        result_data.append(result)
        processed_ids.add(instance_id)
        processed_count += 1
        
        # Auto-save at intervals
        if processed_count % args.save_interval == 0:
            save_results_safely(result_data, args.data_output_path)
            print(f"Auto-saved after processing {processed_count} instances")
    
    return result_data


def process_instances_multi_core(instances, args, process_func: Callable, result_data, desc="Processing instances (multi-core)"):
    """Process instances using multiprocessing with tqdm progress bar."""
    # Filter out already processed instances
    processed_ids = {item['id'] for item in result_data if 'id' in item}
    unprocessed_instances = [
        instance for instance in instances 
        if f"{instance['generator_name']}_{instance['id']}" not in processed_ids
    ]
    
    if not unprocessed_instances:
        print("All instances already processed.")
        return result_data
    
    print(f"Processing {len(unprocessed_instances)} instances with {args.num_process} processes")
    
    processed_count = 0
    
    # Process all instances at once and save results as they complete
    with Pool(processes=args.num_process) as pool:
        # Use imap_unordered for processing all instances and getting results as they complete
        results_iter = pool.imap_unordered(process_func, unprocessed_instances)
        
        # Process results as they complete
        for instance_id, result in tqdm(results_iter, total=len(unprocessed_instances), desc=desc):
            result_data.append(result)
            processed_count += 1
            
            # Auto-save at intervals based on completed instances
            if processed_count % args.save_interval == 0:
                save_results_safely(result_data, args.data_output_path)
                print(f"Auto-saved after processing {processed_count} instances")
    
    return result_data


def sample_unique_templates(data_list, max_count=10, unique_key='question'):
    """Sample unique templates from data list."""
    seen_templates = set()
    result = []        
    for item in sorted(data_list, key=lambda x: x['id']):
        template = item[unique_key]
        if template not in seen_templates:
            seen_templates.add(template)
            result.append(item)
            if len(result) >= max_count:
                break
    return result


def setup_subset_data(sql_results, args):
    """Setup subset data based on dataset type."""
    if not args.subset:
        return sql_results
        
    sql_results_samples = []
    if sql_results[0]['db_id'] == 'mimicsql':
        # Use question as unique key for mimicsql
        unique_key = 'question'
        sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['easy'] and l['exec_acc'] == True], 20, unique_key=unique_key)
        sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['easy'] and l['exec_acc'] == False], 20, unique_key=unique_key)
        sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['medium'] and l['exec_acc'] == True], 20, unique_key=unique_key)
        sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['medium'] and l['exec_acc'] == False], 20, unique_key=unique_key)
        sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['hard'] and l['exec_acc'] == True], 20, unique_key=unique_key)
        sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['hard'] and l['exec_acc'] == False], 20, unique_key=unique_key)
    else:
        # Use template as unique key for other datasets
        unique_key = 'template'
        sql_results_samples = sample_unique_templates([l for l in sql_results if l['type'] in ['easy'] and l['exec_acc'] == True], 20, unique_key=unique_key)
        sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['easy'] and l['exec_acc'] == False], 20, unique_key=unique_key)
        sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['medium'] and l['exec_acc'] == True], 20, unique_key=unique_key)
        sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['medium'] and l['exec_acc'] == False], 20, unique_key=unique_key)
        sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['hard'] and l['exec_acc'] == True], 20, unique_key=unique_key)
        sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['hard'] and l['exec_acc'] == False], 20, unique_key=unique_key)
    
    # Add special types
    sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['vague-question']], 20, unique_key='question')
    sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['vague-word']], 20, unique_key='question')
    sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['ambiguous-reference']], 20, unique_key='question')
    sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['small-talk']], 20, unique_key='question')
    sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['out-of-scope']], 20, unique_key='question')
    sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['missing-column']], 20, unique_key='question')
    sql_results_samples += sample_unique_templates([l for l in sql_results if l['type'] in ['infeasible-faq']], 20, unique_key='question')
    
    return sql_results_samples
