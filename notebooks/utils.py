import copy
import base64
import itertools
import json
import numpy as np
import os
import pandas as pd
import re

'''Data/file manipulation utilities'''

from valueconsistency import MODEL_NAMES_SHORT, option_columns, answer_columns, add_distribution_to_classificaiton

STATIC_BASELINE = 'Static Baseline'

CHAT_MODELS = [
'gpt-4o',
'meta-llama/Meta-Llama-3-70B-Instruct',
'meta-llama/Llama-2-70b-chat-hf',
'meta-llama/Meta-Llama-3-8B-Instruct',
'meta-llama/Llama-2-7b-chat-hf',
'CohereForAI/c4ai-command-r-v01',
'01-ai/Yi-34B-Chat',
'stabilityai/japanese-stablelm-instruct-beta-70b'
]


def group_tasks_by_run(all_dfs, group=True, value_context_agnostic=False):
    task_agnostic = {}
    for var, df in all_dfs:
        cpy = copy.deepcopy(var)
        if cpy['task'] == 'generation':
            if 'annotator' in cpy:
                del cpy['annotator']
                del cpy['annotator-temperature']
            else:
                continue # don't join with the non annotated versions
        irrelevant_variables = ['task', 'filename', 'allow_abstentions',
                               'randomize_option_order', 'follow_up', 'example_answer']
        if value_context_agnostic:
            irrelevant_variables += ['use_values', 'use_context', 'num_values',
                                     'most_ambiguous_value', 'unrelated_value',
                                     'static_value']
        for irrelevant in irrelevant_variables:
            del cpy[irrelevant]
        key = frozenset(cpy.items())
        if key not in task_agnostic:
            task_agnostic[key] = ()
        task_agnostic[key] += ((var, df),)

    if group:
        for key in list(task_agnostic.keys()):
            if len(task_agnostic[key]) != 2: # if more tasks, change this number
                del task_agnostic[key]
    return list(task_agnostic.values())

def group_annotations_by_run(var_dfs, group=True, value_context_agnostic=True):
    task_agnostic = {}
    for var, df in var_dfs:
        cpy = copy.deepcopy(var)
        if 'annotator' in cpy:
            if 'annotator_temperature' not in cpy:
                import pdb; pdb.set_trace()
            del cpy['annotator']
            del cpy['annotator_temperature']
        else:
            continue # don't join with the non annotated versions
        irrelevant_variables = ['system_prompt', 'single_letter_prompt', 'example_answer',
                                'single_question', 'follow_up', 'example_answer']
        if value_context_agnostic:
            irrelevant_variables += ['use_values', 'use_context', 'num_values',
                                     'most_ambiguous_value', 'unrelated_value',
                                     'static_value']
        for irrelevant in irrelevant_variables:
            if irrelevant in cpy:
                del cpy[irrelevant]
        cpy['filename'] = data_filename(cpy['filename'])
        key = frozenset(cpy.items())
        if key not in task_agnostic:
            task_agnostic[key] = ()
        task_agnostic[key] += ((var, df),)

    if group:
        for key in list(task_agnostic.keys()):
            if len(task_agnostic[key]) < 2: 
                del task_agnostic[key]

    return list(task_agnostic.values())

def get_matching_data(df_list, desired_variables, only_annotated=True):
    # could expand this with kwargs
    for variables, df in df_list:
        for variable, value in desired_variables.items():
            if (variable not in variables
                    or (isinstance(value, list) and variables[variable] not in value)
                    or (not isinstance(value, list) and variables[variable] != value)):
                break
        else: # it matches!
            if (only_annotated and variables['task'] == 'generation'
                    and 'annotator' not in variables):
                continue
            return variables, df
    return None

def is_schwartz(var):
    return 'wiki_controversial' not in var['filename']

def data_filename(filename):
    split = filename.split(os.path.sep)
    return split[1] if len(split) > 1 else filename

def chain_var_dfs(all_dfs):
    return list(itertools.chain(*all_dfs.values()))

def errors_dict_reshape(errors):
    d = pd.DataFrame(errors).to_dict()
    
    res = []
    for k, v in d.items():
        first = []
        second = []
        for err in v.values():
            if not isinstance(err, np.ndarray):
                err = [np.nan, np.nan]
            first.append(err[0])
            second.append(err[1])
        res.append([first, second])
    return np.array(res)

###### mturk utils


def base64_decode(encoded):
    json_text = base64.b64decode(bytes(encoded, 'utf-8')).decode('utf-8')
    return json.loads(json_text)

def mturk_explode_df(df):
    keep_columns = ['WorkerId', 'topic']
    if 'Input.original' in df.columns:
        keep_columns.append('original')
    df = df.rename(columns = {'Input.topic' : 'topic', 'Input.original' : 'original'})
    df['questions_json'] = df['Input.questions'].apply(base64_decode)

    variable_columns = {'Answer.q_{0}_question' : 'question',
                        'Answer.q_{0}_options' : 'options',
                        'Answer.q_{0}' : 'answer',
                        'Answer.q_{0}_attn' : 'attention-response',
                        'Answer.q_{0}_attn_answer' : 'attention-answer'}
    frames = []
    exact_matches = [col for col in df.columns if re.fullmatch(r'Answer.q_\d', col)]
    num_questions = len(exact_matches)

    for i in range(num_questions):
        number_cols = {k.format(i): v for k, v in variable_columns.items()}
        selected = df[keep_columns + list(number_cols.keys())].rename(columns=number_cols)
        # selected['question'] = df['questions'].apply(lambda x: x[i]['question'] if i < len(x) else None)
        selected['options'] = df['questions_json'].apply(lambda x: x[i]['options'] if i < len(x) else None)
        frames.append(selected)
    # This will have some null rows; some topics have more questions than others
    df_exploded = pd.concat(frames).dropna()
    df_exploded = pd.concat([df_exploded, df_exploded.apply(add_probability_columns, axis=1)], axis=1)
    df_exploded = add_distribution_to_classificaiton(df_exploded)
    # TODO: later anonymize WorkerIds; treat each as an independent sample
    df_exploded = df_exploded.rename(columns={'WorkerId' : 'sample'})
    return df_exploded

def add_probability_columns(row):
    options = list(row['options'].keys())
    num_options = len(options)
    answer_idx = options.index(row['answer'])
    answers = ([0] * num_options)
    answers[answer_idx] = 1
    result = {}
    opt_columns = option_columns(num_options)
    ans_columns = answer_columns(num_options)
    result.update({k : v for k, v in zip(opt_columns, options)})
    result.update({k : v for k, v in zip(ans_columns, answers)})
    return pd.Series(result)

def passed_attention_checks(df):
    return (df[df['attention-answer'] == 
                df['attention-response']])

####

def shorten_var_name(var, name):
    if name == 'filename':
        return data_filename(var['filename'])
    elif name == "model_type":
        return 'chat' if var['model'] in CHAT_MODELS else 'base'
    elif name == 'country':
        country = data_filename(var['filename']).split('_')[2]
        return 'U.S.' if country == 'us' else country.capitalize()
    elif name == 'language':
        return data_filename(var['filename']).split('_')[1].capitalize()
    elif name == 'controversial':
        return data_filename(var['filename']).split('_')[0]
    elif name == 'model' or name == 'annotator':
        if var[name] in MODEL_NAMES_SHORT:
            return MODEL_NAMES_SHORT[var[name]]
        else:
            return var[name]
    elif name == 'task':
        if var[name] == 'generation':
            return 'open-ended'
        elif var[name] == 'classification':
            return 'multiple-choice'
    elif name == 'allow_abstentions':
        return ('' if var['allow_abstentions'] else 'no ') + 'abstain' 
    return var[name]

def row_cols_non_empty_lists(row):
    for entry in row:
        if not (isinstance(entry, list) or isinstance(entry, np.ndarray)) or len(entry) < 1:
            return False
    return True
