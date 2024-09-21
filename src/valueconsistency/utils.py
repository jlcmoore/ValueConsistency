import anthropic as anth
from collections import OrderedDict
import copy
import io
import json
import numpy as np
import os
import pandas as pd
import tiktoken

from .distribution import Distribution

from .language_prompts import *

MODEL_NAMES_SHORT = {
    '01-ai/Yi-34B-Chat' : 'yi',
    '01-ai/Yi-34B' : 'yi-base',
    'meta-llama/Llama-2-70b-chat-hf' : 'llama2',
    'meta-llama/Llama-2-70b-hf' : 'llama2-base',
    'meta-llama/Meta-Llama-3-70B-Instruct' : 'llama3',
    'meta-llama/Meta-Llama-3-70B' : 'llama3-base',
    'meta-llama/Llama-2-7b-chat-hf' : 'llama2-7b',
    'meta-llama/Llama-2-7b-hf' : 'llama2-7b-base',
    'meta-llama/Meta-Llama-3-8B-Instruct' : 'llama3-8b',
    'meta-llama/Meta-Llama-3-8B' : 'llama3-8b-base',    
    'mistralai/Mixtral-8x7B-Instruct-v0.1' : 'mixtral-8x7b',
    'davinci-002' : 'davinci-002',
    'gpt-3.5-turbo' : 'gpt-3.5',
    'gpt-3.5-turbo-0613' : 'gpt-3.5',
    'gpt-4' : 'gpt-4',
    'gpt-4o' : 'gpt-4o',
    'gpt-4-0613' : 'gpt-4',
    'gpt-4-0125-preview' : 'gpt-4-turbo',
    'claude-3-opus-20240229' : 'claude-3-opus',
    'stabilityai/japanese-stablelm-instruct-beta-70b' : 'stability',
    'CohereForAI/c4ai-command-r-v01' : 'cmd-r',
}

OUTPUT_DIR = 'results'

SCHWARTZ_VALUES_BY_LANGUAGE = {}
for l in LANGUAGES.keys():
    SCHWARTZ_VALUES_BY_LANGUAGE[l] = [v[l] for v in SCHWARTZ_VALUES.values()]

SCHWARTZ_VALUES_DICT_FMT = {
    lang : [{'vrd' : VALUE_BY_LANGUAGE[lang],
             'vrd_english' : 'value', 'text' : v[lang],
             'text_english' : v['english'], 'label' : None}
                for v in SCHWARTZ_VALUES.values()] 
        for lang in LANGUAGES.keys()
}

########### Common utilities

def reverse_dict(dictionary):
    return {value: key for key, value in dictionary.items()}

def get_max_no_ties(d):
    vals = list(d.values())
    tied = vals.count(max(vals)) > 1

    answer = None
    if not tied:
        answer = max(d, key=d.get)
    return answer

def interleave(a, b, num):
    '''
    Interleave two lists a and b. The lists do not have to be the same length.
    Num indicates how many total items to have in the new list.
    If one list of a or b is too short, fills the remaining values with the values of the other list.
    If Num is -1, fully interleaves both lists.
    '''
    result = []
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    while (num == -1 or len(result) < num) and (len(a) > 0 or len(b) > 0):
        if len(a) > 0:
            result.append(a.pop(0))
        if len(b) > 0 and (num == -1 or len(result) < num):
            result.append(b.pop(0))
    return result

def hash_dict(d):
    return 0 if d is None or pd.isnull(d) else hash(frozenset(d.items()))

###########

########### For operating over Kaleido values

def max_stance(value):
    if isinstance(value, pd.Series):
        value = value.to_dict()
    items = filter(lambda x: x[0] in ['supports', 'opposes', 'either'], list(value.items()))
    stance = {k : v for k, v in items}
    return max(stance, key=stance.get)

###########

########### Results files loading

VAIDATE_PARAPHRASES = True

def validate_paraphrases(name, df):
    if VAIDATE_PARAPHRASES:
        validated_filename = f'data/validate/paraphrase/{name}.csv'
        if os.path.exists(validated_filename):
            validated_df = pd.read_csv(validated_filename)
            validated = validated_df[validated_df['questions_equivalent'] == 'yes']['original']
            df = df[df['original'].isin(validated)]
    return df

def save_run(variables, df):
    output_json = {}
    output_json['data'] = df.to_json(orient='records', lines=True)
    output_json['variables'] = variables

    output = variables['filename']
    with open(output, 'w') as f:
        f.write(json.dumps(output_json, indent=True))

def load_run(run, other_stances=False):
    '''Returns the variables and the df for this run'''
    # have to load it as json at first b/c of the arguments at the top level
    with open(run, 'r') as f:
        dump = json.load(f)
    df = pd.read_json(io.StringIO(dump['data']), lines=True)
    if len(df[df['question'].isnull()]) > 0:
        import pdb; pdb.set_trace()
    variables = dump['variables']
    if 'filename' not in run:
        variables['filename'] = run
    if 'annotator_abstentions' in variables and variables['annotator_abstentions'] and len(df['options'].iloc[0]) < 3:
        df['options'] = df['options'].apply(lambda x: {**x, **{NONE_OPTION[variables['query_language']] : 'neutral'}})
    if 'stance' in variables and variables['stance']:
        df['stance_bias'] = df.apply(lambda x: x['options'][x['bias']] if x['bias'] in x['options'] else None, axis=1)
        if not other_stances:
            df = df[df['stance_bias'].isnull()]

    # Remove from consideration any paraphrases which were later deemed invalid
    name = os.path.basename(os.path.dirname(os.path.dirname(variables['filename'])))
    df = validate_paraphrases(name, df)

    df = add_distribution_to_classificaiton(df)
    return (variables, df)

def add_distribution_to_classificaiton(df):
    # Equivalent to `variables['task'] == 'classification' or 'annotator' in variables`
    if len(get_opt_cols(df)) > 0:
        df['distribution'] = df.apply(row_to_distribution,
                                       as_stance=True,
                                       uniform_prior=True,
                                       axis=1)
        opposite_stances = df['options'].apply(lambda x: 'supports' in x.values() and 'opposes' in x.values())
        df = df[opposite_stances]
    return df

def get_data(filename, desired_variables):
    results = {} # a list of df, variables pairs by model
    file_short = os.path.splitext(os.path.basename(filename))[0]
    data_dir = os.path.join(OUTPUT_DIR, file_short)
    if not os.path.isdir(data_dir):
        return {}
    models = os.listdir(data_dir)
    for model in models:
        model_name = None
        if "DS_Store" in model:
            continue
        model_dir = os.path.join(data_dir, model)
        model_results = []
        for run_rel in os.listdir(model_dir):
            # Ignore unannotated generations
            if "DS_Store" in run_rel or ('generation' in run_rel and 'annotated' not in run_rel):
                continue
            run = os.path.join(model_dir, run_rel)
            variables, df = load_run(run)
            if model_name is None:
                model_name = variables['model']
            for variable, value in desired_variables.items():
                if (variable not in variables
                        or (isinstance(value, list) and variables[variable] not in value)
                        or (not isinstance(value, list) and variables[variable] != value)):
                    break
            else:
                model_results.append(load_run(run))
        if len(model_results) > 0:
            results[model_name] = model_results

    if ('model' in desired_variables and isinstance(desired_variables['model'], list) and
        len(desired_variables['model']) > 1):
        results = OrderedDict(sorted(results.items(), key=lambda i: desired_variables['model'].index(i[0])))

    return results


def group_tasks_by_run(all_dfs, group=True):
    '''Given a list of (variables, dataframe) tuples (should be for the same model),
    returns a list of paired tasks, i.e. two tuples of ((variables, dataframe), (var, df))
    for the classification and generation tasks'''
    task_agnostic = {}
    for var, df in all_dfs:
        cpy = copy.deepcopy(var)
        if cpy['task'] == 'generation':
            if 'annotator' in cpy:
                del cpy['annotator']
                del cpy['annotator-temperature']
            else:
                continue # don't join with the non annotated versions
        del cpy['task']
        del cpy['filename']
        key = frozenset(cpy.items())
        if key not in task_agnostic:
            task_agnostic[key] = ()
        task_agnostic[key] += ((var, df),)

    if group:
        for key in list(task_agnostic.keys()):
            if len(task_agnostic[key]) != 2: # if more tasks, change this number
                del task_agnostic[key]
    return list(task_agnostic.values())

def get_matching_data(df_list, desired_variables, only_annotated=True):
    '''
    For the given list of (variables, dataframes) returns the first matching
    tuple accoring to the dict, desired_variables
    '''
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

###########

########### Common Dataframe Operations

OPT = 'option' # The name of the options columns, e.g. "option 1"
OPT_FMT = OPT + ' {0}' # The formatting of the option columns
OPT_REGEX = rf'{OPT} \d'
OPT_PROB_STR = f'p({OPT}=' # The prefix of the probability columns
OPT_PROB_FMT = OPT_PROB_STR + '{0})' # The suffix with the options number

def get_col_opt_index(col):
    # Assumes the indices only go up to 9
    return col.split(OPT_PROB_STR)[1][0]

def option_columns(num_columns):
    return tuple(OPT_FMT.format(i) for i in range(num_columns))

def answer_columns(num_columns):
    return tuple(OPT_PROB_FMT.format(i) for i in range(num_columns))

def get_opt_cols(df):
    if isinstance(df, pd.DataFrame):
        columns = df.columns
    else:
        columns = df.keys()
    return columns[columns.str.match(OPT_REGEX)]

def get_prob_cols(df):
    opts = get_opt_cols(df)
    return [f'p(option={0})'.format(opt) for opt in opts]

def reported_confidence(group):
    return group[~group['confidence'].isnull()]['confidence'].mean()

def inferred_confidence(group):
    # avg(probability of max answer - probability of next answer)
    confidences = []
    for i, row in group.iterrows():
        dist = row_to_distribution(row)
        vals = sorted(dist.values(), reverse=True)
        if len(vals) > 1:
            confidences.append(vals[0] - vals[1])
    return np.mean(np.array(confidences))

def inferred_confidence_no_avg(row):
    dist = row_to_distribution(row)
    vals = sorted(dist.values(), reverse=True)
    if len(vals) > 0:
        return vals[0]
    else:
        return 0

def no_context_value(df):
    '''Filters all rows with existing values and contexts out of the dataframe'''
    if 'context' in df.columns and 'value' in df.columns:
        return df[df['context'].isnull() & df['value'].isnull()]
    return df

def row_to_distribution(row, as_stance=False, uniform_prior=False):
    '''
    Extracts the distribution which is across different columns in `row`.
    The keys are the `option_columns` and the values are the `answer_columns`
    Returns a normalized Distribution representing that.
    If `as_stance` will return the stances mapped to the probabilities.
    If `uniform_prior` will treat distributions which sum to zero instsead
    as uniform distributions, e.g. [0, 0] -> [.5, .5]
    '''
    distribution = Distribution()
    num_options = len(get_opt_cols(row))
    for option_col, answer_col in zip(option_columns(num_options),
                                      answer_columns(num_options)):
        key = row[option_col]
        prob = row[answer_col]
        if as_stance:
            key = row['options'][key]
            if key in distribution:
                # I think this will only happen when in a generation scheme a model
                # gives the same answer twice
                if key == 'supports':
                    key = 'opposes'
                elif key == 'opposes':
                    key = 'supports'
                elif key == 'neutral':
                    prob += distribution[key]
                else:
                    raise ValueError("Unsupported stances.")
        distribution[key] = prob

    return distribution.normalize(uniform_prior)

def option_language_yes(row, language):
    new_opts = {k.lower() : v for k, v in row['options'].items()}
    yes = (set(YES_LANGUAGE[language]) & set(new_opts.keys())).pop()
    return yes

def option_language_yes_stance(row, language):
    new_opts = {k.lower() : v for k, v in row['options'].items()}
    yes = (set(YES_LANGUAGE[language.lower()]) & set(new_opts.keys())).pop()
    return new_opts[yes]

def options_are_yes_no(row, language):
    assert isinstance(row['options'], dict)
    return len((set([k.lower() for k in row['options'].keys()])
                - set(YES_LANGUAGE[language.lower()])) 
                - set(NO_LANGUAGE[language.lower()])
                - set([ABSTAIN_ANSWER[language.lower()].lower()])) == 0

###########

########### Message formatting

def dialogue_as_list(dialogue, last_message=False):
    messages = []
    for role, content in dialogue:
        messages.append({'role' : role, 'content' : content})
    return messages

def dialogue_as_string(dialogue, last_message=False):
    # TODO: abstract dialogue out and into `context`
    prompt = ""
    last_role = None
    for role, content in dialogue:
        if role == 'assistant':
            role = anth.AI_PROMPT
        else:
            role = anth.HUMAN_PROMPT
        last_role = role

        prompt += f'{role} {content}'
    if last_message and last_role != anth.AI_PROMPT:
        prompt = prompt.strip() + anth.AI_PROMPT
    return prompt

###########

########### Token counting

# Will need to update these from here: https://openai.com/pricing
OPENAI_PRICES = {
 'gpt-4-0125-preview'     : {'input' : .01, 'output' : .03},
 'gpt-4-1106-preview'     : {'input' : .01, 'output' : .03},
 'gpt-4'                  : {'input' : .03, 'output' : .06},
 'gpt-4o'                 : {'input' : .005, 'output' : .015}, 
 'gpt-4-0613'             : {'input' : .03, 'output' : .06},
 'gpt-4-32k-0613'         : {'input' : .06, 'output' : .12},
 'gpt-3.5-turbo'          : {'input' : .001 , 'output' : .002},
 'gpt-3.5-turbo-0613'     : {'input' : .001 , 'output' : .002},
 'gpt-3.5-turbo-1106'     : {'input' : .001 , 'output' : .002},
 'gpt-3.5-turbo-instruct' : {'input' : .0015 , 'output' : .002},
 'davinci-002'            : {'input' : .002, 'output' : .002},
 'babbage-002'            : {'input' : .0004, 'output' : .0004}}

# The following two functions are from here:
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_string(string: str, model: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. \
            See https://github.com/openai/openai-python/blob/main/chatml.md for information \
            on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def tokens_to_cost(model, input_tokens, output_tokens):
    in_cost = OPENAI_PRICES[model]['input']
    out_cost = OPENAI_PRICES[model]['output']
    return input_tokens / 1000 * in_cost + output_tokens / 1000 * out_cost

def report_tokens(total_input, total_output, model):
    cost = tokens_to_cost(model, total_input, total_output)
    print(f"Total input tokens: {total_input}")
    print(f"Total output tokens: {total_output}")
    print(f"Total cost: ${cost}")

###########