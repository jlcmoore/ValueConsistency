import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import random
import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from .utils import option_columns, answer_columns

from .query_models import (setup_api,
                           close_api,
                           openai_chat_with_backoff,
                           anthropic_chat_with_backoff,
                           classify_query_chat_model)

from .utils import (load_run,
                    dialogue_as_list,
                    report_tokens,
                    OUTPUT_DIR,
                    hash_dict,
                    reverse_dict,
                    LANGUAGES,
                    MODEL_NAMES_SHORT)

from .prompt import (make_classification_dialogue,
                     model_classification,
                     classify_query_chat_model)

from .language_prompts import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

def judge_stance(df, endpoint, args):
    '''
    Judge the stance of each passage
    '''
    all_results = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):

        if args.allow_abstentions:
            row['options'][ABSTAIN_ANSWER[args.query_language]] = 'neutral'
        options = list(row['options'].keys())

        compare_question = STANCE_PROMPT[args.query_language].format(passage=row['answer'].strip(),
                                                                     question=row['question'])

        # Add in the relevant previous info, but not the irrelevant info
        remove_keys = ['answer', 'confidence', 'full', 'prompt', 'distribution']
        old_info = row.to_dict()
        new_info = {key: old_info[key] for key in old_info.keys() - remove_keys}

        results = model_classification(compare_question, options=options,
                                        query_func=classify_query_chat_model,
                                        context_text="", value_text="",
                                        dialogue_func=dialogue_as_list,
                                        endpoint=endpoint, args=args)

        for result in results:
            result['annotator_question'] = compare_question
            result.update(new_info) # Add in all of the information 
                                    # from the input file

            all_results.append(result)

    return pd.DataFrame(all_results)

def judge(groups, endpoint, args):
    all_results = []

    for _, group in tqdm.tqdm(groups, total=len(groups)):

        bias_to_p = {row['bias'] : row['answer'] for row in group.to_dict(orient='records')}

        biases = group['bias'].to_list()

        non_neutral_biases = list(set(biases) - set([None]))

        new_options = [bias_to_p[bias] for bias in non_neutral_biases]
        random.shuffle(new_options) # TBD whether we want this.
        
        # Getting rid of the 'no option' answer for now but might want to bring it back.
        if args.allow_abstentions:
            new_options.append(NONE_OPTION[args.query_language])

        old_info = group.iloc[0].to_dict()
        
        passage_to_stance = {bias_to_p[bias] : old_info['options'][bias] for bias in non_neutral_biases}

        compare_question = COMPARE_PROMPT[args.query_language].format(target=bias_to_p[None]),

        row = {'passage_to_option' : reverse_dict(bias_to_p),
               'options' : passage_to_stance}

        if len(row['passage_to_option']) < 3:
            # There was a hash conflict; the same passage was written twice
            logger.error("Same passage written twice.")

        # Add in the relevant previous info, but not the irrelevant info
        remove_keys = ['answer', 'options', 'confidence', 'full', 'prompt', 'bias']
        row.update({key: old_info[key] for key in old_info.keys() - remove_keys})

        results = model_classification(compare_question, options=new_options,
                                        query_func=classify_query_chat_model,
                                        context_text="", value_text="",
                                        dialogue_func=dialogue_as_list,
                                        endpoint=endpoint, args=args)

        for result in results:
            row['annotator_question'] = compare_question
            result.update(row) # Add in all of the information 
                               # from the input file


            all_results.append(result)

    return pd.DataFrame(all_results)

def pca_judge(target, options, embedding_model):
    pca = PCA(n_components=1)

    fit_embeddings =  embedding_model.encode(options)
    fit_embeddings = fit_embeddings / np.linalg.norm(fit_embeddings, axis=1, keepdims=True)
    pca.fit(fit_embeddings)
    
    target_raw = embedding_model.encode([target])
    target_raw = target_raw / np.linalg.norm(target_raw)

    embeddings_transformed = pca.transform(target_raw)
    assert len(embeddings_transformed) == 1 and len(embeddings_transformed[0]) == 1
    embeddings_transformed = embeddings_transformed[0][0]
    prob = (embeddings_transformed + 1)/2
    
    if prob > 0.5:
        answer = options[0]
    else:
        answer = options[1]
    answers = [prob, 1-prob]

    result = {'answer' : answer}
    opt_columns = option_columns(len(options))
    ans_columns = answer_columns(len(options))

    result.update({k : v for k, v in zip(opt_columns, options)})
    result.update({k : v for k, v in zip(ans_columns, answers)})

    return [result]

def pca_approach(groups, args):
    all_results = []
    embedding_model_name='google/flan-t5-xl' 
    #embedding_model_name='all-MiniLM-L6-v2'
    
    embedding_model = SentenceTransformer(embedding_model_name)
    for _, group in tqdm.tqdm(groups, total=len(groups)):

        bias_to_p = {row['bias'] : row['answer'] for row in group.to_dict(orient='records')}

        biases = group['bias'].to_list()

        non_neutral_biases = list(set(biases) - set([None]))

        new_options = [bias_to_p[bias] for bias in non_neutral_biases]
        random.shuffle(new_options) # TBD whether we want this.
        
        # Getting rid of the 'no option' answer for now but might want to bring it back.
        if args.allow_abstentions:
            new_options.append(NONE_OPTION[args.query_language])

        old_info = group.iloc[0].to_dict()
        
        passage_to_stance = {bias_to_p[bias] : old_info['options'][bias] for bias in non_neutral_biases}

        compare_question = COMPARE_PROMPT[args.query_language].format(target=bias_to_p[None]),

        row = {'passage_to_option' : reverse_dict(bias_to_p),
               'options' : passage_to_stance}

        if len(row['passage_to_option']) < 3:
            # There was a hash conflict; the same passage was written twice
            logger.error("Same passage written twice.")

        # Add in the relevant previous info, but not the irrelevant info
        remove_keys = ['answer', 'options', 'confidence', 'full', 'prompt', 'bias']
        row.update({key: old_info[key] for key in old_info.keys() - remove_keys})

        results = pca_judge(bias_to_p[None], new_options, embedding_model)

        for result in results:
            result.update(row) 

            all_results.append(result)

    return pd.DataFrame(all_results)

def judge_main(args):
    ########### Loading data
    filename = os.path.join(args.input_directory, args.filename)
    logger.info(f'Reading from {filename}')
    
    variables, df = load_run(filename)

    args.query_language = variables['query_language']
    ###########

    if args.source == "anthropic":
        endpoint = anthropic_chat_with_backoff
    else:
        endpoint = openai_chat_with_backoff

    ########### Querying the model
    logger.info(f'Querying {args.model}')

    if args.sample_size is None or args.sample_size > len(df):
        args.sample_size = len(df)
    df = df.iloc[0: args.sample_size]

    if args.pca_approach: 
        groups = df.groupby(['sample', 'question', 'context_hash', 'value_hash'])
        results_df = pca_approach(groups, args)
    else:
        try:
            setup_api(source=args.source, origin=None, model=args.model)

            if args.stance: 
                results_df = judge_stance(df, endpoint, args)

            else:
                groups = df.groupby(['sample', 'question', 'context_hash', 'value_hash'])

                results_df = judge(groups, endpoint, args)

        ###########

            if args.count_tokens:
                report_tokens(results_df['input'].sum(),
                                  results_df['output'].sum(),
                                  args.model)
                return

        finally:
            close_api()

    ########### Formatting the data for output
    output_json = {}
    variables['annotator'] = args.model
    variables['annotator_temperature'] = args.temperature
    variables['annotator_abstentions'] = args.allow_abstentions
    variables['stance'] = args.stance
    output_json['variables'] = variables

    base_filename = os.path.basename(args.filename)
    run = os.path.splitext(base_filename)[0]

    run += f'_annotated'

    if args.pca_approach:
        run += '_pca'
    elif args.stance:
        run += '_stance'

    if args.allow_abstentions:
        run += '_abstain'

    run += f"_{MODEL_NAMES_SHORT[args.model]}"
    
    run += '.json'

    one_dir_up = os.path.dirname(args.filename)
    model_dir = os.path.basename(one_dir_up)

    input_filename_dir = os.path.basename(os.path.dirname(one_dir_up))

    output_dir = os.path.join(os.path.join(args.output_directory, input_filename_dir),
                              model_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output = os.path.join(output_dir, run)

    logger.info(f'Outputting to {os.path.abspath(output)}')
    output_json['data'] = results_df.to_json(orient='records', lines=True)

    with open(output, 'w') as f:
        f.write(json.dumps(output_json, indent=True))

def main():

    # TODO: these arguements are quite redundant with `prompt`

    parser = argparse.ArgumentParser(
                    prog='judge',
                    description='Judges the similarity between various generated responses.')
    parser.add_argument('--filename', required=True, help="The file to test on.")
    parser.add_argument('--input_directory', default="", help="Where to look for `filename`")
    parser.add_argument('--output-directory', default=OUTPUT_DIR, help="Where to output the results.")

    parser.add_argument('--allow-abstentions', default=False, action='store_true', help=
        "Whether to allow a 'none of the above' response.")

    parser.add_argument('--pca-approach', required=False, default=False, type=bool, help="Whether to use the baseline PCA approach.")

    parser.add_argument('--temperature', default=0, type=float,
        help="The temperature at which to query the model.")

    parser.add_argument('--sample-size', default=None, type=int,
        help="The number of samples to take from the combinations of the items in `filename`.")    

    parser.add_argument('--example-answer', default=False, action='store_true', help=
        "Whether to encourage the model in classification tasks to respond in a single " +
        "option letter by an example doing so to the context.")

    parser.add_argument('--follow-up', action='store_true', help=
        "Whether to encourage the model in classification tasks to respond in a single " +
        "option letter by appending the beginning of their answer. If not set, adds " + 
        "explicit instruction to answer with one letter.")
    parser.add_argument('--no-follow-up', dest='follow_up', action='store_false')
    parser.set_defaults(follow_up=True)

    parser.add_argument('--system-prompt', action='store_true', help=
        "Whether use a system prompt")
    parser.add_argument('--no-system-prompt', dest='system_prompt', action='store_false')
    parser.set_defaults(system_prompt=True)

    parser.add_argument('--stance', action='store_true', help=
        "Whether to prompt for the stance of each paragraph as opposed to grouping them.")
    parser.add_argument('--no-stance', dest='stance', action='store_false')
    parser.set_defaults(stance=False)

    parser.add_argument('--single-letter-prompt', action='store_true', 
        dest='single_letter_prompt', help="Whether to append the single letter prompt.")
    parser.add_argument('--no-single-letter-prompt', dest='single_letter_prompt', action='store_false')
    parser.set_defaults(single_letter_prompt=True)

    parser.add_argument('--randomize-option-order', default=False, action="store_true",
        help=("Randomizes the order of options in each sample regardless of other settings."))

    parser.add_argument('--ask-confidence', default=False, action='store_true', help=
        "Ask the model for its confidence in each answered question.")
    
    ########### Model arguments and set up
    parser.add_argument('--model', required=False,
        help="The model to query.")

    parser.add_argument('--source', choices=['openai', 'vllm', 'anthropic'], required=False, help="which source to query")
    
    parser.add_argument('--count-tokens', default=False, action='store_true', help=
        "Count the max number of tokens when using OpenAI models. Does not query models. Writes to stdout.")
    ###########

    args = parser.parse_args()

    if args.model and 'mistral' in args.model and args.follow_up:
        logging.warn("Mistral models require alternating dialogue or system prompts.")
        logging.warn("\tSetting follow_up and system_prompt to false")
        args.follow_up = False
        args.system_prompt = False

    if args.stance and args.pca_approach:
        raise ValueError("Mutually exclusive options")

    judge_main(args)
