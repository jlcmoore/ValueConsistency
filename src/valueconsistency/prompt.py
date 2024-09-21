#!env-valueconsistency/bin/python

'''

Author: Jared Moore

Run `prompt --help`

'''

import argparse
import copy
import datetime
import itertools
import json
import logging
import math
import os
import pandas as pd
import random
import re
import sys
import textwrap
import tqdm

from .query_models import (setup_api,
                            close_api,
                            openai_chat_with_backoff,
                            openai_completion_with_backoff,
                            anthropic_completion_with_backoff,
                            anthropic_chat_with_backoff,
                            classify_query_completions_model,
                            classify_query_chat_model,
                            generate_query_completions_model,
                            generate_query_chat_model,
                            OPTIONS,
                            CONFIDENCE_QUESTION)
from .utils import (OUTPUT_DIR,
                    dialogue_as_list,
                    dialogue_as_string,
                    report_tokens,
                    interleave,
                    save_run,
                    option_columns,
                    answer_columns,
                    max_stance,
                    hash_dict,
                    LANGUAGES,
                    VALUE_BY_LANGUAGE)

from .language_prompts import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

TASKS = ['classification', 'generation']

QUESTION_CLASSIFY = "{question}{value_statement}{context_statement}"

CONTEXT_STATEMENT = " {context}"

STATIC_VALUE = {
lang : {'vrd': VALUE_BY_LANGUAGE[lang],
        'vrd_english' : 'value',
        'text' :  VALUE_BY_LANGUAGE[lang],
        'text_english' : 'value',
        'label' : 'either'}
for lang in LANGUAGES
}

VERSION = ".07" # The version of the experimental protocol.

def model_generation(question, options, context_text, value_text,
                     query_func, dialogue_func, endpoint,
                     args):
    '''
    Queries `model` with a generated prompt for a generation setting. Returns a list of 
    dicts with the results.
    '''
    results = []
    options = copy.deepcopy(options)
    options += [None]

    for option in options:
        prompt = make_generation_dialogue(question, option,
                                          context_text, value_text,
                                          dialogue_func, query_language=args.query_language,
                                          use_system_prompt=args.system_prompt)
        response = query_func(prompt, temperature=args.temperature,
                              model=args.model, endpoint=endpoint,
                              count_tokens=args.count_tokens,
                              confidence=args.ask_confidence)
        if args.count_tokens:
            input_tokens, output_tokens = response
            result = {'input' : input_tokens, 'output' : output_tokens}
        else:
            answer, confidence, full = response
            result = {'answer' : answer,
                      'confidence' : confidence,
                      'full' : full,
                      'prompt' : prompt,
                      'bias' : option}

        results.append(result)

    return results

def model_classification(question, options, context_text, value_text,
                         query_func, dialogue_func, endpoint,
                         args):
    '''
    Queries `model` with a generated prompt for a classification setting. Returns a list of 
    dicts with the results.
    '''
    options = copy.deepcopy(options)
    if args.randomize_option_order:
        options = random.sample(options, len(options))

    # Dynamically, randomly order the options
    examples = []
    example_answer = None
    if hasattr(args, 'example_answer') and args.example_answer:
        examples, example_answer = make_few_shot_example(args.query_language, follow_up=args.follow_up,
                                         single_letter_prompt=args.single_letter_prompt)

    prompt = make_classification_dialogue(question, options, context_text, value_text,
                                          dialogue_func, query_language=args.query_language,
                                          examples=examples, follow_up=args.follow_up,
                                          use_system_prompt=args.system_prompt,
                                          single_letter_prompt=args.single_letter_prompt)
    opt_columns = option_columns(len(options))
    ans_columns = answer_columns(len(options))

    response = query_func(options, prompt, temperature=args.temperature,
                          model=args.model, endpoint=endpoint,
                          count_tokens=args.count_tokens,
                          confidence=args.ask_confidence)
      
    if args.count_tokens:
        input_tokens, output_tokens = response
        result = {'input' : input_tokens, 'output' : output_tokens}
    else:
        distribution, confidence, text, full, answer_logprobs = response
        answer = None
        if distribution:
            answers = ()
            for option in options:
                if option not in distribution:
                    distribution[option] = 0
                answers += (distribution[option],)

            tied = answers.count(max(answers)) > 1
            if not tied:
                answer = max(distribution, key=distribution.get)

        
        result = {'answer' : answer,
                  'confidence' : confidence,
                  'text' : text,
                  'full' : full,
                  'prompt' : prompt,
                  'answer_logprobs' : answer_logprobs,
                  'example_answer' : example_answer}

        result.update({k : v for k, v in zip(opt_columns, options)})
        result.update({k : v for k, v in zip(ans_columns, answers)})

    return [result]

def make_classification_dialogue(question, options, context_text, value_text,
                                 dialogue_func, query_language='english', examples=[],
                                 follow_up=False, use_system_prompt=True,
                                 single_letter_prompt=True):
    messages = []
    if use_system_prompt:
        system_prompt = SYSTEM[query_language]
        messages.append(('system', system_prompt))
    messages += examples
    result = messages

    full_question = QUESTION_CLASSIFY.format(question=question,
                                             context_statement=context_text,
                                             value_statement=value_text)

    prompt = options_text(full_question, options, query_language, single_letter_prompt)
    result += [('user', prompt)]

    if follow_up:
        result += [('assistant', FOLLOW_UP_COMPLETIONS[query_language])]
    result = dialogue_func(result, last_message=True)
    return result

def options_text(question, options, query_language='english', single_letter_prompt=False):
    # NB: Putting the response of 'A' or 'B' closer to the end to deal with the
    # 'cognitive load' point raised by Jennifer Hu from Harvard.
    prompt = ""
    if single_letter_prompt:
        prompt = SINGLE_LETTER[query_language] + '\n\n'

    prompt += question.strip() + '\n'

    for letter, option in zip(OPTIONS[0:len(options)], options):
        prompt += f'- ({letter.upper()}): {option}\n'

    return prompt.strip()

def make_few_shot_example(query_language, follow_up=False, single_letter_prompt=True):
    '''
    For the given `query_language` returns a few shot example prompt to show
    how to answer a question with a single-letter answer. Randomly orders the
    options.
    '''
    answers = list(FEW_SHOT_EXAMPLE_ANSWER[query_language]['answers'])
    correct = FEW_SHOT_EXAMPLE_ANSWER[query_language]['correct']
    random.shuffle(answers)
    answer_option = OPTIONS[answers.index(correct)]

    prompt = options_text(FEW_SHOT_EXAMPLE_QUESTION[query_language], answers,
                          query_language, single_letter_prompt)
    messages = [('user', prompt)]
    answer =  answer_option
    if follow_up:
        answer = FOLLOW_UP_COMPLETIONS[query_language] + answer + ')'
    return (messages + [('assistant', answer)], answer_option)

def make_generation_dialogue(question, answer, context_text, value_text,
                             dialogue_func, query_language='english',
                             examples=[], use_system_prompt=True):
    messages = []
    if use_system_prompt:
        system_prompt = SYSTEM[query_language]
        messages.append(('system', system_prompt))
    messages += examples
    result = dialogue_func(messages)

    if answer is not None:
        prompt = QUESTION_ANS_GENERATE[query_language].format(question=question,
                                              answer=answer, 
                                              value_statement=value_text,
                                              context_statement=context_text)
        # A prompt to bias toward a particular answer
    else:
        prompt = QUESTION_GENERATE[query_language].format(question=question,
                                          value_statement=value_text,
                                          context_statement=context_text)
        # The neutral prompt; the one we want to discover if it has bias toward an answer
    result += dialogue_func([('user', prompt)], last_message=True)

    return result

def contexts_to_text(contexts):
    result = []
    for context in contexts:
        context_text = ""
        if context is not None:
            context_text = CONTEXT_STATEMENT.format(context=context['text'])
        result.append({'context' : context, 'context_text' : context_text})
    return result

def value_to_text(value, query_language):
    if value is None:
        return ""
    text = VALUE_PART[query_language].format(vrd=value['vrd'].lower(),
                                             text=value['text'].lower())
    statement = VALUE_STATEMENT[query_language].format(value_text=text)
    return statement

def values_to_text(values, num_values, use_most_ambiguous, query_language):
    vals = pd.DataFrame(values)
    if 'label' not in vals.columns:
        vals['label'] = vals.apply(max_stance, axis=1)

    if 'either' in vals.columns:
        most_ambiguous_idx = vals.sort_values(by=['either'], ascending=True).index[0]
        most_ambiguous = vals.iloc[most_ambiguous_idx].to_dict()
        vals.drop(most_ambiguous_idx)

        support = vals[vals['label'] == 'supports'].sort_values(by=['relevant'],
                                                                ascending=False)\
                                                   .to_dict(orient='records')
        oppose = vals[vals['label'] == 'opposes'].sort_values(by=['relevant'],
                                                              ascending=False)\
                                                 .to_dict(orient='records')
        # TODO: this approach currently discards neutral values. Consider changing.
        # If we are keeping only a few values, we want to keep the most divergent ones.
        values = interleave(support, oppose, num_values)
    else:
        sample_size = num_values if num_values > 0 else len(values)
        values = random.sample(values, sample_size)
    result = []

    if len(values) < num_values:
        logging.warn(f"Only generated {len(values)} values, {values}")

    if use_most_ambiguous:
        values.append(most_ambiguous)

    for value in values:
        value_text = value_to_text(value, query_language)
        result.append({'value' : value, 'value_text' : value_text})

    return result

def main():

    parser = argparse.ArgumentParser(
                    prog='prompt',
                    description='Gauges the values of various LLMs.')
    parser.add_argument('--filename', required=True, help="The preference file to test on.")
    parser.add_argument('--input_directory', default="", help="Where to look for `filename`")
    parser.add_argument('--output-directory', default=OUTPUT_DIR, help="Where to output the results.")

    parser.add_argument('--sample-size', default=None, type=int,
        help="The number of samples to take from the combinations of the items in `filename`.")

    ########### Robustness checks
    parser.add_argument('--samples', default=1, type=int, required=False, 
        help=("The number of times to ask the model each individual example. The default " +
        "is once, but asking the same question multiple times (when the temperature is " +
        "greater than zero) approximates getting the token probabilities from the model " +
        "(which are often not available)."))

    parser.add_argument('--randomize-option-order', default=False, action="store_true",
        help=("Randomizes the order of options in each sample regardless of other settings."))
    ###########

    ########### Independent variables (treatments)
    parser.add_argument('--task', choices=TASKS, required=True,
        help="Whether to pose the task as a multiple choice question or as a generation task.")

    parser.add_argument('--no-rephrase', default=False, action="store_true",
        help=("Does not ask rephrased questions."))

    parser.add_argument('--single-question', default=False, action="store_true",
        help=("Asks only a single question per topic."))
    
    parser.add_argument('--temperature', default=0, type=float,
        help="The temperature at which to query the model.")

    ###########

    ########### Model arguments and set up
    parser.add_argument('--endpoint', choices=['chat', 'completion'], default='chat',
        help=("Models are either chat or completion based. Querying a model at the wrong " +
        "endpoint will result in an error."))
    parser.add_argument('--model', required=True, help="The model to query.")
    parser.add_argument('--query-language', choices=LANGUAGES.keys(), default='english',
        help=("The language to query the models in."))

    parser.add_argument('--allow-abstentions', default=False, action='store_true', help=
        "Whether to allow a 'none of the above' response.")

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

    parser.add_argument('--single-letter-prompt', action='store_true', 
        dest='single_letter_prompt', help="Whether to append the single letter prompt.")
    parser.add_argument('--no-single-letter-prompt', dest='single_letter_prompt', action='store_false')
    parser.set_defaults(single_letter_prompt=True)

    parser.add_argument('--example-answer', default=False, action='store_true', help=
        "Whether to encourage the model in classification tasks to respond in a single " +
        "option letter by an example doing so to the context.")

    parser.add_argument('--count-tokens', default=False, action='store_true', help=
        "Count the max number of tokens when using OpenAI models. Does not query models. Writes to stdout.")

    parser.add_argument('--ask-confidence', default=False, action='store_true', help=
        "Ask the model for its confidence in each answered question. Adds many tokens")

    parser.add_argument('--use-context', default=False, action='store_true', help=
        "Whether to prompt with context for each answer (multiplies queries by number of answers). \
         See `--use-values`.")

    parser.add_argument('--use-values', default=False, action='store_true', help=
        "Whether to prompt with values for each answer (multiplies queries by number of values). \
         Presently does not take the intersection when also called with `use-context`.")

    parser.add_argument('--num-values', default=2, type=int, help="The number of values from \
        the data to query with if `--use-values` is on. -1 means all values.")

    parser.add_argument('--most-ambiguous-value', default=False, action='store_true', help=
        "When prompting with values whether to include the most ambiguous value. \
        Unless `--num-values` is the size of the values, will increase `--num-values` by 1")

    parser.add_argument('--unrelated-value', default=False, action='store_true', help=
        "When prompting with values whether to include the unrelated value. \
        Unless `--num-values` is the size of the values, will increase `--num-values` by 1")

    parser.add_argument('--static-value', default=True, action='store_true', help=
        "When prompting with values whether to include the unrelated value. \
        Unless `--num-values` is the size of the values, will increase `--num-values` by 1")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source', choices=['openai', 'anthropic', 'vllm'], help="which source to query")
    group.add_argument('--origin', help=("The address to use when using the openai api " +
        "(e.g. with llm-lab)."))
    ###########

    args = parser.parse_args()

    if args.count_tokens and args.source != 'openai':
        raise ValueError("Token counting only available for OpenAI models")

    if 'mistral' in args.model and args.follow_up:
        logging.warn("Mistral models require alternating dialogue and no system prompts.")
        logging.warn("\tSetting follow_up and system_prompt to false")
        args.follow_up = False
        args.system_prompt = False

    try:
        setup_api(source=args.source, origin=args.origin, model=args.model)

        ########### Setting up the relevant functions with which to query the model
        if args.endpoint == 'chat':
            dialogue_func = dialogue_as_list
        else:
            dialogue_func = dialogue_as_string

        if args.task == 'classification':
            if args.endpoint == 'chat':
                query_func = classify_query_chat_model
            else:
                query_func = classify_query_completions_model
            task_function = model_classification
        else:
            if args.endpoint == 'chat':
                query_func = generate_query_chat_model
            else:
                query_func = generate_query_completions_model
            task_function = model_generation

        if args.source and args.source == 'anthropic':
            if args.endpoint == 'chat':
                endpoint = anthropic_chat_with_backoff
            else:
                endpoint = anthropic_completion_with_backoff
        else:
            if args.endpoint == 'chat':
                endpoint = openai_chat_with_backoff
            else:
                endpoint = openai_completion_with_backoff
        ###########

        ########### Loading data
        filename = os.path.join(args.input_directory, args.filename)
        logger.info(f'Reading from {filename}')
        
        df = pd.read_json(filename, lines=True)

        if args.sample_size is None or args.sample_size > len(df):
            args.sample_size = len(df)

        if args.use_values and 'values' not in df.columns:
            raise ValueError("Data file should have a values column.")

        if args.use_context and 'contexts' not in df.columns:
            raise ValueError("Data file should have a contexts column.")


        filtered = df
        
        filtered = filtered.iloc[0: args.sample_size]
        ###########

        ########### Querying the model
        logger.info(f'Querying {args.model}')
        all_results = []

        if args.no_rephrase:
            filtered = filtered[~filtered['rephrase']]

        if args.single_question:
            filtered = filtered.groupby(['topic']).apply(lambda x: x.iloc[0])

        for _, row in tqdm.tqdm(filtered.iterrows(), total=len(filtered)):

            # Extracting the relevant information from each row
            results = []

            question = row['question']
            if args.allow_abstentions:
                row['options'][ABSTAIN_ANSWER[args.query_language]] = 'neutral'
            options = list(row['options'].keys())

            #  NB: We only add other values if this question is an original question
            #  NB: This will fail in the modality case but we should probably not query
            #      for values in the modality case anyway.
            first_of_rephrased = True
            if 'rephrase' in row:
                first_of_rephrased = not row['rephrase']

            value_context_args = [{},]
            if args.use_context and first_of_rephrased:
                value_context_args += contexts_to_text(list(row['contexts']))

            if args.use_values and first_of_rephrased:
                value_context_args += values_to_text(row['values'],
                                      args.num_values, args.most_ambiguous_value,
                                      args.query_language)
                if args.unrelated_value:
                    row['unrelated_value']['label'] = max_stance(row['unrelated_value'])
                    unrelated_text = value_to_text(row['unrelated_value'], args.query_language)
                    value_context_args.append({'value' : row['unrelated_value'],
                                               'value_text' : unrelated_text})
                if args.static_value:
                    # TODO have to change the value baggage from kaleido..., esp. in analysis
                    static_text = value_to_text(STATIC_VALUE[args.query_language],  args.query_language)
                    value_context_args.append({'value' : STATIC_VALUE[args.query_language],
                                               'value_text' : static_text})

            num_samples = args.samples if hasattr(args, 'samples') else 1
            for i in range(num_samples):
                for v_c_arg in value_context_args:
                    value = None
                    context = None
                    context_text = ""
                    value_text = ""
                    if 'value' in v_c_arg:
                        value = v_c_arg['value']
                        value_text = v_c_arg['value_text']
                    if 'context' in v_c_arg:
                        context = v_c_arg['context']
                        context_text = v_c_arg['context_text']

                    # Query the model based on question type
                    results = task_function(question, options, context_text, value_text,
                                            query_func, dialogue_func,
                                            endpoint, args)
                    for result in results:
                        result.update(row) # Add in all of the information
                                           # from the input file
                        result.update({'sample' : i,
                                       'value' : value,
                                       'value_hash' : hash_dict(value),
                                       'context' : context,
                                       'context_hash' : hash_dict(context),})
                        all_results.append(result)

        results_df = pd.DataFrame(all_results)
        ###########

        if args.count_tokens:
            report_tokens(results_df['input'].sum(),
                          results_df['output'].sum(),
                          args.model)
            return

        ########### Formatting the data for output
        conditions = []
        if args.allow_abstentions:
            conditions += ["allow-abstentions"]
        condition_str = "_".join(conditions)

        base_filename = os.path.basename(args.filename)
        run = os.path.splitext(base_filename)[0]
        run_dir = os.path.join(args.output_directory, run)
        iso = datetime.datetime.now().date().isoformat()

        model_name_safe = args.model.replace('/', '_')
        model_dir = os.path.join(run_dir, model_name_safe)
        output = args.task
        if len(condition_str) > 0:
            output += f"_{condition_str}"
        output += f"-{iso}.json"

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        output = os.path.join(model_dir, output)

        variables = dict(vars(args))
        variables['filename'] = output
        variables['version'] = VERSION
        logger.info(f'Outputting to {os.path.abspath(output)}')
        save_run(variables, results_df)
        ###########
    finally:
        # NB: consider saving a partially completed job here
        close_api()

if __name__ == '__main__':
    main()
