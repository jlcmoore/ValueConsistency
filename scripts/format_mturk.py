import argparse
import base64
import csv
import json
import os

import pandas as pd

from valueconsistency import (ABSTAIN_ANSWER)

def base64_encode(var):
    as_json = json.dumps(var, ensure_ascii=False)
    return base64.b64encode(bytes(as_json, 'utf-8')).decode('utf8')

def base64_decode(encoded):
    json_text = base64.b64decode(bytes(encoded, 'utf-8')).decode('utf-8')
    return json.loads(json_text)

# Function to group every n rows
def group_n_rows(df, n):
    grouped_data = df.groupby(df.index // n).agg(lambda x: x.tolist())
    return grouped_data

def main():
    parser = argparse.ArgumentParser(
                    prog='prompt',
                    description='Gauges the values of various LLMs.')
    parser.add_argument('--filename', required=True, help="The data file to generate for.")
    parser.add_argument('--kind', required=True, choices=['topic', 'rephrase'],
                        help="Whether to make rephrase or topic data")
    parser.add_argument('--validate', default=False, action='store_true',
                        help="whether this is to be used for validating the questions themselves.")
    parser.add_argument('--all-questions', default=False, action='store_true',
                        help="If `kind` is 'rephrase' includes all, not just the first, questions")
    parser.add_argument('--allow-abstentions', default=False, action='store_true', help=
        "Whether to allow a 'none of the above' response.")

    query_language = 'english'
    args = parser.parse_args()

    abstain_option = ABSTAIN_ANSWER[query_language]

    df = pd.read_json(args.filename, lines=True)

    if len(df['question'].unique()) != len(df):
        raise ValueError("Data file does not have unique questions")

    if len(set(df['original']) - set(df['question'])) > 0:
        raise ValueError("Originals are not fully contained in questions")

    def make_question_lists(group, validate=False):
        if validate:
            result = {'questions' : group['question'].to_list()}
            if isinstance(group.name, tuple) and len(group.name) > 1: # grouped by topic, original
                _, original = group.name
                result['original'] = original
            return result
        else:
            return group[['question', 'options']].to_dict(orient='records')

    if args.allow_abstentions:
        df['options'] = df['options'].apply(lambda x: x + (abstain_option,))

    if args.kind == 'topic':
        d = df[['topic', 'original', 'options']].groupby(['original']).first()\
            .reset_index().rename(columns={'original' : 'question'})
        grouping = d.groupby(['topic'])
    else:
        # NB: does not ask rephrases for every question, just the first 'original' for each topic
        if args.all_questions:
            d = df[['topic', 'original', 'options', 'question']]
        else:
            d = df[['topic', 'original', 'options']].groupby(['topic']).first().reset_index()\
                .merge(df[['original', 'question']], on=['original'], how='left')
        grouping = d.groupby(['topic', 'original'])

    questions = grouping.apply(make_question_lists, validate=args.validate).rename('questions')

    questions = questions.reset_index()

    if args.validate:
        if args.kind == 'topic':
            questions = group_n_rows(questions, 6)
        else:
            questions = questions.groupby(['topic']).agg(lambda x: x.tolist()).reset_index()

    questions['questions'] = questions['questions'].apply(base64_encode)

    base_name = os.path.splitext(args.filename)[0]
    name = f'{base_name}_mturk_{args.kind}'
    if args.validate:
        name += '_validate'
    name += '.csv'
    questions.to_csv(name,index=False, quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
    main()
