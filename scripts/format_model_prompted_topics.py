import argparse
import copy
import json
import logging
import os
import pandas as pd
import random

from valueconsistency import (num_tokens_from_messages,
                      report_tokens,
                      rephrase,
                      contextualize,
                      topic_related,
                      extract_answers,
                      contextual_values,
                      unrelated_values,
                      topic_questions,
                      topics,
                      translate,
                      LANGUAGES,
                      COUNTRIES,
                      SCHWARTZ_VALUES,
                      SCHWARTZ_VALUES_BY_LANGUAGE,
                      SCHWARTZ_VALUES_DICT_FMT)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.DEBUG)

'''
Example commands

```
python scripts/format_model_prompted_topics.py \
--filename controversial_chinese_china.jsonl \
--query-language chinese \
--annotator gpt-4 \
new \
--country china \
--num-topics 30 \
--num-questions 5 \
--num-rephrases 4

python  scripts/format_model_prompted_topics.py \
--filename controversial_english_china.jsonl \
--query-language english \
--annotator gpt-4 \
translate \
--reference-file model_prompted_chinese_china.jsonl

```
'''

def generate(args):
    logger.info("Querying for controversial topics.")

    tops = topics(args.country, args.query_language, n=args.num_topics,
                    model=args.annotator, count_tokens=args.count_tokens,
                    controversial=args.controversial)
    # Need to query for questions about these topics
    if args.count_tokens:
        tops = [('', '')] * args.num_topics

    logger.info("Querying for questions and answers")
    questions_by_topic = topic_questions(tops, n=args.num_questions, model=args.annotator,
                                        query_language=args.query_language, count_tokens=args.count_tokens,
                                        controversial=args.controversial)

    topics_only = [t[0] for t in tops]
    translated_topics = translate(topics_only, model=args.annotator, target_language='English',
                                  count_tokens=args.count_tokens)

    records = []
    if not args.count_tokens:
        topic_translations = {topic : translated for topic, translated in zip(topics_only, translated_topics)}

        for topic, questions in questions_by_topic.items():
            if len(questions) > args.num_questions:
                questions = random.sample(questions, args.num_questions)
            for question in questions:
                records.append({'topic' : topic, 'topic_english' : topic_translations[topic],
                                'question' : question})
    df = pd.DataFrame(records)

    # For some reason sometimes a dict is returned
    df['question'] = df['question'].apply(lambda x: x['question'] if 'question' in x else x)

    num_records = args.num_questions * args.num_topics
    if args.count_tokens:
        question_topics = list(zip([''] * num_records, [''] * num_records))
    else: 
        question_topics = list(zip(df['question'], df['topic']))

    options = extract_answers(question_topics, args.annotator, query_language=args.query_language,
                              count_tokens=args.count_tokens)

    if not args.count_tokens:
        df['options'] = options
        df = df[df['options'].notnull()].copy()

        # Right now just preventing any questions with more than two options
        # More elgant would be to prevent groups by topic from having different
        # numbers of options
        df = df[df['options'].map(len) <= 2].copy()

        # Get rid of neutral questions, also those that in a different language don't use the right keys.
        opposite_stances = df['options'].apply(lambda x: 'supports' in x.values() and 'opposes' in x.values())
        df = df[opposite_stances]

        # Get rid of topics that have too few questions and clamp the number of questions 
        # for each topic
        grouped = df.groupby('topic').apply(lambda x: x.head(args.topic_questions) 
                                                      if len(x) >= args.topic_questions else None)
        question_answers = list(zip(df['question'], df['options']))
    else:
        question_answers = list(zip([''] * num_records, [{'a' : '', 'b' : ''}] * num_records))

    logger.info("Rephrasing questions")
    rephrased = rephrase(question_answers, args.annotator, n=args.num_rephrases,
                         query_language=args.query_language,
                         count_tokens=args.count_tokens)

    # logger.info("Contextualizing questions")
    # contexts = contextualize(question_answers, args.annotator, query_language=args.query_language,
    #                          count_tokens=args.count_tokens)
    # df['contexts'] = contexts

    if args.count_tokens:
        return

    if not args.no_schwartz:
        values = SCHWARTZ_VALUES_DICT_FMT[args.query_language]
        df['values'] = [values] * len(df)

    elif args.query_language == "english" and args.kaleido_model is not None:
        logger.info("Querying for values related to original questions")
        values = contextual_values(df['question'], args.kaleido_model)
        df['values'] = values

        logger.info("Querying for values unrelated to original questions")
        unrelated = unrelated_values(df['question'], args.kaleido_model)
        df['unrelated_value'] = unrelated

    # Editing the dataframe for the rephrasings
    new_df = add_rephrased(df, rephrased)

    # Get rid of questions with fewer than three rephrases
    new_df = new_df.groupby(['original']).apply(lambda x: x if x['question'].nunique() >= 3 else None)\
                   .reset_index(drop=True)
    # Get rid of topic with fewer than three questions
    new_df = new_df.groupby(['topic']).apply(lambda x: x if x['original'].nunique() >= 3 else None)\
                   .reset_index(drop=True)

    new_df.to_json(os.path.join(args.output_directory, args.filename), orient='records', lines=True)

def add_rephrased(df, rephrased):
    records = []
    for (_, row), rephrasings in zip(df.iterrows(), rephrased):
        original = row['question']
        try:
            questions = set([original] + rephrasings)
        except TypeError:
            logging.error(f"Unhashable: {original}, {rephrasings}")
            continue
        record = row.to_dict()
        if 'question' in record:
            del record['question']
        for question in questions:
            new_record = copy.deepcopy(record)
            new_record.update({'question' : question, 'original' : original,
                                'rephrase' : question != original})
            records.append(new_record)

    new_df = pd.DataFrame(records)
    return new_df

def translate_group(group, query_language, annotator, count_tokens=False):
    originals = group.groupby(['original']).first().reset_index()
    del originals['question']
    del originals['rephrase']
    # Capitalize the first letter.
    target_language = query_language[:1].upper() + query_language[1:]
    
    topic = translate([originals['topic'].iloc[0]], model=annotator,
                      target_language=query_language, count_tokens=count_tokens)
    originals['topic'] = topic * len(originals)

    if query_language == 'english':
        originals['question'] = originals['original_english']
    else:
        originals['question'] = translate(originals['original'], model=annotator,
                                          target_language=query_language, count_tokens=count_tokens)

    originals = originals[originals['question'].notnull()]
    # Translate the options
    if query_language == 'english':
        originals['options'] = originals['options_english']
    else:
        options = set(originals['options'].apply(lambda x: list(x.keys())).sum())

        translated_options = translate(options, model=annotator, target_language=query_language,
                                       count_tokens=count_tokens)
        option_translations = {opt : translated for opt, translated in zip(options, translated_options)}

        CHINESE_YES_NO_ENGLISH = {'否': 'No', '是': 'Yes'} # for some reason 是 is translating to is
        if CHINESE_YES_NO_ENGLISH.keys() == option_translations.keys():
            option_translations = CHINESE_YES_NO_ENGLISH
        originals['options'] = originals['options'].apply(lambda x: {option_translations[opt] : stance 
                                                                     for opt, stance in x.items()})

    # Translate the contexts
    # if 'contexts' in originals.columns:
    #     def translate_context(contexts):
    #         new_contexts = []
    #         for context in contexts:
    #             new_contexts.append({
    #              'answer' : option_translations[context['answer']],
    #              'text' : translate([context['text']], model=annotator,
    #                                 target_language=query_language,
    #                                 count_tokens=count_tokens)[0]
    #             })
    #         return new_contexts
    #     originals['contexts'] = originals['contexts'].apply(translate_context)
    return originals

def translate_func(args):
    if args.count_tokens:
        raise ValueError("Not implemented")

    df = pd.read_json(args.reference_file, orient='records', lines=True)

    new = df.groupby(['topic']).apply(translate_group,
                                      query_language=args.query_language,
                                      annotator=args.annotator,
                                      count_tokens=args.count_tokens)\
            .reset_index(drop=True)

    num_rephrases = df.groupby(['original']).size().max()

    question_answers = list(zip(new['question'], new['options']))

    rephrased = rephrase(question_answers, args.annotator, n=num_rephrases,
                             query_language=args.query_language,
                             count_tokens=False)

    new = add_rephrased(new, rephrased)

    if 'values' in df.columns:
        new['values'] = [SCHWARTZ_VALUES_DICT_FMT[args.query_language]] * len(new)

    new.to_json(os.path.join(args.output_directory, args.filename), orient='records', lines=True)

def main():
    logger.info(f'HF_HOME: {os.environ.get("HF_HOME")}')
    parser = argparse.ArgumentParser(
                    prog='format')

    parser.add_argument('--filename', required=True, help="What to output the results as.")

    parser.add_argument('--output-directory', default='', help="Where to output the results.")

    parser.add_argument('--annotator', default=None, required=True, type=str,
        help="Which OpenAI model should annotate the data")

    parser.add_argument('--controversial', action='store_true', 
        dest='controversial', help="Whether to prompt for controversial topics")
    parser.add_argument('--no-controversial', dest='controversial', action='store_false')
    parser.set_defaults(controversial=True)

    parser.add_argument('--query-language', choices=LANGUAGES.keys(), required=True, type=str,
        help="The language to query the models in.")

    parser.add_argument('--count-tokens', default=False, action='store_true', help=
        "Count the max number of tokens when using OpenAI models. Does not query models. Writes to stdout.")

    subparsers = parser.add_subparsers(dest='cmd', required=True)

    new_gen_parser = subparsers.add_parser("new",
        help='Generate an entirely new data set.')

    new_gen_parser.add_argument('--no-schwartz', default=False, action='store_true', help=
        "Whether to append Schwartz's values.")

    new_gen_parser.add_argument('--country', choices=COUNTRIES.keys(), required=True, type=str,
        help="The country to probe for controversial topics in.")

    new_gen_parser.add_argument('--topic-questions', type=int, default=3, help=
        "The number of questions to include for each topic.")

    new_gen_parser.add_argument('--num-topics', type=int, required=True, help=
        "The number of topics to generate.")

    new_gen_parser.add_argument('--kaleido-model', default=None, required=False, type=str,
        help="Which Kaleido model should annotate the values")

    new_gen_parser.add_argument('--num-questions', type=int, default=2, help=
        "The number of questions for each topic.")

    new_gen_parser.add_argument('--num-rephrases', type=int, default=2, help=
        "The number of rephrased questions for each topic (this num. plus the one original).")

    new_gen_parser.set_defaults(func=generate)

    translate_parser = subparsers.add_parser("translate",
        help='Translate an existing data set.')

    translate_parser.add_argument('--reference-file', type=str, required=True, help=
        "The reference data set to base this one off of.")
    translate_parser.set_defaults(func=translate_func)

    args = parser.parse_args()

    args.func(args)

if __name__ == "__main__":
    main()
