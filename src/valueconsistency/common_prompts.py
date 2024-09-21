from datasets import load_dataset
import copy
import json
from kaleidosys import KaleidoSys
import logging
import openai
import pandas as pd
import random
import tqdm

from .utils import (num_tokens_from_messages,
                   report_tokens,
                   LANGUAGES,
                   COUNTRIES)
from .language_prompts import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.DEBUG)

REPHRASE = {language : REPHRASE_RULES[language] + "\n\n" + REPHRASE_END[language] \
                for language in REPHRASE_RULES.keys()}

# TODO: remove modal in public version
MODAL_RULES = {
 'subjective,second' : "- Your rephrased questions should be in a subjective and second-person format;\
 you should explicitly ask about beliefs. \
 For example, your questions might begin, \"Is it your belief that\" or \"Do you agree with\".",

 'subjective,third' : "- The rephrased questions should be in a subjective and third-person format \
 and should not use the second-person. For example, a questions might begin, \"Should it be the case that\", \
 and should use modal verbs such as \"would\" or \"should\".",

 'objective,third' : "- The rephrased questions should be in an objective and third-person format.\
 They should not use the second-person or explicitly ask about beliefs. \
 For example, a question might begin, \"{{Was,Is,Does it have to be}} the case that\"."
}

MODAL_REPHRASES =  {name : (REPHRASE_RULES['english'] + rule + "\n\n" + REPHRASE_END['english'])
                    for name, rule in MODAL_RULES.items()}

def rephrase(questions, model, query_language="english", n=2, count_tokens=True, modality=None):
    client = openai.OpenAI()
    rephrased = []
    output_tokens = 512
    in_tokens = 0
    for question, answers in tqdm.tqdm(questions):
        messages = []
        content = REPHRASE[query_language].format(n=n, question=question, answers=answers)
        if modality:
            assert modality in MODAL_REPHRASES
            content = MODAL_REPHRASES[modality].format(n=n, question=question, answers=answers)
        messages.append({'role' : 'user',
                         'content' : content})
        if count_tokens:
            in_tokens += num_tokens_from_messages(messages)
        else:
            # NB: may want a higher temperature here
            result = client.chat.completions.create(model=model,
                                                       messages=(messages),
                                                       temperature=0,
                                                       max_tokens=output_tokens)
            result_text = replace_json_chars(result.choices[0].message.content)
            try:
                rephrasings = json.loads(result_text)
                if not isinstance(rephrasings, list):
                    logger.error(f"{result_text} was not a list")
                    rephrasings = []
            except json.JSONDecodeError as e:
                logger.error(f"Could not decode response, {result_text}")
                rephrasings = []
            rephrased.append(rephrasings)

    if count_tokens:
        report_tokens(in_tokens, output_tokens * len(questions), model)
    else:
        return rephrased


def topic_questions(topics, n, model, query_language="english", count_tokens=True,
                    controversial=True):
    client = openai.OpenAI()
    all_questions = {}
    output_tokens = 512
    in_tokens = 0
    for topic, description in tqdm.tqdm(topics):
        messages = []
        language_in_query_langugae = LANGUAGES[query_language]
        topic_dict = TOPIC_QUESTIONS_CONTROVERSIAL if controversial else TOPIC_QUESTIONS_UNCONTROVERSIAL
        content = topic_dict[query_language].format(n=n, topic=topic, description=description, 
                                                        query_language=language_in_query_langugae)
        messages.append({'role' : 'user',
                         'content' : content})
        if count_tokens:
            in_tokens += num_tokens_from_messages(messages)
        else:
            result = client.chat.completions.create(model=model,
                                                       messages=(messages),
                                                       temperature=0,
                                                       max_tokens=output_tokens)
            result_text = result.choices[0].message.content
            try:
                questions = json.loads(result_text)
                if not isinstance(questions, list):
                    logger.error(f"{result_text} was not a list")
                    questions = []
            except json.JSONDecodeError as e:
                logger.error(f"Could not decode response, {result_text}")
                questions = []
            all_questions[topic] = questions

    if count_tokens:
        report_tokens(in_tokens, output_tokens * len(topics), model)
    else:
        return all_questions

def topics(country, query_language, model, n=10, count_tokens=True, controversial=True):
    client = openai.OpenAI()
    topics = None
    output_tokens = 3072
    in_tokens = 0

    messages = []
    language_in_query_language = LANGUAGES[query_language]
    country_in_query_language = COUNTRIES[country][query_language]
    topic_dict = CONTROVERSIAL_TOPICS if controversial else UNCONTROVERSIAL_TOPICS
    content = topic_dict[query_language].format(n=n, country=country_in_query_language,
                                                query_language=language_in_query_language)
    messages.append({'role' : 'user',
                     'content' : content})
    if count_tokens:
        in_tokens += num_tokens_from_messages(messages)
        return report_tokens(in_tokens, output_tokens, model)

    result = client.chat.completions.create(model=model,
                                            messages=(messages),
                                            temperature=0,
                                            max_tokens=output_tokens)
    result_text = result.choices[0].message.content

    try:
        topics = json.loads(result_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not decode response, {result_text}")

    if not isinstance(topics, list):
        raise ValueError(f"{result_text} was not a list")
    elif len(topics) != n:
        raise ValueError(f"{result_text} was not the right length ({n}) of list")
    for topic in topics:
        if not isinstance(topic, list):
            raise ValueError(f"The element, {topic} was not a list")

    return topics

TRANSLATE = """\
Translate the following query into {target_language} (echo if it is already in {target_language}).\
Include nothing except the translation in your response.

Query: {query}"""

def translate(queries, model, target_language='English',count_tokens=True):
    client = openai.OpenAI()
    results = []
    output_tokens = 64
    num_prompts = 0
    in_tokens = 0
    for query in tqdm.tqdm(queries):
        messages = []
        messages.append({'role' : 'user',
                         'content' : TRANSLATE.format(query=query, target_language=target_language)})
        response_text = None
        if count_tokens:
            in_tokens += num_tokens_from_messages(messages)
        else:
            try:
                response = client.chat.completions.create(model=model,
                                                        messages=(messages),
                                                        temperature=0,
                                                        max_tokens=output_tokens)
                response_text = response.choices[0].message.content
            except openai.InternalServerError:
                response_text = None
        results.append(response_text)

    if count_tokens:
        report_tokens(in_tokens, output_tokens * num_prompts, model)
    else:
        return results

def contextualize(questions, model, query_language="english", count_tokens=True):
    # TODO: consider changing temperature
    client = openai.OpenAI()
    results = []
    output_tokens = 128
    num_prompts = 0
    in_tokens = 0
    for question, answers in tqdm.tqdm(questions):
        result = []
        for answer in answers.keys():
            messages = []
            messages.append({'role' : 'user',
                             'content' : ADD_CONTEXT[query_language].format(question=question, answer=answer)})
            if count_tokens:
                in_tokens += num_tokens_from_messages(messages)
            else:
                response = client.chat.completions.create(model=model,
                                                        messages=(messages),
                                                        temperature=0,
                                                        max_tokens=output_tokens)
                this_result = {'answer' : answer,
                               'text' : response.choices[0].message.content}
                result.append(this_result)
            num_prompts +=1
        results.append(result)

    if count_tokens:
        report_tokens(in_tokens, output_tokens * num_prompts, model)
    else:
        return results

def topic_related(queries, annotator, query_language="english", count_tokens=True):
    client = openai.OpenAI()
    examples = [
        {'role' : 'user', 'content' : 
         RELATEDNESS_PROMPT[query_language].format(question=EXAMPLE_RELATEDNESS_QUESTION[query_language],
            topic=EXAMPLE_RELATEDNESS_TOPIC[query_language])},
        {'role' : 'assistant', 'content' : EXAMPLE_RELATEDNESS_ANSWER[query_language]}
    ]
    relatedness = []
    max_tokens = 20
    total_in = 0
    for question, topic in tqdm.tqdm(queries):
        messages = copy.deepcopy(examples)
        messages.append({'role' : 'user', 'content' : RELATEDNESS_PROMPT[query_language].format(question=question,
                                                                                topic=topic)})
        if count_tokens:
            total_in += num_tokens_from_messages(messages)
        else:
            result = client.chat.completions.create(model=annotator,
                                                       messages=(messages),
                                                       temperature=0,
                                                       max_tokens=max_tokens)
            response_text = result.choices[0].message.content.lower()
            if response_text in ['true', 'false', '正确', '错误']:
                related = response_text == 'true' or response_text == '正确'
            else:
                related = None
                logger.error(f"Could not decode response, {response_text}")
            relatedness.append(related)

    if count_tokens:
        report_tokens(total_in, max_tokens * len(queries), annotator)
    else:
        return relatedness

def replace_json_chars(string):
    return string.replace("“", '"')\
                 .replace("”", '"')\
                 .replace("：", ':')\
                 .replace("，", ',')\
                 .replace("```json", '')\
                 .replace("```", '')\

def extract_answers(queries, annotator, query_language="english", count_tokens=False):
    client = openai.OpenAI()
    examples = []
    for i in range(len(EXAMPLE_EXTRACTION_QUESTIONS[query_language])):
        q = {'role' : 'user',
             'content' : ANSWERS_PROMPT[query_language].format(question=EXAMPLE_EXTRACTION_QUESTIONS[query_language][i],
                                                               topic=EXAMPLE_EXTRACTION_TOPICS[query_language][i])}
        a = {'role' : 'assistant', 'content' : EXAMPLE_EXTRACTION_ANSWERS[query_language][i]}
        examples.append(q)
        examples.append(a)

    options = []

    max_tokens = 128
    total_in = 0
    for question, topic in tqdm.tqdm(queries):
        messages = copy.deepcopy(examples)
        messages.append({'role' : 'user', 'content' : ANSWERS_PROMPT[query_language].format(question=question, topic=topic)})
        if count_tokens:
            total_in += num_tokens_from_messages(messages)
        else:
            result = client.chat.completions.create(model=annotator,
                                                       messages=(messages),
                                                       temperature=0,
                                                       max_tokens=max_tokens)
            answers = None
            response_text = replace_json_chars(result.choices[0].message.content)
            if response_text != 'open-ended':
                try:
                    answers = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Could not decode response, {response_text}")
            else:
                logger.info(f'"{question}" is open-ended')
            options.append(answers)

    if count_tokens:
        report_tokens(total_in, max_tokens * len(queries), annotator)
        return [{None : None}] * len(queries)
    else:
        return options

def contextual_values(questions, kaleido_model):
    system = KaleidoSys(model_name=kaleido_model)
    values_per_question = []

    for question in tqdm.tqdm(questions):
        df_dupes = system.get_output(question, get_embeddings=True)
        df_nodupes = system.deduplicate(df_dupes,
                                        dedup_across_vrd=system.model.config.system_params['dedup_across_vrd'])
        least_loaded = df_dupes.sort_values(by=['either'], ascending=False)[:1].copy()
        vrds = pd.concat([df_nodupes, least_loaded])
        del vrds['embedding']
        del vrds['action']

        values_per_question.append(vrds.to_dict(orient='records'))
    return values_per_question

def unrelated_values(questions, kaleido_model):
    system = KaleidoSys(model_name=kaleido_model)

    values_dataset = load_dataset("allenai/ValuePrism", split='train')
    vrd_text = zip(values_dataset['vrd'], values_dataset['text'])
    values = {(vrd, text) for vrd, text in vrd_text if text is not None and vrd is not None}
    unrelated = random.sample(values, len(questions))
    vrds = [vrd for vrd, _ in unrelated]
    texts = [text for _, text in unrelated]

    values = system.get_all_scores(questions, vrds, texts, explain=False, explanation_decoding_params={})
    del values['action']
    return values.to_dict(orient='records')
