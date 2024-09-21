import anthropic as anth
import importlib.metadata
import logging
import math
import openai
import os
import packaging.version
import pprint
import string
import tenacity
import torch
import requests
from requests.adapters import HTTPAdapter, Retry
import signal
import subprocess
import tiktoken
import huggingface_hub

from .utils import (num_tokens_from_string,
                    num_tokens_from_messages)
from .distribution import Distribution

from .language_prompts import *

OPTIONS = string.ascii_uppercase

OPENAI_VERSION = importlib.metadata.version('openai')
OPENAI_API_CHANGE_VERSION = '0.28'
OLD_OPENAI_API = (packaging.version.parse(OPENAI_VERSION) <=
                  packaging.version.parse(OPENAI_API_CHANGE_VERSION))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger("httpx").setLevel(logging.ERROR)
# The library used by anthropic.


global client
global process
client = None
process = None

def wait_for_load(origin):
    s = requests.Session()

    retries = Retry(total=30,
                    backoff_factor=10,
                    status_forcelist=[ 500, 502, 503, 504 ])

    s.mount('http://', HTTPAdapter(max_retries=retries))

    response = s.get(f'{origin}models')
    logger.debug(pprint.pformat(response.json()))
    logger.info('Server loaded')

def setup_api(source=None, origin=None, model=None):
    global client
    global process

    if source == 'vllm':
        gpus = torch.cuda.device_count()
        download_dir = os.path.join(os.environ.get("HF_HOME"), 'hub')
        huggingface_hub.login(token=os.getenv("HUGGING_FACE_HUB_TOKEN").strip())
        vllm_args = ['python', '-m', 'vllm.entrypoints.openai.api_server',
                     '--model', model,
                     '--download-dir', download_dir,
                     '--trust-remote-code',
                     '--tensor-parallel-size', str(gpus)]
        process = subprocess.Popen(vllm_args, shell=False)
        # If debugging, you can redirect output of vllm:
        #  stdout=subprocess.DEVNULL)
        logger.info(f"Vllm process {process}")
        origin = "http://localhost:8000/v1/"
        wait_for_load(origin)
    if source and source == 'anthropic':
        api_key = os.getenv("ANTHROPIC_API_KEY").strip()
        client = anth.Anthropic(api_key=api_key)
    else: 
        # Use the openai framework otherwise
        if source and source == 'openai':
            origin = "https://api.openai.com/v1/"
            api_key = os.getenv("STANFORD_OPENAI_API_KEY").strip()
        else:
            api_key = "EMPTY"
            origin = origin + "/" if not origin.endswith("/") else origin
        client = openai.OpenAI(api_key = api_key.strip(), base_url = origin)

def close_api():
    global client
    global process

    client.close()
    if process:
        logger.info(f"Terminating process {process}")
        process.terminate()
        process.wait(timeout=60)
        process.kill()

@tenacity.retry(wait=tenacity.wait_chain(*[tenacity.wait_fixed(3) for i in range(3)] +
                       [tenacity.wait_fixed(5) for i in range(2)] +
                       [tenacity.wait_fixed(10)]))
def openai_chat_with_backoff(**kwargs):
    if process is not None and 'logprobs' in kwargs: # This is a  hacky way to check if it is a `vllm` call
        kwargs['top_logprobs'] = 5
    if OLD_OPENAI_API:
        response = client.ChatCompletion.create(**kwargs)
    else:
        response = client.chat.completions.create(**kwargs)
    ret = {'text' : response.choices[0].message.content,
            'got_stop_seq' : response.choices[0].finish_reason == 'stop'}
    if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs is not None:
        # Have to convert to the old format
        logprobs = None
        if response.choices[0].logprobs.content is not None and len(response.choices[0].logprobs.content) > 0:
            logprobs = []
            for token in response.choices[0].logprobs.content:
                logprobs.append({x.token : x.logprob for x in token.top_logprobs})
        elif hasattr(response.choices[0].logprobs, 'top_logprobs') and response.choices[0].logprobs.top_logprobs is not None:
            # This is how vllm returns them which does not adhere to the api
            logprobs = response.choices[0].logprobs.top_logprobs
        ret['logprobs'] = logprobs
    return ret

@tenacity.retry(wait=tenacity.wait_chain(*[tenacity.wait_fixed(3) for i in range(3)] +
                       [tenacity.wait_fixed(5) for i in range(2)] +
                       [tenacity.wait_fixed(10)]))
def openai_completion_with_backoff(**kwargs):
    # This is a  hacky way to check if it is a `vllm` call
    if process is not None and 'logprobs' in kwargs: 
        kwargs['logprobs'] = 5
    if OLD_OPENAI_API:
        response = client.Completion.create(**kwargs)        
    else:
        response = client.completions.create(**kwargs)
    ret = {'text' : response.choices[0].text,
            'got_stop_seq' : response.choices[0].finish_reason == 'stop'}
    if (hasattr(response.choices[0], 'logprobs') 
        and hasattr(response.choices[0].logprobs, 'top_logprobs')):
        ret['logprobs'] = response.choices[0].logprobs.top_logprobs
    return ret

@tenacity.retry(wait=tenacity.wait_chain(*[tenacity.wait_fixed(3) for i in range(3)] +
                       [tenacity.wait_fixed(5) for i in range(2)] +
                       [tenacity.wait_fixed(10)]))
def anthropic_chat_with_backoff(**kwargs):
    global client
    # Anthropic does not allow system messages in the messages
    system = None
    for i in range(len(kwargs['messages']) - 1, -1, -1):
        if kwargs['messages'][i]['role'] == 'system':
            system = kwargs['messages'][i]['content']
            del kwargs['messages'][i]
    if not hasattr(kwargs, 'system'):
        kwargs['system'] = system
    if 'stop' in kwargs:
        kwargs['stop_sequences'] = kwargs['stop']
        del kwargs['stop']
    if 'logprobs' in kwargs:
        del kwargs['logprobs']
    if 'top_logprobs' in kwargs:
        del kwargs['top_logprobs']

    response = client.messages.create(**kwargs)
    return {'text' : response.content[0].text,
            'got_stop_seq' : (response.stop_reason == 'end_turn' or 
                              response.stop_reason == 'stop_sequence'),
            'logprobs' : None}

@tenacity.retry(wait=tenacity.wait_chain(*[tenacity.wait_fixed(3) for i in range(3)] +
                       [tenacity.wait_fixed(5) for i in range(2)] +
                       [tenacity.wait_fixed(10)]))
def anthropic_completion_with_backoff(**kwargs):
    global client
    if 'max_tokens' in kwargs:
        kwargs['max_tokens_to_sample'] = kwargs['max_tokens']
        del kwargs['max_tokens']
    if 'stop' in kwargs:
        kwargs['stop_sequences'] = kwargs['stop']
        del kwargs['stop']
    if 'logprobs' in kwargs:
        del kwargs['logprobs']
    if 'top_logprobs' in kwargs:
        del kwargs['top_logprobs']

    response = client.completions.create(**kwargs)
    return {'text' : response.completion,
            'got_stop_seq' : response.stop_reason == 'stop_sequence',
            'logprobs' : None}

NO_ANSWER = 'none'

def filter_distribution_for_answers(full_distribution, num_options, weight_non_answers=False):
    '''
    Based on the chosen tokens, extracts the probabilities
    for each of the possible answer options, or an upper bound.
    See {santurkar_whose_2023} appendix a.3
    '''
    possible_answers = list(OPTIONS[:num_options].upper())
    distribution = Distribution()
    prob_sum = 0
    prob_min = None

    for token, prob in full_distribution.items():
        answer = get_option(token)
        if prob_min is None or prob < prob_min:
            prob_min = prob
        if answer in possible_answers:
            distribution[answer] += prob
        if answer not in possible_answers and weight_non_answers:
            distribution[None] += prob
        prob_sum += prob

    min_prob = min(prob_min, (1 - prob_sum))

    for answer in possible_answers:
        if answer not in distribution:
            distribution[answer] = min_prob
    return distribution.normalize()

def find_answer(option, answers):
    answer = None
    option = option.upper()
    if option in list(OPTIONS[:len(answers)].upper()):
        answer = answers[OPTIONS.index(option)]
    else:
        logger.warn("No option included")
    return answer

def get_highest_option_mass_distribution(logprobs, num_options, as_option=True):
    # Because we have multiple tokens returned, some of which may be filler
    # (such as '(') we need to only use the distribution with the largest
    # probability mass on the actual answers.
    possible_answers = list(OPTIONS[:num_options].upper())

    distributions = []
    for token_logprobs in logprobs:
        dist = Distribution()
        for elem, value in token_logprobs.items():
            if as_option:
                elem = get_option(elem)
            dist[elem] += math.exp(value)

        # NB: Don't operate on distritubtions storing log probs -- they 
        # will return 0 for items not in the distribution. Bad!
        distributions.append(dist.normalize())

    dist_options_weights = []
    for distribution in distributions:
        dist_options_weights.append(sum([distribution[answer] for answer in possible_answers]))

    max_weight_idx = dist_options_weights.index(max(dist_options_weights))

    return distributions[max_weight_idx].normalize()

def answer_to_distribution(response, answers):
    if 'logprobs' in response and response['logprobs'] is not None:
        distribution = get_highest_option_mass_distribution(response['logprobs'],
                                                               len(answers))

        distribution_filtered = filter_distribution_for_answers(distribution, len(answers))
        answer_distribution = {}

        for ((_, prob), answer) in zip(sorted(distribution_filtered.items()), answers):
            answer_distribution[answer] = prob

    else:
        option = get_option(response['text'])
        answer = find_answer(option, answers)

        if answer is None:
            logger.error(f'\tNo matching answer string.')

        answer_distribution = Distribution()
        answer_distribution[answer] = 1
    return answer_distribution

def get_option(text):
    token = text.strip()
    # This is the 'lower one eigth block'
    # a character used to indicate spaces.
    token = token.replace('▁', '').replace('(', '')\
                 .replace(')', '').replace(':', '')\
                 .replace('.', '')
    token = token.split(' ')[0]
    # Returning more than the first character here 
    # because tokenization should treat 'A' apart from 'Aardvark', e.g.
    return token

def generate_query_completions_model(prompt, temperature, model, query_language='english',
                                    endpoint=openai_completion_with_backoff,
                                    confidence=False, count_tokens=False):

    assistant_progress = anth.AI_PROMPT + " "

    confidence_prompt = anth.HUMAN_PROMPT + " " + CONFIDENCE_QUESTION[query_language] + anth.AI_PROMPT + " "

    if count_tokens:
        msg_tokens = num_tokens_from_string(prompt + assistant_progress, model)
        confidence_tokens = num_tokens_from_string(confidence_prompt, model)
        input_tokens = (msg_tokens)
        output_tokens = 512
        if confidence:
            input_tokens += (msg_tokens + 512 + confidence_tokens)
            output_tokens += 4
        return (input_tokens, output_tokens)

    try:
        ########### Asking the actual prompt
        response = endpoint(model=model,
                               prompt=(prompt + assistant_progress),
                               temperature=temperature,
                               stop=[anth.HUMAN_PROMPT],
                               max_tokens=512)

        logger.debug(f'\tProgress: {assistant_progress.strip()}')

        if not response['got_stop_seq']:
            logger.warn(f'\tNo stop')

        assistant_progress += response['text'].strip()
        full_conversation = (prompt + assistant_progress)
        ###########

        if confidence:
            ########### Asking for confidence
            confidence_response = endpoint(model=model,
                                 prompt=(full_conversation + 
                                         confidence_prompt),
                                 temperature=0,
                                 logprobs=5,
                                 max_tokens=4)

            full_conversation += (confidence_prompt + confidence_response['text'])
            confidence_str = confidence_response['text'].strip().lower()
            confidence = None
            try:
                confidence = float(confidence_str)
            except ValueError:
                logger.error(f"Model confidence {confidence_str} was not a float")
            ###########
        else:
            confidence = None

    except tenacity.RetryError:
        logger.error(f'\tEncountered a retry error, skipping.')

    return (assistant_progress, confidence, full_conversation)

def classify_query_completions_model(answers, prompt, temperature, model, query_language='english',
                                     endpoint=openai_completion_with_backoff,
                                     confidence=False, count_tokens=False):

    assistant_progress = ""

    answer_distribution = None
    answer_logprobs = None

    confidence_prompt = anth.HUMAN_PROMPT + " " + CONFIDENCE_QUESTION[query_language] + anth.AI_PROMPT + " "

    initial_tokens = 5

    if count_tokens:
        msg_tokens = num_tokens_from_string(prompt + assistant_progress, model)
        confidence_tokens = num_tokens_from_string(confidence_prompt, model)
        input_tokens = (msg_tokens)
        output_tokens = initial_tokens
        if confidence:
            input_tokens += (msg_tokens + 1 + confidence_tokens)
            output_tokens += 4
        return (input_tokens, output_tokens)

    try:
        ########### Asking the prompt
        response = endpoint(model=model,
                            prompt=(prompt),
                            temperature=temperature,
                            logprobs=5,
                            stop=[')'],
                            max_tokens=initial_tokens)

        logger.debug(f'\tProgress: {assistant_progress.strip()}')

        if not response['got_stop_seq']:
            logger.warn(f'\tNo stop')

        assistant_progress += response['text'].strip()
        ###########

        response_text = response['text']
        if response['got_stop_seq']:
            response_text += ')'
        full_conversation = (prompt + response_text)

        answer_distribution = answer_to_distribution(response, answers)
        if 'logprobs' in response and response['logprobs'] is not None:
            answer_logprobs = response['logprobs']
        ###########

        if confidence:
            ########### Asking for confidence
            confidence_response = endpoint(model=model,
                                           prompt=(full_conversation + 
                                                   confidence_prompt),
                                           temperature=0,
                                           logprobs=5,
                                           max_tokens=4)

            full_conversation += (confidence_prompt + confidence_response['text'])
            confidence_str = confidence_response['text'].strip().lower()
            confidence = None
            try:
                confidence = float(confidence_str)
            except ValueError:
                logger.error(f"Model confidence {confidence_str} was not a float")
            ###########
        else:
            confidence = None

    except tenacity.RetryError:
        logger.error(f'\tEncountered a retry error, skipping.')

    return (answer_distribution, confidence, assistant_progress, full_conversation, answer_logprobs)

def classify_query_chat_model(answers, messages, temperature, model, query_language='english',
                              endpoint=openai_chat_with_backoff,
                              confidence=False, count_tokens=False):

    assistant_progress = ''
    answer_distribution = None
    answer_logprobs = None

    confidence_prompt = [{'role' : 'user', 'content' : CONFIDENCE_QUESTION[query_language]}]

    classify_tokens = 5

    if count_tokens:
        msg_tokens = num_tokens_from_messages(messages)
        confidence_tokens = num_tokens_from_messages(confidence_prompt)
        input_tokens = (msg_tokens)
        output_tokens = classify_tokens
        if confidence:
            input_tokens += (msg_tokens + 1 + confidence_tokens)
            output_tokens += 4
        return (input_tokens, output_tokens)

    try:
        ########### Asking the actual prompt
        response = endpoint(model=model,
                            messages=(messages),
                            temperature=temperature,
                            logprobs=True,
                            top_logprobs=5,
                            max_tokens=classify_tokens)
        assistant_progress += response['text']


        assistant_message = [{'role' : 'assistant', 'content' : assistant_progress}]

        full_conversation = messages + assistant_message 

        answer_distribution = answer_to_distribution(response, answers)
        if 'logprobs' in response and response['logprobs'] is not None:
            answer_logprobs = response['logprobs']
        ###########

        if confidence:
            ########### Asking for confidence
            confidence_response = endpoint(model=model,
                                 messages=(full_conversation +
                                         confidence_prompt),
                                 temperature=0,
                                 logprobs=True,
                                 top_logprobs=5,
                                 max_tokens=4)

            confidence_message = [{'role' : 'assistant', 'content' : confidence_response['text']}]
            full_conversation += confidence_prompt + confidence_message
            confidence_str = confidence_response['text'].strip().lower()
            confidence = None
            try:
                confidence = float(confidence_str)
            except ValueError:
                logger.error(f"Model confidence \"{confidence_str}\" was not a float")
            ###########
        else:
            confidence = None

    except tenacity.RetryError:
        logger.error(f'\tEncountered a retry error, skipping.')

    return (answer_distribution, confidence, assistant_progress, full_conversation, answer_logprobs)

def generate_query_chat_model(messages, temperature, model, query_language='english',
                             endpoint=openai_chat_with_backoff,
                             confidence=False, count_tokens=False):

    assistant_progress = ''
    confidence_prompt = [{'role' : 'user', 'content' : CONFIDENCE_QUESTION[query_language]}]

    if count_tokens:
        msg_tokens = num_tokens_from_messages(messages)
        confidence_tokens = num_tokens_from_messages(confidence_prompt)
        input_tokens = (msg_tokens)
        output_tokens = 512
        if confidence:
            input_tokens += (msg_tokens + 512 + confidence_tokens)
            output_tokens += 4
        return (input_tokens, output_tokens)

    try:
        ########### Asking the actual prompt
        response = endpoint(model=model,
                           messages=(messages),
                           temperature=temperature,
                           max_tokens=512)
        assistant_progress += response['text']

        logger.debug(f'\tProgress: {assistant_progress.strip()}')

        if not response['got_stop_seq']:
            logger.warn(f'\tNo stop')

        assistant_message = [{'role' : 'assistant', 'content' : assistant_progress}]
        ###########

        full_conversation = messages + assistant_message

        if confidence:
            ########### Asking for confidence
            confidence_response = endpoint(model=model,
                                 messages=(full_conversation +
                                         confidence_prompt),
                                 temperature=0,
                                 logprobs=True,
                                 top_logprobs=5,
                                 max_tokens=4)

            confidence_message = [{'role' : 'assistant', 'content' : confidence_response['text']}]
            full_conversation += confidence_prompt + confidence_message
            confidence_str = confidence_response['text'].strip().lower()
            confidence = None
            try:
                confidence = float(confidence_str)
            except ValueError:
                logger.error(f"Model confidence \"{confidence_str}\" was not a float")
            ###########
        else:
            confidence = None

    except tenacity.RetryError:
        logger.error(f'\tEncountered a retry error, skipping.')

    return (assistant_progress, confidence, full_conversation)
