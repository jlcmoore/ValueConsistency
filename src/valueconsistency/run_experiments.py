'''
Sends off various jobs to beaker to run the model experiment

Author: Jared Moore

Usage, e.g.:
`run_experiments run --filename run_experiments_sub.yaml`
'''

import argparse
from beaker import Beaker
from beaker.data_model.experiment_spec import ExperimentSpec
from beaker.exceptions import ExperimentNotFound
import copy
import datetime
import json
import logging
import os
import time
import shutil
import string
import tqdm
import yaml

logging.basicConfig(level=logging.INFO)

BEAKER_RESULTS_DIR = "beaker"

RESULTS_DIR = 'results/'

TEMPLATE_EXP_FILE = "beaker_experiment.yaml"

WORKSPACE = "ai2/valuebank"

def file_to_beaker_file(filename):
    run = os.path.splitext(os.path.basename(filename))[0]
    iso = datetime.datetime.now().date().isoformat()
    return os.path.join(BEAKER_RESULTS_DIR, f'{run}-{iso}.json')

def beaker_results(args, beaker):

    if not os.path.exists(args.filename):
        raise ValueError("Invalid filename. Does not exist.")

    if not os.path.exists(args.filename):
        raise ValueError("Experiment file does not exist.")

    experiments = {}

    try:
        with open(args.filename, 'r') as exp_file:
            experiments = json.load(exp_file)

            for exp_id in experiments.keys():
                try:
                    exp = beaker.experiment.get(exp_id)
                except ExperimentNotFound:
                    logging.warn(f"{exp_id} not found")
                    continue
                # TODO:
                succeeded = False 
                for job in exp.jobs:
                    status = job.status
                    if status.started is not None and status.exit_code is not None:
                        succeeded = True

                dataset = beaker.experiment.results(exp)
                url = beaker.experiment.url(exp)
                if not succeeded and not args.cancel:
                    logging.info(f"Experiment {exp.id} not finished. Consult the url: {url}")
                    continue
                if dataset is None:
                    logging.error(f"Experiment {exp.id}, {exp.name} had status {status}."
                        + f"Consult the url: {url}")
                    if args.cancel:
                        beaker.experiment.stop(exp)
                    if args.rerun:
                        spec = beaker.experiment.spec(exp)
                        beaker.experiment.delete(exp)
                        exp = beaker.experiment.create(spec, name=exp.name)
                        experiments[exp.id] = experiments[exp_id]
                        del experiments[exp_id]
                    continue

                try:
                    beaker.dataset.fetch(dataset, target=RESULTS_DIR)
                except FileExistsError as e:
                    logging.error(f"File exists: {e}")

                gantry_dir = os.path.join(RESULTS_DIR, ".gantry")
                if os.path.exists(gantry_dir):
                    shutil.rmtree(gantry_dir)

    finally:
        with open(args.filename, 'w') as exp_file:
            exp_file.write(json.dumps(experiments))

def convert_to_underscore(name):
    """"E.g. gpuCount to gpu_count"""
    result = [name[0].lower()]
    for c in name[1:]:
        if c in (string.ascii_uppercase):
            result.append('_')
            result.append(c.lower())
        else:
            result.append(c)
    return ''.join(result)

# Test the function

def run_beaker(job, default_yaml, beaker, dry_run):
    default_yaml = copy.deepcopy(default_yaml)

    # All this is just a way to get the nested variables out of the relevant
    # dicts
    beaker_args = enumerate_args(job['beaker'])
    for variables, value in beaker_args.items():
        variables = [convert_to_underscore(variable) for variable in variables]
        resource_part = default_yaml.tasks[0]
        if len(variables) > 0:
            while len(variables) > 1:
                resource_part = getattr(resource_part, variables[0])
                variables = variables[1:]
            setattr(resource_part, variables[0], value)

    default_yaml.tasks[0].arguments = job['arguments']

    logging.info(f"Running experiment: {job['arguments']}")
    if not hasattr(default_yaml, 'budget') or default_yaml.budget is None:
        logging.error(f"Experiment {default_yaml}")

    if dry_run:
        return type('var', (object,), {'id': None})

    experiment = beaker.experiment.create(default_yaml)
    time.sleep(5)
    # for some reason the budgets keep failing. Waiting seems to fix this...

    return experiment

def expand_yaml_dict(yaml):
    return expand(yaml, [], {})

def expand(node, arguments, beaker):
    assert isinstance(node, dict)

    assert 'arguments' in node
    arguments += node['arguments']

    if 'beaker' in node:
        beaker.update(node['beaker'])

    if 'subs' in node: # recursive case
        runs = []
        for sub in node['subs']:
            runs += expand(sub, copy.deepcopy(arguments), copy.deepcopy(beaker))
    else: # base case
        runs = [{'arguments' : arguments, 'beaker' : beaker}]
    return runs

def enumerate_args(d, keys=()):
    items = []
    for k, v in d.items():
        new_keys = keys + (k,)
        if isinstance(v, dict):
            items.extend(enumerate_args(v, new_keys).items())
        else:
            items.append((new_keys, v))
    return dict(items)

def run_all_beaker(args, beaker):
    if not os.path.exists(args.filename):
        raise ValueError("Invalid filename. Does not exist.")
    with open(args.filename, 'r') as f:
        jobs = expand_yaml_dict(yaml.safe_load(f))

    exp_filename = file_to_beaker_file(args.filename)

    if os.path.exists(exp_filename):
        raise ValueError("Experiments already run. Did you mean to run `results`?")

    default_exp = ExperimentSpec.from_file(TEMPLATE_EXP_FILE)

    experiments = {}

    try:
        for job in jobs:
            experiment = run_beaker(job, default_exp, beaker, args.dry_run)
            experiments[experiment.id] = job['arguments']
    finally:
        as_json = json.dumps(experiments)
        if not args.dry_run:
            with open(exp_filename, 'w') as exp_file:
                exp_file.write(as_json)

def main():
    parser = argparse.ArgumentParser(
                prog='run_model_experiments',
                description='Sends off various jobs to beaker to run the model experiment')

    subparsers = parser.add_subparsers(dest='cmd', required=True)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--filename', required=True)

    run_parser = subparsers.add_parser("run", parents=[parent_parser],
        help='Runs all of the model experiments in the given file with.')
    run_parser.set_defaults(func=run_all_beaker)

    run_parser.add_argument('--dry-run', default=False, action='store_true',
        help='Does not actually spin up any experiments.')

    results_parser = subparsers.add_parser("results", parents=[parent_parser],
        help='Downloads the results of the model experiments for the experiments file, if they exist.')

    results_parser.set_defaults(func=beaker_results)

    results_parser.add_argument('--rerun', default=False, action='store_true',
        help='Reruns failed experiments.')

    results_parser.add_argument('--cancel', default=False, action='store_true',
        help='Cancels currently running experiments.')

    args = parser.parse_args()

    beaker = Beaker.from_env(default_workspace=WORKSPACE)
    args.func(args, beaker)
