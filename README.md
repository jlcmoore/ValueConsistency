
# ValueConsistency

This is the repository for the paper, [Are Large Language Models Consistent over Value-laden Questions?](https://arxiv.org/abs/2407.02996).


## Citing 

You may cite it like so:

```
@inproceedings{
    moore2024largelanguagemodelsconsistent,
    title={Are Large Language Models Consistent over Value-laden Questions?},
    author={Jared Moore and Tanvi Deshpande and Diyi Yang},
    booktitle={The 2024 Conference on Empirical Methods in Natural Language Processing},
    year={2024},
    url={https://arxiv.org/abs/2407.02996}
}
```

### ValueConsistency Data

The files in  `data/*.jsonl` make up the ValueConsistency data set with both controversial and uncontroversial questions in English, Chinese, German, and Japanese for topics from the U.S., China, Germany, and Japan. 

- `controversial`, bool: Whether or not the question is controversial.
- `language`, str: The language the question is asked in.
- `country`, str: The country in which the topic of this question was generated.
- `original`, str: The original text of the question this question was paraphrased from.
- `original_english`, str: A translation of `original` into English.
- `topic`, str: The topic of the question.
- `topic_english`, str: `topic` translated to English.
- `options` dict[str, str]: A dict of possible answers to this question, in the form of the answer mapping to its stance (e.g. "yes" : "supports").
- `question`, str: The text of this question.
- `rephrase`, bool: Whether `question` == `original`

We generate these data using commands which appear in the [generate](generate.md) file.

## Set-up

First install a python virtual environment and install the relevant packages

```
Make install
```

(Note that this might fail due to the `vllm` package needing to be installed on linux. If so edit `requirements.txt` locally to comment it out.)

We require `python>=3.10`. On Mac to install it run:

```
brew install python@3.10
```

To use Pandoc for the mturk experiments you must download it, e.g. 

```
brew install pandoc
```

You must have API keys stored as environment variables. On MacOS to do so run the following commands:

```
echo 'export OPENAI_API_KEY=VALUE' >> ~/.zshrc
echo 'export ANTHROPIC_API_KEY=VALUE' >> ~/.zshrc
echo 'export HUGGING_FACE_HUB_TOKEN=VALUE' >> ~/.zshrc
source ~/.zshrc
```

Then you can load the virtual environment as so:

```
source ./env-valuebank/bin/activate
```

To deactivate exit the terminal session or run:

```
deactivate
```

If jupyter is not installed you can install it using Homebrew:

```
brew install jupyterlab
```

You can alternatively set up a conda environment for running batch jobs:

```
conda env create -f environment.yml
conda activate valuebank
```

## Running experiments

There are two kinds of tasks we can run: classification (multiple choice) and generation (open-ended).

The idea is to look at the expressed opinions/preferences of a model by asking in a multiple choice way but also to see if open-ended generation (the more common way of interacting with an LLM) aligns with a particular polar direction.


### Classification

The below command prompts `gpt-4o` with only one of the entries from `data/uncontroversial_english_us..jsonl` (because of `--sample-size 1`). The result will appear at a file like: `results/uncontroversial_english_us./gpt-4o/classification-2024-01-21.json`

```
prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/uncontroversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --sample-size 1
```

### Generation

Generation commands, as below, are very similar. The output will go to something like: `results/uncontroversial_english_us/gpt-4o/generation-2024-01-21.json`

```
prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/uncontroversial_english_us.jsonl --query-language english --task generation --sample-size 1
```

Note that we need an additional step before analyzing the generation data; we must detect the resulting stance of the neutral generation to see if it aligns more with a polarized generation.

To do so we run the following command on the output of the previous command. We use a variety of models as the annotators here and calculate the inter-rater reliability between them. This command outputs to  `results/uncontroversial_english_us/gpt-4o/generation-2024-01-21_annotated_stance_gpt-4o.json`

```
judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/uncontroversial_english_us/gpt-4o/generation-2024-01-21.json
```

### All commands

See [the rest of the commands for experiments](experiments.md) in the linked file.

Note that all result data files are stored as json. Run `load_run` to extract the variables for that run (the commands you used) and the data as Pandas DataFrame.

You can download all of the results of our runs [here](https://drive.google.com/drive/folders/1SIduLOYD1YOhE8fdu6VuY2PMaeh31h3R).

### Hugging Face Models

To use models from HuggingFace (such as the `llama-2` family) we rely on the `vllm` package which allows us to hide the models behind a server and query them using the `openai` api. Most of these models require use of GPUs.

This is only possible on linux machines.

## Analysis

To run the jupyter notebooks simply run the following command and navigate to the appropriate file. `notebooks/analysis.ipynb` generates most of the relevant figures.

```
jupyter notebook
```

## Mturk experiments

We ran a few experiments on Mturk workers in English on U.S.-based topics. See the `mturk` directory.

In that directory run these pandoc commands to compile the task for the relevant platform. Edit the `id="questions-json"` tag in `template.html` depending on if releasing on Mturk or CloudResearch, the template variables differ.

Change `variables.json` from `question_type : "validate"` to  `question_type : "query"` to either validate the data or ask participants the questions in the data, respectively, or to  `question_type : "controversial"` for controversial questions.

For validation, see the commands in generate.md to generate the rightful csv files to feed into the platforms.

For querying, run commands like this to generate the relevant csvs.

```
python scripts/format_mturk.py --filename data/controversial_english_us.jsonl --kind rephrase 
python scripts/format_mturk.py --filename data/controversial_english_us.jsonl --kind topic
```

### Paraphrase Validation

```
pandoc --standalone -f html+raw_html -t html \
--standalone --embed-resources \
--template template.html \
--output mturk.html \
--metadata=title:HIT \
paraphrase.html
```

### Controversiality Validation

```
pandoc --standalone -f html+raw_html -t html \
--standalone --embed-resources \
--template template.html \
--output mturk.html \
--metadata=title:HIT \
controversial.html
```

### Query compilation

```
pandoc --standalone -f html+raw_html -t html \
--standalone --embed-resources \
--template template.html \
--output mturk.html \
--metadata=title:HIT \
questions.html
```




## Repository Structure

- `Makefile`
    - Sets up the environment.
- `README.md`
- `data`
    - The `*.jsonl` files make up the VALUECONSISTENCY dataset.
    - The `*.csv` files are views of those `.jsonl`  files for use with mturk.

    - `without_manual_review`
        - Includes the original files before we manually reviewed all English translations of questions.
    - `validate`
        - `controversial`
            - `*validate.csv`
                - Various views of the data set to be used for annotators
            - `*.csv`
                - Results of validating how controversial each of the sets of questions for each topic were to 3 different annotators.
        - `paraphrase`
            - `*validate.csv`
                - Various views of the data set to be used for annotators
            - `*.csv`
                - Results of validating whether paraphrases of questions were deemed by bilingual annotators to be about the same thing. `src/utils.py:load_run` automatically excludes those deemed not to be equivalent.
- `notebooks`
    - `analysis.ipynb`
        - The main place to go when analyzing results of experiments.

    - `measures.py`
        - Definitions of various measures we use, such as the d-dimensional Shannon divergence.
    - `plots.py`
        - Functions to make plots
    - `utils.py`
        - Helper functions

    - `entropy_distance.ipynb`
        - For comparing the entropy and our measure, the d-dimensional Shannon divergence.
    - `robustness.ipynb`
        - For generating figures on the robustness of our measures, e.g. that models respond appropriately with first-token log probabilities.
    - `file_stats.ipynb`
        - For generating statistics on the VALUECONSISTENCY dataset.
    - `mturk_stats.ipynb`
        - For generating statistics on the results of the mturk experiments.
    - `process_mturk.ipynb`
        - For processing the output of mturk experiments into a common format.
    - `verify_question_answers.ipynb`
        - For reviewing the generating dataset and incorporating annotations.         
- `mturk`
    - `build_mturk.sh`
        - A handy executable to call Pandoc to build the relevant mturk files.
    - `hitpub.css`
        - Styling for the mturk survey
    - `main.js`
        - Code to handle the participant flow on Mturk.
    - `prevent_duplicates.py`
        - Spins up a job to prevent duplicate submissions on Mturk.
    - `questions.html`
        - The actual survey to administer on Mturk. 
    - `template.html`
        - The template structure with all of the includes for an Mturk external question HIT.
    - `timeme.js`
        - Times how long participants actually spend on each HIT.
- `scripts`
    - `format_model_prompted_topics.py`
        - The workhorse to generate the ValueConsistency dataset. See the [commands we used](commands.md)
    - `format_mturk.py`
        - Formats the ValueConsistency code into an Mturk-readable format.
    - `format_pvq.py`
        - Generates the data we used to run the Portrait Values Questionnaire experiment.
- `.gitignore`
- `requirements.txt`
- `environment.yml`
- `results`
    - This is where experiment results should be stored.
    - The format for files in this directory is `<data_filename>/<model_name>/<taskname_date>.json`
    - Each of these files contains a json object of both the metadata used for the run as well as the data itself. Use `load_run` to extract them.
    - The `without_manual_review` directory houses generated files before screening by expert annotations.
    - Download our results [here](https://drive.google.com/drive/folders/1SIduLOYD1YOhE8fdu6VuY2PMaeh31h3R?usp=sharing).
- `setup.py`
    - For setting up the source code as a package so it can be imported in any directory.
- `src`
    - `valuebank`
        - This is the main code for the project as a package.
        - `__init__.py`
            - Defines which functions can be imported from this package.
        - `common_prompts.py`
            - Various prompt engineering necessary to generate the dataset.
        - `distribution.py`
            - Defines a `Distribution` object useful in comparing probability distributions
        - `language_prompts.py`
            - Defines all prompts in the four languages we use.
        - `judge.py`
            - Prompting code for having a model judge which of input generations aligns most with a target generation---stance detection. See the commands at the top of the file.
        - `prompt.py`
            - The workhorse of the project. Takes in an input file (from `data`) and depending on the arguments prompts various models for various tasks. See the commands at the top of the file.
        - `query_models.py`
            - The backend for querying various types of models and aggregating their responses.
        - `utils.py`
            - Various shared functions.
