
# Generating ValueConsistency

Various commands for creating the ValueConsistency data set.

#### For the U.S.

```
python scripts/format_model_prompted_topics.py \
--filename controversial_english_us.jsonl \
--query-language english \
--annotator gpt-4 \
new \
--country us \
--num-topics 30 \
--num-questions 5 \
--num-rephrases 4
```

#### Japan
```
python scripts/format_model_prompted_topics.py \
--filename controversial_japanese_japan.jsonl \
--query-language japanese \
--annotator gpt-4 \
new \
--country japan \
--num-topics 30 \
--num-questions 5 \
--num-rephrases 4
```

#### China

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
```

#### Germany

```
python scripts/format_model_prompted_topics.py \
--filename controversial_german_germany.jsonl \
--query-language german \
--annotator gpt-4 \
new \
--country germany \
--num-topics 30 \
--num-questions 5 \
--num-rephrases 4
```

### To translate the generated data from the original country to other languages

Like this: 

##### China to English
```
python  scripts/format_model_prompted_topics.py \
--filename data/controversial_english_china.jsonl \
--query-language english \
--annotator gpt-4 \
translate \
--reference-file data/controversial_chinese_china.jsonl
```

##### Japan to English
```
python  scripts/format_model_prompted_topics.py \
--filename data/controversial_english_japan.jsonl \
--query-language english \
--annotator gpt-4 \
translate \
--reference-file data/controversial_japanese_japan.jsonl
```

##### Germany to English
```
python  scripts/format_model_prompted_topics.py \
--filename data/controversial_english_germany.jsonl \
--query-language english \
--annotator gpt-4 \
translate \
--reference-file data/controversial_german_germany.jsonl
```

##### U.S. to Chinese
```
python  scripts/format_model_prompted_topics.py \
--filename data/controversial_chinese_us.jsonl \
--query-language chinese \
--annotator gpt-4 \
translate \
--reference-file data/controversial_english_us.jsonl
```

##### U.S. to German
```
python  scripts/format_model_prompted_topics.py \
--filename data/controversial_german_us.jsonl \
--query-language german \
--annotator gpt-4 \
translate \
--reference-file data/controversial_english_us.jsonl
```

##### U.S. to Japanese
```
python  scripts/format_model_prompted_topics.py \
--filename data/controversial_japanese_us.jsonl \
--query-language japanese \
--annotator gpt-4 \
translate \
--reference-file data/controversial_english_us.jsonl
```

### For the same as the above but now uncontroversial topics

#### Chinese

```
python scripts/format_model_prompted_topics.py \
    --filename data/uncontroversial_chinese_china.jsonl \
    --query-language chinese \
    --annotator gpt-4 \
    --no-controversial \
    new \
    --country china \
    --num-topics 30 \
    --num-questions 5 \
    --num-rephrases 4
```

#### German

```
python scripts/format_model_prompted_topics.py \
    --filename data/uncontroversial_german_germany.jsonl \
    --query-language german \
    --annotator gpt-4 \
    --no-controversial \
    new \
    --country germany \
    --num-topics 30 \
    --num-questions 5 \
    --num-rephrases 4
```


#### English

```
python scripts/format_model_prompted_topics.py \
    --filename data/uncontroversial_english_us.jsonl \
    --query-language english \
    --annotator gpt-4 \
    --no-controversial \
    new \
    --country us \
    --num-topics 30 \
    --num-questions 5 \
    --num-rephrases 4
```

#### Japanese

```
python scripts/format_model_prompted_topics.py \
    --filename data/uncontroversial_japanese_japan.jsonl \
    --query-language japanese \
    --annotator gpt-4 \
    --no-controversial \
    new \
    --country japan \
    --num-topics 30 \
    --num-questions 5 \
    --num-rephrases 4
```

## Validation

### Mturk validation of paraphrases

```
python scripts/format_mturk.py --filename data/controversial_english_us.jsonl --kind rephrase --validate --all-questions
python scripts/format_mturk.py --filename data/controversial_german_germany.jsonl --kind rephrase --validate --all-questions
python scripts/format_mturk.py --filename data/controversial_japanese_japan.jsonl --kind rephrase --validate --all-questions
python scripts/format_mturk.py --filename data/controversial_chinese_china.jsonl --kind rephrase --validate --all-questions

```
