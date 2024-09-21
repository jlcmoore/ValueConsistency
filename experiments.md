# Commands to Query Different Models on ValueConsistency

## Classification jobs

Various commands to query a variety of models with the ValueConsistency dataset.

### Open-weight models using `vllm`

```
prompt --output-directory /results --model meta-llama/Llama-2-70b-hf --endpoint completion --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --example-answer 
prompt --output-directory /results --model meta-llama/Llama-2-70b-hf --endpoint completion --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --example-answer 
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/uncontroversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_chinese_us.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_english_china.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_japanese_us.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_english_japan.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_german_us.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_german_germany.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_german_germany.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_english_germany.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model 01-ai/Yi-34B --endpoint completion --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt --example-answer 
prompt --output-directory /results --model 01-ai/Yi-34B-Chat --endpoint chat --source vllm --filename data/uncontroversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt
prompt --output-directory /results --model 01-ai/Yi-34B-Chat --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model 01-ai/Yi-34B-Chat --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model 01-ai/Yi-34B-Chat --endpoint chat --source vllm --filename data/controversial_chinese_us.jsonl --query-language chinese --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model 01-ai/Yi-34B-Chat --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model 01-ai/Yi-34B-Chat --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model 01-ai/Yi-34B-Chat --endpoint chat --source vllm --filename data/controversial_english_china.jsonl --query-language english --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model stabilityai/japanese-stablelm-instruct-beta-70b --endpoint chat --source vllm --filename data/uncontroversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt --example-answer
prompt --output-directory /results --model stabilityai/japanese-stablelm-instruct-beta-70b --endpoint chat --source vllm --filename data/controversial_japanese_us.jsonl --query-language japanese --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt --example-answer 
prompt --output-directory /results --model stabilityai/japanese-stablelm-instruct-beta-70b --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt --example-answer --allow-abstentions
prompt --output-directory /results --model stabilityai/japanese-stablelm-instruct-beta-70b --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt --example-answer 
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/uncontroversial_german_germany.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_chinese_us.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_english_china.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_japanese_us.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_english_japan.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_german_us.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_german_germany.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_german_germany.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_english_germany.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B --endpoint completion --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --example-answer 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/uncontroversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_chinese_us.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_english_china.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_japanese_us.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_english_japan.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_german_us.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_german_germany.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_german_germany.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_english_germany.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
```

smaller models after review

```
prompt --output-directory /results --model meta-llama/Meta-Llama-3-8B --endpoint completion --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --example-answer  
prompt --output-directory /results --model meta-llama/Meta-Llama-3-8B-Instruct --endpoint chat --source vllm --filename data/uncontroversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-8B-Instruct --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-8B-Instruct --endpoint chat --source vllm --filename data/controversial_chinese_us.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-8B-Instruct --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-8B-Instruct --endpoint chat --source vllm --filename data/controversial_english_china.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-8B-Instruct --endpoint chat --source vllm --filename data/controversial_japanese_us.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-8B-Instruct --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-8B-Instruct --endpoint chat --source vllm --filename data/controversial_english_japan.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-8B-Instruct --endpoint chat --source vllm --filename data/controversial_german_us.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-8B-Instruct --endpoint chat --source vllm --filename data/controversial_german_germany.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Meta-Llama-3-8B-Instruct --endpoint chat --source vllm --filename data/controversial_english_germany.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt
prompt --output-directory /results --model meta-llama/Meta-Llama-3-8B-Instruct --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions


prompt --output-directory /results --model meta-llama/Llama-2-7b-hf --endpoint completion --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --example-answer 
prompt --output-directory /results --model meta-llama/Llama-2-7b-hf --endpoint completion --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --example-answer 
prompt --output-directory /results --model meta-llama/Llama-2-7b-chat-hf --endpoint chat --source vllm --filename data/uncontroversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt
prompt --output-directory /results --model meta-llama/Llama-2-7b-chat-hf --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-7b-chat-hf --endpoint chat --source vllm --filename data/controversial_chinese_us.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-7b-chat-hf --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-7b-chat-hf --endpoint chat --source vllm --filename data/controversial_english_china.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-7b-chat-hf --endpoint chat --source vllm --filename data/controversial_japanese_us.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-7b-chat-hf --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-7b-chat-hf --endpoint chat --source vllm --filename data/controversial_english_japan.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-7b-chat-hf --endpoint chat --source vllm --filename data/controversial_german_us.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-7b-chat-hf --endpoint chat --source vllm --filename data/controversial_german_germany.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
prompt --output-directory /results --model meta-llama/Llama-2-7b-chat-hf --endpoint chat --source vllm --filename data/controversial_english_germany.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt
prompt --output-directory /results --model meta-llama/Llama-2-7b-chat-hf --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions
```

### Closed models

```
prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/uncontroversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_english_us.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_chinese_us.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_chinese_china.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_english_china.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_japanese_us.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_japanese_japan.jsonl --query-language japanese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_english_japan.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_german_us.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_german_germany.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --allow-abstentions

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_german_germany.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_english_germany.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt 
```

## generation jobs

Various commands to query a variety of models with the ValueConsistency dataset, like above

### Prompting models just to respond

```
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task generation
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task generation
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task generation
prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/controversial_german_germany.jsonl --query-language german --task generation
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task generation
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task generation
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task generation
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/controversial_german_germany.jsonl --query-language german --task generation
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task generation
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task generation
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task generation
prompt --output-directory /results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/controversial_german_germany.jsonl --query-language german --task generation
prompt --output-directory /results --model 01-ai/Yi-34B-Chat --endpoint chat --source vllm --filename data/controversial_english_us.jsonl --query-language english --task generation
prompt --output-directory /results --model 01-ai/Yi-34B-Chat --endpoint chat --source vllm --filename data/controversial_chinese_china.jsonl --query-language chinese --task generation
prompt --output-directory /results --model stabilityai/japanese-stablelm-instruct-beta-70b --endpoint chat --source vllm --filename data/controversial_japanese_japan.jsonl --query-language japanese --task generation

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_english_us.jsonl --query-language english --task generation

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_chinese_china.jsonl --query-language chinese --task generation

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_japanese_japan.jsonl --query-language japanese --task generation

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/controversial_german_germany.jsonl --query-language german --task generation
```

### Judging the stances of the models

#### Using Claude to judge stances, allowing Claude to abstain

```
judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_chinese_china/01-ai_Yi-34B-Chat/generation-2024-04-25.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_english_us/01-ai_Yi-34B-Chat/generation-2024-04-25.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_chinese_china/meta-llama_Llama-2-70b-chat-hf/generation-2024-04-25.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_english_us/meta-llama_Llama-2-70b-chat-hf/generation-2024-04-24.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_german_germany/meta-llama_Llama-2-70b-chat-hf/generation-2024-05-11.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/meta-llama_Llama-2-70b-chat-hf/generation-2024-04-25.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/stabilityai_japanese-stablelm-instruct-beta-70b/generation-2024-04-25.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_english_us/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_chinese_china/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_german_germany/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_english_us/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_german_germany/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_chinese_china/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json
```


todo -- ran out of credits for now...
```
judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions  --filename results/controversial_english_us/gpt-4o/generation-2024-05-22.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions  --filename results/controversial_german_germany/gpt-4o/generation-2024-05-22.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions  --filename results/controversial_chinese_china/gpt-4o/generation-2024-05-22.json

judge --output-directory results --randomize-option-order --model claude-3-opus-20240229 --source anthropic --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/gpt-4o/generation-2024-05-22.json
```

#### Using llama3 to judge stances, not allowing llama3 to abstain

```
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_chinese_china/01-ai_Yi-34B-Chat/generation-2024-04-25.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_english_us/01-ai_Yi-34B-Chat/generation-2024-04-25.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_chinese_china/meta-llama_Llama-2-70b-chat-hf/generation-2024-04-25.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_english_us/meta-llama_Llama-2-70b-chat-hf/generation-2024-04-24.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_german_germany/meta-llama_Llama-2-70b-chat-hf/generation-2024-05-11.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_japanese_japan/meta-llama_Llama-2-70b-chat-hf/generation-2024-04-25.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_japanese_japan/stabilityai_japanese-stablelm-instruct-beta-70b/generation-2024-04-25.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_english_us/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_german_germany/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_chinese_china/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_japanese_japan/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_english_us/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_german_germany/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_chinese_china/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_japanese_japan/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json
```

todo
```
judge --output-directory results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_english_us/gpt-4o/generation-2024-05-22.json

judge --output-directory results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_german_germany/gpt-4o/generation-2024-05-22.json

judge --output-directory results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_chinese_china/gpt-4o/generation-2024-05-22.json

judge --output-directory results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --filename results/controversial_japanese_japan/gpt-4o/generation-2024-05-22.json
```


#### Using llama3 to judge stances, allowing llama3 to abstain

```
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_chinese_china/01-ai_Yi-34B-Chat/generation-2024-04-25.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_english_us/01-ai_Yi-34B-Chat/generation-2024-04-25.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_chinese_china/meta-llama_Llama-2-70b-chat-hf/generation-2024-04-25.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_english_us/meta-llama_Llama-2-70b-chat-hf/generation-2024-04-24.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_german_germany/meta-llama_Llama-2-70b-chat-hf/generation-2024-05-11.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/meta-llama_Llama-2-70b-chat-hf/generation-2024-04-25.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/stabilityai_japanese-stablelm-instruct-beta-70b/generation-2024-04-25.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_english_us/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_german_germany/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_chinese_china/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_english_us/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_german_germany/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_chinese_china/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json
judge --output-directory /results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json

judge --output-directory results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions  --filename results/controversial_english_us/gpt-4o/generation-2024-05-22.json

judge --output-directory results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions  --filename results/controversial_german_germany/gpt-4o/generation-2024-05-22.json

judge --output-directory results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions  --filename results/controversial_chinese_china/gpt-4o/generation-2024-05-22.json

judge --output-directory results --randomize-option-order --model meta-llama/Meta-Llama-3-70B-Instruct --source vllm --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/gpt-4o/generation-2024-05-22.json
```

#### Using gpt-4o to judge stances, allowing gpt-4o to abstain

```
judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_chinese_china/meta-llama_Llama-2-70b-chat-hf/generation-2024-04-25.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_chinese_china/01-ai_Yi-34B-Chat/generation-2024-04-25.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_english_us/01-ai_Yi-34B-Chat/generation-2024-04-25.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_english_us/meta-llama_Llama-2-70b-chat-hf/generation-2024-04-24.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_german_germany/meta-llama_Llama-2-70b-chat-hf/generation-2024-05-11.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/meta-llama_Llama-2-70b-chat-hf/generation-2024-04-25.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/stabilityai_japanese-stablelm-instruct-beta-70b/generation-2024-04-25.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_english_us/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_chinese_china/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_german_germany/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/CohereForAI_c4ai-command-r-v01/generation-2024-05-12.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_english_us/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_german_germany/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_chinese_china/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json

judge --output-directory results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/meta-llama_Meta-Llama-3-70B-Instruct/generation-2024-05-13.json

judge --output-directory /results --randomize-option-order --model gpt-4o --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_english_us/gpt-4o/generation-2024-05-22.json

judge --output-directory /results --randomize-option-order --model gpt-4o  --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_german_germany/gpt-4o/generation-2024-05-22.json

judge --output-directory /results --randomize-option-order --model gpt-4o  --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_chinese_china/gpt-4o/generation-2024-05-22.json

judge --output-directory /results --randomize-option-order --model gpt-4o  --source openai --no-follow-up --system-prompt --single-letter-prompt --stance --allow-abstentions --filename results/controversial_japanese_japan/gpt-4o/generation-2024-05-22.json
```

## Portait Values Questionnaire Experiment

### Making the data to run the experiment

```
python scripts/format_pvq.py --query-language german --filename data/pvq_german.jsonl

python scripts/format_pvq.py --query-language english --filename data/pvq_german.jsonl

python scripts/format_pvq.py --query-language chinese --filename data/pvq_chinese.jsonl
```

### Running the experiment

```
prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B --endpoint chat --source vllm --filename data/pvq_chinese.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B --endpoint chat --source vllm --filename data/pvq_german.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory /results --model meta-llama/Llama-2-70b-hf --endpoint chat --source vllm --filename data/pvq_chinese.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory /results --model meta-llama/Llama-2-70b-hf --endpoint chat --source vllm --filename data/pvq_german.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory /results --model 01-ai/Yi-34B --endpoint chat --source vllm --filename data/pvq_english.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory /results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/pvq_english.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/pvq_chinese.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory results --model meta-llama/Meta-Llama-3-70B-Instruct --endpoint chat --source vllm --filename data/pvq_german.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory results --model meta-llama/Meta-Llama-3-70B --endpoint chat --source vllm --filename data/pvq_english.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --example-answer --use-values --num-values -1 --static-value

prompt --output-directory results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/pvq_english.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/pvq_chinese.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory results --model CohereForAI/c4ai-command-r-v01 --endpoint chat --source vllm --filename data/pvq_german.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/pvq_english.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/pvq_chinese.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory results --model gpt-4o --endpoint chat --source openai --filename data/pvq_german.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/pvq_english.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/pvq_chinese.jsonl --query-language chinese --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory /results --model meta-llama/Llama-2-70b-chat-hf --endpoint chat --source vllm --filename data/pvq_german.jsonl --query-language german --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory /results --model meta-llama/Llama-2-70b-hf --endpoint chat --source vllm --filename data/pvq_english.jsonl --query-language english --task classification --randomize-option-order --no-follow-up --system-prompt --single-letter-prompt --example-answer --use-values --num-values -1 --static-value


prompt --output-directory /results --model 01-ai/Yi-34B-Chat --endpoint chat --source vllm --filename data/pvq_english.jsonl --query-language english --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory /results --model 01-ai/Yi-34B-Chat --endpoint chat --source vllm --filename data/pvq_chinese.jsonl --query-language chinese --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt --use-values --num-values -1 --static-value

prompt --output-directory /results --model 01-ai/Yi-34B --endpoint chat --source vllm --filename data/pvq_chinese.jsonl --query-language chinese --task classification --randomize-option-order --follow-up --system-prompt --single-letter-prompt --example-answer --use-values --num-values -1 --static-value
```

