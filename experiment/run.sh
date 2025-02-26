#!/bin/bash

python download_data.py

python segment.py \
    --load_path data/mtbp/mtbp.jsonl \
    --save_path result/mtbp/gpt4seg_mtbp.jsonl

python compress.py \
    --load_path result/mtbp/gpt4seg_mtbp.jsonl \
    --save_path result/mtbp/llmlingua2comp_mtbp.jsonl

granularity_list=(segment session turn)
topk_list=(1 1 3)
length=${#topk_list[@]}
for ((i=0; i<length; i++)); do
    granularity=${granularity_list[i]}
    topk=${topk_list[i]}
    echo "granularity: $granularity, topk: $topk"
    python retrieve.py \
        --load_path result/mtbp/llmlingua2comp_mtbp.jsonl \
        --secom_config_path ../secom/configs/bm25.yaml \
        --granularity ${granularity} \
        --topk ${topk} \
        --save_path result/mtbp/retrieval/bm25/${granularity}-k${topk}_mtbp.jsonl
    python chat.py \
        --load_path result/mtbp/retrieval/bm25/${granularity}-k${topk}_mtbp.jsonl \
        --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \
        --save_path result/mtbp/retrieval/bm25/mistral/${granularity}-k${topk}_mtbp.jsonl
    python chat.py \
        --load_path result/mtbp/retrieval/bm25/${granularity}-k${topk}_mtbp.jsonl \
        --model_name_or_path openai/gpt-3.5-turbo-0125 \
        --save_path result/mtbp/retrieval/bm25/gpt-3.5-turbo-0125/${granularity}-k${topk}_mtbp.jsonl
done

granularity_list=(segment session turn)
topk_list=(1 1 3)
length=${#topk_list[@]}
for ((i=0; i<length; i++)); do
    granularity=${granularity_list[i]}
    topk=${topk_list[i]}
    echo "granularity: $granularity, topk: $topk"
    python retrieve.py \
        --load_path result/mtbp/llmlingua2comp_mtbp.jsonl \
        --secom_config_path ../secom/configs/mpnet.yaml \
        --granularity ${granularity} \
        --topk ${topk} \
        --save_path result/mtbp/retrieval/mpnet/${granularity}-k${topk}_mtbp.jsonl
    python chat.py \
        --load_path result/mtbp/retrieval/mpnet/${granularity}-k${topk}_mtbp.jsonl \
        --model_name_or_path mistralai/Mistral-7B-Instruct-v0.3 \
        --save_path result/mtbp/retrieval/mpnet/mistral/${granularity}-k${topk}_mtbp.jsonl
    python chat.py \
        --load_path result/mtbp/retrieval/mpnet/${granularity}-k${topk}_mtbp.jsonl \
        --model_name_or_path openai/gpt-3.5-turbo-0125 \
        --save_path result/mtbp/retrieval/mpnet/gpt-3.5-turbo-0125/${granularity}-k${topk}_mtbp.jsonl
done
