#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate readability_summ

VAL_FILE='../data/test_prompt_score.json'
MODEL_PATH=$1


OUTPUT_DIR='outputs/1/'
CUDA_VISIBLE_DEVICES=0 python -u run_summarization.py --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} --text_column input_noprompt --summary_column summary \
 --train_file ${VAL_FILE} \
 --validation_file ${VAL_FILE} \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --source_prefix "Write highlights for this article with a flesch kincaid score of 90:\n\n" \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 16 \
 --predict_with_generate \
 --do_predict &

P1=$!


OUTPUT_DIR='outputs/2/'
CUDA_VISIBLE_DEVICES=1 python -u run_summarization.py --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} --text_column input_noprompt --summary_column summary \
 --train_file ${VAL_FILE} \
 --validation_file ${VAL_FILE} \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --source_prefix "Write highlights for this article with a flesch kincaid score of 70:\n\n" \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 16 \
 --predict_with_generate \
 --do_predict &

P2=$!


OUTPUT_DIR='outputs/3/'
CUDA_VISIBLE_DEVICES=2 python -u run_summarization.py --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} --text_column input_noprompt --summary_column summary \
 --train_file ${VAL_FILE} \
 --validation_file ${VAL_FILE} \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --source_prefix "Write highlights for this article with a flesch kincaid score of 50:\n\n" \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 16 \
 --predict_with_generate \
 --do_predict &

P3=$!


OUTPUT_DIR='outputs/4/'
CUDA_VISIBLE_DEVICES=3 python -u run_summarization.py --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} --text_column input_noprompt --summary_column summary \
 --train_file ${VAL_FILE} \
 --validation_file ${VAL_FILE} \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --source_prefix "Write highlights for this article with a flesch kincaid score of 30:\n\n" \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 16 \
 --predict_with_generate \
 --do_predict &

P4=$!

wait $P1 $P2 $P3 $P4

conda deactivate