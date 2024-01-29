#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate readability_summ

VAL_FILE='../data/test_prompt_category.json'
MODEL_PATH=$1


OUTPUT_DIR='outputs/1/'
CUDA_VISIBLE_DEVICES=4 python -u run_summarization.py --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} --text_column input_noprompt --summary_column summary \
 --train_file ${VAL_FILE} \
 --validation_file ${VAL_FILE} \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 256 \
 --max_target_length 256 \
 --generation_max_length 256 \
 --num_beams 3 \
 --source_prefix "Write highlights for this article for a 11 years old student:\n\n" \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 16 \
 --predict_with_generate \
 --do_predict &

P1=$!


OUTPUT_DIR='outputs/2/'
CUDA_VISIBLE_DEVICES=5 python -u run_summarization.py --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} --text_column input_noprompt --summary_column summary \
 --train_file ${VAL_FILE} \
 --validation_file ${VAL_FILE} \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 256 \
 --max_target_length 256 \
 --generation_max_length 256 \
 --num_beams 3 \
 --source_prefix "Write highlights for this article for a middle school student:\n\n" \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 16 \
 --predict_with_generate \
 --do_predict &

P2=$!


OUTPUT_DIR='outputs/3/'
CUDA_VISIBLE_DEVICES=6 python -u run_summarization.py --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} --text_column input_noprompt --summary_column summary \
 --train_file ${VAL_FILE} \
 --validation_file ${VAL_FILE} \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 256 \
 --max_target_length 256 \
 --generation_max_length 256 \
 --num_beams 3 \
 --source_prefix "Write highlights for this article for a high school student:\n\n" \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 16 \
 --predict_with_generate \
 --do_predict &

P3=$!


OUTPUT_DIR='outputs/4/'
CUDA_VISIBLE_DEVICES=7 python -u run_summarization.py --model_name_or_path ${MODEL_PATH} \
 --output_dir ${OUTPUT_DIR} --text_column input_noprompt --summary_column summary \
 --train_file ${VAL_FILE} \
 --validation_file ${VAL_FILE} \
 --test_file ${VAL_FILE} \
 --max_source_length 1024 \
 --val_max_target_length 256 \
 --max_target_length 256 \
 --generation_max_length 256 \
 --num_beams 3 \
 --source_prefix "Write highlights for this article for a college student:\n\n" \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 1 \
 --per_device_eval_batch_size 16 \
 --predict_with_generate \
 --do_predict &

P4=$!

wait $P1 $P2 $P3 $P4

conda deactivate