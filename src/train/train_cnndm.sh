#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate readability_summ

FOLDER_OUTPUT=/mnt/hd3/checkpoints/exec-$RANDOM

TRAIN_FILE='../data/train_prompt_category.json'
VAL_FILE='../data/validation_prompt_category.json'

MODEL_NAME='google/flan-t5-large'

deepspeed --master_port 61002 --include localhost:0,1,2,3,4,5,6,7 run_summarization.py --model_name_or_path ${MODEL_NAME} \
 --output_dir ${FOLDER_OUTPUT} --text_column input --summary_column summary \
 --train_file ${TRAIN_FILE} \
 --validation_file ${VAL_FILE} \
 --learning_rate 1e-4 \
 --max_source_length 1024 \
 --source_prefix "" \
 --num_train_epochs 20 \
 --logging_steps 200 \
 --preprocessing_num_workers 100 \
 --eval_steps 10000 \
 --save_steps 10000 \
 --save_total_limit 2 \
 --evaluation_strategy "steps" \
 --per_device_train_batch_size 4 \
 --per_device_eval_batch_size 4 \
 --metric_for_best_model "rouge1" \
 --load_best_model_at_end \
 --predict_with_generate \
 --deepspeed ds_config_stage3_fb16.json \
 --bf16 \
 --bf16_full_eval \
 --do_train

conda deactivate