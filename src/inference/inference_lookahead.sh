#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate readability_summ

LOOKAHEAD_LENGTH=20
DOC_FILE='../data/test_prompt_category.json'
MODEL_PATH=$1


PROMPT="Write highlights for this article for a 11 years old student:\n\n"
OUTPUT_FILE="11yold.txt"
SCORE=90
CUDA_VISIBLE_DEVICES=0 python run_lookahead.py --document_file ${DOC_FILE} --output_file ${OUTPUT_FILE} --do_lookahead --lookahead_decoding_type greedy --model_name ${MODEL_PATH} --lookahead_length ${LOOKAHEAD_LENGTH} \
 --prompt "${PROMPT}" --score ${SCORE} &
P1=$!


PROMPT="Write highlights for this article for a middle school student:\n\n"
OUTPUT_FILE="middle-school.txt"
SCORE=70
CUDA_VISIBLE_DEVICES=1 python run_lookahead.py --document_file ${DOC_FILE} --output_file ${OUTPUT_FILE} --do_lookahead --lookahead_decoding_type greedy --model_name ${MODEL_PATH} --lookahead_length ${LOOKAHEAD_LENGTH} \
 --prompt "${PROMPT}" --score ${SCORE} &
P2=$!


PROMPT="Write highlights for this article for a high school student:\n\n"
OUTPUT_FILE="high-school.txt"
SCORE=50
CUDA_VISIBLE_DEVICES=2 python run_lookahead.py --document_file ${DOC_FILE} --output_file ${OUTPUT_FILE} --do_lookahead --lookahead_decoding_type greedy --model_name ${MODEL_PATH} --lookahead_length ${LOOKAHEAD_LENGTH} \
 --prompt "${PROMPT}" --score ${SCORE} &
P3=$!


PROMPT="Write highlights for this article for a college student:\n\n"
OUTPUT_FILE="college-student.txt"
SCORE=30
CUDA_VISIBLE_DEVICES=3 python run_lookahead.py --document_file ${DOC_FILE} --output_file ${OUTPUT_FILE} --do_lookahead --lookahead_decoding_type greedy --model_name ${MODEL_PATH} --lookahead_length ${LOOKAHEAD_LENGTH} \
 --prompt "${PROMPT}" --score ${SCORE} &
P4=$!

wait $P1 $P2 $P3 $P4

conda deactivate