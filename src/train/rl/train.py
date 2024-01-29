from typing import List
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from readability import Readability
import numpy as np
import sys
eps = sys.float_info.epsilon
import math

import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
from trlx.models.modeling_ppo import PPOConfig

model_dir = 'checkpoints/exec-XXXX'  # select the checkpoint from the prompt-based methods

config = TRLConfig(
    train=TrainConfig(
        seq_length=1024,
        epochs=500,
        total_steps=100000,
        batch_size=2,
        batch_size_eval=2,
        checkpoint_interval=10000,
        eval_interval=500,
        save_optimizer=False,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        checkpoint_dir='checkpoint-diverse',
        save_best=True
    ),
    model=ModelConfig(
        model_path=model_dir,
        model_arch_type="seq2seq",
        num_layers_unfrozen=-1,
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path=model_dir,
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1.0e-5,
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 10000,
            "eta_min": 1.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        num_rollouts=512,
        chunk_size=4,
        ppo_epochs=4,
        init_kl_coef=0.05,
        target=6,
        horizon=10000,
        gamma=0.99,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1.0,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 256,
        },
        gen_experience_kwargs={
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95,
        },
    ),
)


def get_flesch_kincaid(text):
    r = Readability(text)
    fk = r.flesch_kincaid()
    return fk.score


def get_flesch(text):
    r = Readability(text)
    f = r.flesch()
    return f.score

import random

def change_scores(input_data):
    new_data = []
    for text in input_data:
        score_sum = random.choice([10, 15, 25, 30, 33, 35, 37, 40, 45, 48, 50, 52, 60, 64, 68, 70, 71, 75, 83, 84, 88, 89, 90, 92, 93, 94, 95])
        new_text = "Write highlights for this article with a flesch kincaid score of " + str(score_sum) + ":\n\n" + text
        new_data.append(new_text)
    return new_data

sigma = 10
def calc_nd(value, mean):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (value - mean) ** 2 / (2 * sigma ** 2)) / 0.039894228040143274


import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import os
model_name = "roberta-large"
device = "cuda:" + str(os.environ.get('LOCAL_RANK',0))
num_layers = 17
cache_dir=".cache"
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])

def encode_text(input_str):
    """
    Helper function to encode any string to tensor using the tokenizer
    """
    inputs = tokenizer(input_str, padding=True, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    # idf
    idf = torch.clone(inputs["attention_mask"]).float()
    idf[idf == tokenizer.sep_token_id] = 0
    idf[idf == tokenizer.cls_token_id] = 0
    idf.div_(idf.sum(dim=1, keepdim=True))

    return F.normalize(outputs[0], dim=-1), inputs["attention_mask"], idf

def compute_bertscore(doc_embedding, doc_masks, doc_idf, summ_embedding, summ_masks, summ_idf):
    """
    Helper function that is modified from the official code (greedy_cos_idf() method) https://github.com/Tiiiger/bert_score/blob/dbcf6db37e8bd6ff68446f06b0ba5d0763b62d20/bert_score/utils.py#L469
    """

    batch_size = doc_embedding.size(0)
    sim = torch.bmm(summ_embedding, doc_embedding.transpose(1, 2))
    masks = torch.bmm(summ_masks.unsqueeze(2).float(), doc_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

    masks = masks.float().to(sim.device)
    sim = sim * masks

    precision = sim.max(dim=2)[0]
    precision_scale = summ_idf.to(precision.device)
    P = (precision * precision_scale).sum(dim=1)

    summ_zero_mask = summ_masks.sum(dim=1).eq(2)
    if torch.any(summ_zero_mask):
        P = P.masked_fill(summ_zero_mask, 0.0)

    doc_zero_mask = doc_masks.sum(dim=1).eq(2)
    if torch.any(doc_zero_mask):
        P = P.masked_fill(doc_zero_mask, 0.0)

    return P


if __name__ == "__main__":

    def reward_fn(samples: List[str], prompts: List[str], outputs: List[str]):

        flesch_scores = []
        original_scores = []
        summaries = []
        docs = []
        for (generated_summary, input_doc) in zip(outputs, prompts):
            score_sum = int(input_doc.split("Write highlights for this article with a flesch kincaid score of ")[1][:2].replace(":", ""))
            original_scores.append(score_sum)
            doc = input_doc.split("Write highlights for this article with a flesch kincaid score of ")[1][2:]
            docs.append(doc)
            summaries.append(generated_summary.strip())

            try:
                flesch_scores.append(get_flesch(generated_summary.strip()))
            except:
                flesch_scores.append(0)

        all_bertscore_scores = []
        for doc, summary in zip(docs, summaries):

            bertscore_input_embedding, bertscore_input_attention_mask, bertscore_input_idf = encode_text([doc])
            bertscore_output_embedding, bertscore_output_attention_mask, bertscore_output_idf = encode_text([summary])

            bertscore_scores = compute_bertscore(
                bertscore_input_embedding,
                bertscore_input_attention_mask,
                bertscore_input_idf,
                bertscore_output_embedding,
                bertscore_output_attention_mask,
                bertscore_output_idf,
            )
            bertscore_scores = bertscore_scores.tolist()
            all_bertscore_scores.extend(bertscore_scores)

        assert len(original_scores) == len(flesch_scores) == len(all_bertscore_scores)

        flesch_scores = [calc_nd(fs, o_fs) for fs, o_fs in zip(flesch_scores, original_scores)]

        readability_weight = 1
        flesch_scores = torch.tensor(flesch_scores)
        all_bertscore_scores = torch.tensor(all_bertscore_scores)
        flesch_scores = readability_weight * flesch_scores + (1 - readability_weight) * all_bertscore_scores
        flesch_scores = flesch_scores.tolist()

        return flesch_scores


    train_file = '../../data/train_prompt_score.json'
    validation_file = '../../data/train_prompt_score.json'
    data_files = {"train": train_file, "validation": validation_file}
    dataset = load_dataset("json", data_files=data_files)
    dataset['train'] = dataset['train'].shuffle(seed=42)
    dataset['validation'] = dataset['validation'].shuffle(seed=42)

    validation_examples = 2000
    val_prompts = [prompt for prompt in dataset['validation']["input_noprompt"][0:validation_examples]]
    print('\ntest 0\n', val_prompts[0])
    val_summaries = dataset['validation']["summary"][0:validation_examples]
    val_prompts = change_scores(val_prompts)
    assert len(val_prompts) == len(val_summaries)
    print('\ntest after 0 \n', val_prompts[0])

    prompts = dataset['train']["input_noprompt"]
    summaries = dataset['train']["summary"]
    prompts = [prompt for prompt in prompts]
    prompts = change_scores(prompts)
    assert len(prompts) == len(summaries)

    # make dictionary of prompts and labels to use for reward function
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_path)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    tokenizer.sep_token = "<sep>"
    prompt_label = {}
    max_length = config.train.seq_length

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=val_prompts,
        config=config,
    )
