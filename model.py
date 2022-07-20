import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPT2PreTrainedModel
import pathlib


def get_tokenizer(model_name_or_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    return tokenizer

def get_model(model_name_or_path):
    config = GPT2Config.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config)
    return model


def save_model(model, tokenizer, gloab_step, accelerator):
    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    path = f"./output/global_step-{gloab_step}/"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)