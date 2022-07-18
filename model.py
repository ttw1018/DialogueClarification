import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPT2PreTrainedModel


def get_tokenizer(model_name_or_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
    return tokenizer

def get_model(model_name_or_path):
    config = GPT2Config.from_pretrained(model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path, config=config)
    return model