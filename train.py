import torch
from torch import nn
from model import *
from utils.preprocess import preprocess, get_data_loader, TaskDataset
from tqdm import tqdm
from torch.optim import Adam
from inference import *

def train_epoch(model, tokenizer, optimizer, loader, global_step, device):
    for x in tqdm(loader, desc='train'):
        input, label = (x, x)
        input = input.to(device)
        label = label.to(device)
        output = model(input, labels=label)
        loss = output[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step = global_step + 1
        if global_step % 1000 == 0:
            save_model(model, tokenizer, global_step)
    return model, optimizer, global_step

def main(model_name_or_path):
    epoches = 3
    lr = 0.00001
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(model_name_or_path)
    tokenizer = get_tokenizer(model_name_or_path)
    data = preprocess()
    dataset = TaskDataset(data, tokenizer)
    loader = get_data_loader(dataset, tokenizer, batch_size=8)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    global_step = 0
    for epoch in tqdm(iterable=range(epoches), desc="epoch"):
        model, optimizer, global_step = train_epoch(model, tokenizer, optimizer, loader, global_step, device)

if __name__ == "__main__":
    main("/Users/tianwentang/models/gpt2")
    model_name_or_path = "/Users/tianwentang/models/gpt2"
    sents = preprocess()
    model = get_model(model_name_or_path)
    tokenize = get_tokenizer(model_name_or_path)
    clarification(model, tokenize, sents[0])