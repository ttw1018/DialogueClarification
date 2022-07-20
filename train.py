import torch
from torch import nn
from model import *
from utils.preprocess import get_data_loader, TaskDataset, preprocess_task1
from tqdm import tqdm
from torch.optim import Adam
import accelerate

def train_epoch(model, tokenizer, optimizer, loader, global_step, accelerator, device):
    for x in tqdm(loader, desc='train'):
        input, label = x[0], x[1]
        input = input.to(device)
        label = label.to(device)
        output = model(input, labels=input)
        loss = output[0]
        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        global_step = global_step + 1
        if global_step % 10000 == 0:
            save_model(model, tokenizer, global_step, accelerator)
    return model, optimizer, global_step

def main(model_name_or_path):
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    epoches = 10 
    # lr = 0.00001
    lr = 3e-5
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(model_name_or_path)
    tokenizer = get_tokenizer(model_name_or_path)
    data = preprocess_task1('train')
    dataset = TaskDataset(data, tokenizer)
    loader = get_data_loader(dataset, tokenizer, batch_size=4)
    optimizer = Adam(model.parameters(), lr=lr)
    model = model.to(device)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    global_step = 0
    for epoch in tqdm(iterable=range(epoches), desc="epoch"):
        model, optimizer, global_step = train_epoch(model, tokenizer, optimizer, loader, global_step, accelerator, device)

if __name__ == "__main__":
    # main("/Users/tianwentang/models/gpt2")
    main("/data2/twtang/models/gpt2")