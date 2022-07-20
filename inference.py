import json
import torch
import pdb
from utils.preprocess import *
from utils.dataset import *
from model import *

def clarification_generation(model, tokenizer, context):
    break_token = tokenizer.encode(str(tokenizer._eos_token))
    token = tokenizer.encode(context)
    predict_token = token[-1]
    ans = []
    cnt = 0
    print('\n')
    while predict_token != break_token and cnt < 30:
        print(torch.tensor(token).shape, end = ' ')
        output = model(torch.tensor(token).unsqueeze(0))[0]
        print(output.shape)
        predict_token = torch.argmax(model(torch.tensor(token))[0][-1, :]).item()
        token.append(predict_token)
        ans.append(predict_token)
        print('\r', tokenizer.decode(ans), end='', flush=True)
        cnt = cnt + 1

def entity_predict(model, tokenizer, context):
    pass

if __name__ == "__main__":
    model_name_or_path = "./output/global_step-20000"
    model = get_model(model_name_or_path)
    print(model)
    tokenizer = get_tokenizer(model_name_or_path)

    # data = TaskDataset(preprocess_task1('dev'), tokenizer)
    # loader = get_data_loader(data, tokenizer, 1)
    data = preprocess_task1('dev')

    for tokens in data:
        clarification_generation(model, tokenizer, tokens[0])