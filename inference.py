import json
import torch
def clarification(model, tokenizer, context):
    break_token = tokenizer.encode(str(tokenizer._eos_token))
    token = tokenizer.encode(context)
    predict_token = token[-1]
    while predict_token != break_token:
        predict_token = torch.argmax(model(torch.tensor(token))[0][-1, :]).item()
        print(tokenizer.decode([predict_token]))
        token.append(predict_token)