from utils.dataset import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import pathlib
import json

def process1(file_name):
    sents = []
    fin = open(file_name, "r")
    for i in fin:
        dia = json.loads(i)
        sent = '<|context|>' + dia['context'] + '<|endofcontext|>'
        sent += '<|entity1|>' + dia['entity1'] + '<|endofentity1|>'
        sent += '<|entity2|>' + dia['entity2'] + '<|endofentity2|>'
        sent += '<|notneedclarify|>' if dia['label'] == '1' else '<|needclarify|>'
        sents.append(sent)
    return sents

def process2():
    pass

def process3(file_name, file_type='train', refresh=False):
    sents = []
    fin = open(file_name, 'r')
    for i in fin:
        dia = json.loads(i)
        contexts = dia['context'].split("<EOS>")
        if len(contexts) <= 3:
            raise ValueError("dialogue turns should over 3 turns!!")
        context = ""
        for n in range(len(contexts) - 2):
            if n % 2 == 0:
                context = '<|user|>' + contexts[n]
            else:
                context = '<|system|>' + contexts[n]
        sent = '<|context|>' + context + '<|endofcontext|>'
        sent += '<|entity1|>' + dia['entity1'] + '<|endofentity1|>'
        sent += '<|entity2|>' + dia['entity2'] + '<|endofentity2|>'
        sent += '<|notneedclarify|>'
        sent += '<|response|>' + contexts[-2] + '<|endofresponse|>'
        sent += '<|clarifyuserinput|>' + contexts[-1] + '<|endofclarifyuserinput|>'
        sents.append(sent)
    return sents

def preprocess():
    sents = process1('./data/multi-turn/task1/train_classifier.txt')
    sents.extend(process3('./data/multi-turn/task3/train_sml.txt'))
    return sents

if __name__ == '__main__':
    data = process3('./data/multi-turn/task3/dev_sml.txt', refresh=True)
    tokenizer = GPT2Tokenizer.from_pretrained("/Users/tianwentang/models/gpt2")
    config = GPT2Config.from_pretrained("/Users/tianwentang/models/gpt2")
    print(tokenizer.special_tokens_map)
    # tokenizer.add_special_tokens({'pad_token': '<|PAD|>'})
    # tokenizer.add_special_tokens(['<|context|>', '<|endofcontext|>',
    #                       '<|entity1|>', '<|endofentity1|>',
    #                       '<|entity2|>', '<|endofentity2|>',
    #                       '<|notneedclarify|>', '<|needclarify|>',
    #                       '<|endofresponse|>', '<|response|>',
    #                       '<|clarifyuserinput|>', '<|endofclarifyuserinput|>'])

    dataset = TaskDataset(data, tokenizer)
    loader = get_data_loader(dataset, tokenizer, 8)
    model = GPT2LMHeadModel.from_pretrained('/Users/tianwentang/models/gpt2', config=config)
    for i in loader:
        input, label = (i, i)
        output = model(input, labels=label)
        print(output)
        break
