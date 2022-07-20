import os
from utils.dataset import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import pathlib
import json

def process1(file_name, turn_type='multi-turn', file_type='train'):
    file_name = file_name + f'/{turn_type}/task1/{file_type}_classifier.txt'
    sents = []
    fin = open(file_name, "r")
    for i in fin:
        dia = json.loads(i)
        sent = '<|context|> ' + dia['context'] + ' <|endofcontext|> '
        sent += '<|entity1|> ' + dia['entity1'] + ' <|endofentity1|> '
        sent += '<|entity2|> ' + dia['entity2'] + ' <|endofentity2|> '
        sent = '<|endoftext|> ' + sent + ' <|endoftext|>'
        label = ' <|notneedclarify|> ' if dia['label'] == '1' else ' <|needclarify|> '
        sents.append((sent, label))
    return sents

def process2(file_name, turn_type='multi-turn', file_type='train'):
    src_file = file_name + f'/{turn_type}/task2/src-{file_type}.txt'
    tgt_file = file_name + f'/{turn_type}/task2/tgt-{file_type}.txt'
    src_fin = open(src_file, 'r').readlines()
    tgt_fin = open(tgt_file, 'r').readlines()
    assert len(src_fin) == len(tgt_fin)
    data = []
    for i in range(len(src_fin)):
        src_list = src_fin[i].strip().split('<SP>')
        tgt = src_fin[i].strip()
        src = '<|context|> ' + src_list[0] + ' <|endofcontext|> '
        src += '<|entity1|> ' + src_list[1] + ' <|endofentity1|> '
        src += '<|entity2|> ' + src_list[2] + ' <|endofentity2|> '
        src = '<|endoftext|> ' + src + ' <|endoftext|>'

        tgt = '<|needclarify|> <|response|> ' + tgt + ' <|endofresponse|>'
        data.append((src, tgt))
    return data

def process3(file_name, turn_type='multi-turn', file_type='train'):
    file_name = file_name + f'/{turn_type}/task3/{file_type}_sml.txt'
    sents = []
    fin = open(file_name, 'r')
    for i in fin:
        dia = json.loads(i)
        contexts = dia['context'].split("<EOS>")
        if len(contexts) < 3:
            raise ValueError("dialogue turns should over 3 turns!!")
        context = ""
        for n in range(len(contexts) - 2):
            if n % 2 == 0:
                context = '<|user|> ' + contexts[n] + ' <EOS> '
            else:
                context = '<|system|> ' + contexts[n] + ' <EOS> '
        sent =  '<|context|> ' + context + ' <|endofcontext|> '
        sent += '<|entity1|> ' + dia['entity1'] + ' <|endofentity1|> '
        sent += '<|entity2|> ' + dia['entity2'] + ' <|endofentity2|> '
        sent += ' <|needclarify|> '
        sent += ' <|response|> ' + contexts[-2] + ' <|endofresponse|> '
        sent += ' <|clarifyuserinput|> ' + contexts[-1] + ' <|endofclarifyuserinput|> '
        sent = '<|endoftext|> ' + sent + ' <|endoftext|>'
        answer = '<|answer|> ' + dia['answer'] + ' <|endofanswer|>'
        sents.append((sent, answer))
    return sents


def preprocess_task1(file_type='train'):
    print('preparing data')
    data = process1('./data', turn_type='multi-turn', file_type=file_type)
    data.extend(process1('./data', turn_type='single-turn', file_type=file_type))
    data.extend(process2('./data', turn_type='multi-turn', file_type=file_type if file_type!='dev' else 'val'))
    data.extend(process2('./data', turn_type='single-turn', file_type=file_type if file_type!='dev' else 'val'))
    return data


def preprocess_task2(file_type='train'):
    sents = process3('./data', turn_type='multi-turn', file_type=file_type)
    sents.extend(process3('./data', turn_type='single-turn', file_type=file_type))
    return sents