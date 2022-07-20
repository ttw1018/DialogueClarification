from utils.preprocess import *
from model import get_tokenizer

model_name_or_path = './output/global_step-10000'
tokenizer = get_tokenizer(model_name_or_path)

tokenizer.add_special_tokens({'bos_token': '<|context|>'})
tokenizer.add_tokens(['<|endofcontext|>', '<|entity1|>'])
print(tokenizer.vocab_size)

tokenizer.save_pretrained("./output/test")

print(tokenizer.encode(tokenizer.bos_token))

tokens = tokenizer.tokenize("<|context|><|entity1|>This is a apple.")
print(tokens)
enc = tokenizer.encode(tokens)
print(enc)

dec = tokenizer.decode(enc)
print(dec)
