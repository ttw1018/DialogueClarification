from random import shuffle
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch


class TaskDataset(Dataset):
    def __init__(self, data, tokenizer):
        super(TaskDataset, self).__init__()
        input = [i[0] for i in data]
        label = [i[1] for i in data]
        self.input = tokenizer.batch_encode_plus(input, max_length=1024, truncation=True)['input_ids']
        self.label = tokenizer.batch_encode_plus(label, max_length=1024, truncation=True)['input_ids']

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        return torch.tensor(self.input[item]), torch.tensor(self.label[item])


def get_data_loader(dataset, tokenizer, batch_size):
    def collate_fn(data):
        input = [i[0] for i in data]
        label = [i[1] for i in data]
        return pad_sequence(input, batch_first=True), pad_sequence(label, batch_first=True)

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
