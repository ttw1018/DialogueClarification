from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch


class TaskDataset(Dataset):
    def __init__(self, data, tokenizer):
        super(TaskDataset, self).__init__()
        self.data = tokenizer.batch_encode_plus(data, max_length=1024, truncation=True)['input_ids']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return torch.tensor(self.data[item])


def get_data_loader(dataset, tokenizer, batch_size):
    def collate_fn(data):
        return pad_sequence(data, batch_first=True)

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
