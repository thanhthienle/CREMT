import torch
from torch.utils.data import Dataset, DataLoader


class data_set(Dataset):
    def __init__(self, data, config=None):
        self.data = data
        self.config = config
        self.bert = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], idx)

    def collate_fn(self, data):
        label = torch.tensor([item[0]["relation"] for item in data])
        tokens = [torch.tensor(item[0]["tokens"]) for item in data]
        ind = [item[1] for item in data]

        try:
            key = [torch.tensor(item[0]["key"]) for item in data]
            return (label, tokens, key, ind)
        except:
            return (label, tokens, ind)


def get_data_loader(config, data, shuffle=False, drop_last=False, batch_size=None):
    dataset = data_set(data, config)
    if batch_size == None:
        batch_size = config.batch_size
    batch_size = min(batch_size, len(data))
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=config.num_workers, collate_fn=dataset.collate_fn, drop_last=drop_last)
