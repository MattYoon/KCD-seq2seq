from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import pandas as pd
import torch
import numpy as np
from transformers import PreTrainedTokenizer


class KUMedCls(Dataset):
    def __init__(self, csv_path, tokenizer, le, max_length):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.le = le
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text, label = row['주호소 및 현병력'], row['진단코드']
        bert_in = self.tokenizer(
            text, max_length=self.max_length, padding='max_length', truncation=True)
        label = self.le.transform([label])
        return {'bert_in': bert_in.convert_to_tensors(tensor_type='pt'), 'label': label[0]}


class KUMedClsModule(LightningDataModule):
    def __init__(self, tokenizer, le, hparams):
        super().__init__()
        self.train_set = KUMedCls(
            hparams['train_path'], tokenizer, le, hparams['max_seq_len'])
        self.test_set = KUMedCls(
            hparams['test_path'], tokenizer, le, hparams['max_seq_len'])
        self.batch_size = hparams['batch_size']
        self.num_workers = hparams['num_workers']

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)
