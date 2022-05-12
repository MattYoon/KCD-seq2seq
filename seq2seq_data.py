from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizer
from kobart import get_kobart_tokenizer


class KUMedSeq2Seq(Dataset):
    def __init__(self, csv_path, tokenizer, max_seq_len, max_label_len):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.max_seq_len = max_seq_len
        # <s>G473</s>
        self.max_label_len = max_label_len

    def __len__(self):
        return len(self.df)

    # KoBART Tokenizer가 Template Processing이 안돼 직접 추가 필요
    # https://github.com/haven-jeon/KoBART-chatbot/blob/main/kobart_chit_chat.py
    def make_input_id_mask(self, tokens, max_len):
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        if len(input_id) < max_len:
            while len(input_id) < max_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            input_id = input_id[:max_len - 1] + [
                self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:max_len]
        return input_id, attention_mask

    def __getitem__(self, index):
        row = self.df.iloc[index]
        q, a = row['주호소 및 현병력'], row['진단코드']
        q_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(q) + [self.eos_token]
        a_tokens = [self.bos_token] + \
            self.tokenizer.tokenize(a) + [self.eos_token]
        enc_input_id, enc_attention_mask = self.make_input_id_mask(
            q_tokens, self.max_seq_len)
        dec_input_id, dec_attention_mask = self.make_input_id_mask(
            q_tokens, self.max_label_len)

        # label은 eos 이후부터 max_seq_len까지 자름
        labels = self.tokenizer.convert_tokens_to_ids(
            a_tokens[1:(self.max_label_len + 1)])
        if len(labels) < self.max_label_len:
            while len(labels) < self.max_label_len:
                labels += [self.tokenizer.pad_token_id]

        return {'input_ids': np.array(enc_input_id),
                'attention_mask': np.array(enc_attention_mask),
                'decoder_input_ids': np.array(dec_input_id),
                'decoder_attention_mask': np.array(dec_attention_mask),
                'labels': np.array(labels)}


class KUMedSeq2SeqModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.tokenizer = get_kobart_tokenizer(hparams['cache_path'])
        self.train_set = KUMedSeq2Seq(
            hparams['train_path'], self.tokenizer, hparams['max_seq_len'], hparams['max_label_len'])
        self.test_set = KUMedSeq2Seq(
            hparams['test_path'], self.tokenizer, hparams['max_seq_len'], hparams['max_label_len'])
        self.batch_size = hparams['batch_size']
        self.num_workers = hparams['num_workers']

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

