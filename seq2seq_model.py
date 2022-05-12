from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import BartForConditionalGeneration
from kobart import get_kobart_tokenizer, get_pytorch_kobart_model
import pandas as pd
import torch
import os
from pathlib import Path

from seq2seq_data import KUMedSeq2SeqModule
from utils import calc_metrics, get_trie_restrict_fn

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Seq2SeqModel(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.hp = hparams
        self.tokenizer = get_kobart_tokenizer(hparams['cache_path'])
        self.bart = BartForConditionalGeneration.from_pretrained(
            get_pytorch_kobart_model(cachedir=hparams['cache_path']))
        self.restrict_vocab_fn = get_trie_restrict_fn(
            self.tokenizer, hparams['train_path'])
        self.test_results = pd.DataFrame(columns=['text', 'preds', 'labels'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hp['learning_rate'])
        return optimizer

    def forward(self, inputs):
        return self.bart(**inputs)

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True,
                 on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        preds, labels = self.infer(batch)
        # print(preds)
        macro, acc = calc_metrics(preds, labels)

        self.log('valid_loss', loss, prog_bar=True, on_epoch=True)
        self.log('valid_acc', acc, prog_bar=True, on_epoch=True)
        self.log('valid_macro', macro, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        preds, labels = self.infer(batch)
        text = self.tokenizer.batch_decode(
            batch['input_ids'], skip_special_tokens=True)

        for t, p, l in zip(text, preds, labels):
            self.test_results.loc[len(self.test_results)] = [t, p, l]

    def save_test_result(self, name='test_result.csv'):
        self.test_results.to_csv(name, index=False)

    def infer_more_than_one(self, batch, has_label=True):
        res_ids = self.bart.generate(batch['input_ids'],
                                     max_length=self.hp['max_label_len'],
                                     num_beams=self.hp['num_beams'],
                                     num_return_sequences=self.hp['num_beams'],
                                     early_stopping=True,
                                     eos_token_id=self.tokenizer.eos_token_id,
                                     bad_words_ids=[[self.tokenizer.unk_token_id]])

        # (batch * num_return_sequences, max_length)
        preds = self.tokenizer.batch_decode(res_ids, skip_special_tokens=True)
        preds = [preds[i:i+self.hp['num_beams']]
                 for i in range(0, len(preds), self.hp['num_beams'])]
        if has_label:
            labels = self.tokenizer.batch_decode(
                batch['labels'], skip_special_tokens=True)
            return preds, labels
        return preds

    def infer(self, batch, has_label=True):
        res_ids = self.bart.generate(batch['input_ids'],
                                     max_length=self.hp['max_label_len'],
                                     num_beams=self.hp['num_beams'],
                                     prefix_allowed_tokens_fn=self.restrict_vocab_fn,
                                     eos_token_id=self.tokenizer.eos_token_id,
                                     bad_words_ids=[
                                         [self.tokenizer.unk_token_id]],
                                     top_k=self.hp['top_k'],
                                     early_stopping=self.hp['early_stopping'])

        preds = self.tokenizer.batch_decode(res_ids, skip_special_tokens=True)
        if has_label:
            labels = self.tokenizer.batch_decode(
                batch['labels'], skip_special_tokens=True)
            return preds, labels
        return preds

    def infer_real(self, text: str):
        tokens = self.tokenizer.encode(text)
        tokens = [self.tokenizer.bos_token_id] + \
            tokens + [self.tokenizer.eos_token_id]
        tokens = torch.unsqueeze(torch.tensor(tokens), 0)
        return self.infer({'input_ids': tokens}, has_label=False)


if __name__ == '__main__':
    seed_everything(seed=42)
    hparams = {
        'train_path': 'data/main_train.csv',
        'test_path': 'data/main_test_sample.csv',
        'max_seq_len': 128,
        'max_label_len': 7,
        'batch_size': 128,
        'learning_rate': 5e-5,
        'num_workers': 8,
        'num_beams': 5,
        'top_k': 10,
        'early_stopping': True,
    }

    train_args = {
        'gpus': [2],
        'max_epochs': 10,
        'precision': 16,
    }

    cache_path = os.path.join(Path.home(), '.cache')
    hparams['cache_path'] = cache_path

    dm = KUMedSeq2SeqModule(hparams)
    model = Seq2SeqModel(hparams)

    logger = WandbLogger(project='kumed', name='BART_seq')
    logger.experiment.config.update(hparams)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3, monitor='valid_acc', mode='max')
    trainer = Trainer(**train_args, logger=logger,
                      callbacks=[checkpoint_callback])

    trainer.fit(model, dm)
