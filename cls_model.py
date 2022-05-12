from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from transformers import AutoModel, AutoTokenizer
import torch
from torch.nn import functional as F
from torchmetrics import F1Score, Precision, Recall, Accuracy
import os
from pathlib import Path

from cls_data import KUMedClsModule
from utils import get_label_encoder


class ClsModel(LightningModule):
    def __init__(self, tokenizer, bert, class_size, hparams):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert: AutoModel = bert
        self.linear = torch.nn.Linear(768, class_size)
        self.hp = hparams
        self.f1 = F1Score(num_classes=class_size, average='macro')
        self.p = Precision(num_classes=class_size, average='macro')
        self.r = Recall(num_classes=class_size, average='macro')
        self.acc = Accuracy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hp['learning_rate'])
        return optimizer

    def forward(self, inputs):
        x = self.bert(**inputs).last_hidden_state[:, 0]
        x = self.linear(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        labels = batch['label']
        logits = self(batch['bert_in'])
        loss = F.nll_loss(logits, labels)
        self.log('train_loss', loss, prog_bar=True,
                 on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['label']
        logits = self(batch['bert_in'])
        loss = F.nll_loss(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.acc(preds, labels)
        self.f1(preds, labels)
        self.p(preds, labels)
        self.r(preds, labels)

        self.log('valid_loss', loss, prog_bar=True, on_epoch=True)
        self.log('valid_acc', self.acc, prog_bar=True, on_epoch=True)
        self.log('valid_macro_f1', self.f1, prog_bar=True, on_epoch=True)
        self.log('valid_macro_p', self.p, prog_bar=True, on_epoch=True)
        self.log('valid_macro_r', self.r, prog_bar=True, on_epoch=True)


if __name__ == '__main__':
    seed_everything(seed=42)
    hparams = {
        'train_path': 'data/main_train.csv',
        'test_path': 'data/main_test_sample.csv',
        'max_seq_len': 128,
        'batch_size': 128,
        'learning_rate': 5e-5,
        'num_workers': 8
    }

    train_args = {
        'gpus': [2],
        'max_epochs': 10,
        'precision': 16,
    }

    cachedir = os.path.join(Path.home(), '.cache')
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    le = get_label_encoder('data/main.csv')
    bert = AutoModel.from_pretrained("klue/roberta-base")

    dm = KUMedClsModule(tokenizer, le, hparams)
    model = ClsModel(tokenizer, bert, le.classes_.shape[0], hparams)

    logger = WandbLogger(project='kumed', name='RoBERTa_cls')
    logger.experiment.config.update(hparams)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3, monitor='valid_acc', mode='max')
    trainer = Trainer(**train_args, logger=logger,
                      callbacks=[checkpoint_callback])

    trainer.fit(model, dm)
