import torch
import pytorch_lightning as pl
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score


class ModelTrainer(pl.LightningModule):
    def __init__(self,
                 model,
                 criterion,
                 lr=1e-4):
        super().__init__()

        self.model = model

        # other
        self.lr = lr

        self.res = {'prob': [], 'pred': [], 'label': []}

        self.criterion = criterion

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        return output

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=1e-8)
        return optimizer

    def training_step(self, batch, batch_idx):
        text = batch['input_ids']
        labels = batch['label']
        mask = batch['attention_mask']

        predictions = self(text, mask)
        loss = self.criterion(predictions.float(), labels.float())

        predict = torch.round(torch.sigmoid(predictions))
        predict = predict.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        acc = accuracy_score(labels, predict)
        f1 = f1_score(labels, predict)

        values = {'train_loss': loss,
                  'train_accuracy': acc,
                  'train_f1': f1}

        self.log_dict(values)

        return loss

    def validation_step(self, batch, batch_idx):
        text = batch['input_ids']
        labels = batch['label']
        mask = batch['attention_mask']

        predictions = self(text, mask)
        loss = self.criterion(predictions.float(), labels.float())

        predict = torch.round(torch.sigmoid(predictions))
        predict = predict.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        acc = accuracy_score(labels, predict)
        f1 = f1_score(labels, predict)

        values = {'val_loss': loss,
                  'val_accuracy': acc,
                  'val_f1': f1}

        self.log_dict(values)

        return loss

    def test_step(self, batch, batch_idx):
        text = batch['input_ids']
        labels = batch['label']
        mask = batch['attention_mask']

        predictions = self(text, mask)
        loss = self.criterion(predictions.float(), labels.float())

        predict = torch.round(torch.sigmoid(predictions))
        predict = predict.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

        acc = accuracy_score(labels, predict)
        f1 = f1_score(labels, predict)

        self.res['pred'].append(predict)
        self.res['label'].append(labels)
        self.res['prob'].append(predictions)

        acc = accuracy_score(labels, predict)
        f1 = f1_score(labels, predict)

        values = {'test_loss': loss,
                  'test_accuracy': acc,
                  'test_f1': f1}

        self.log_dict(values)

        return loss
