from comet_ml import API
import torch
import argparse
import time
from models.DebertaLayerCat import *
from loss.FocalLoss import *
from utils.trainer import *
from transformers import AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def predict_class(text, tokenizer, model=None, load_model=False, device=device):
    start_time = time.time()
    if load_model:
        api = API(api_key='HWfJT3eyByVJWe4nEbi1pGosA')
        api.download_registry_model("danildmitriev1999", "deberta-jb", "1.0.0",
                                    output_path="./", expand=True)

        n_layers = [6, 7, 8]
        deberta = DebertaLayerCat('microsoft/deberta-base', n_layers)
        criterion = FocalLoss().to(device)
        model = ModelTrainer(model=deberta,
                             criterion=criterion,
                             ).to(device)
        model.load_from_checkpoint(checkpoint_path='../DeBERTa.ckpt')

    tokens = tokenizer.encode_plus(text,
                                   padding='max_length',
                                   add_special_tokens=True,
                                   max_length=95,
                                   return_tensors="pt").to(device)
    inputs = tokens['input_ids']
    mask = tokens['attention_mask']
    predictions = model(inputs, mask)

    predict = torch.round(torch.sigmoid(predictions)).detach().cpu().numpy()[0]

    print(f'Predict: {predict}, Time min: {(time.time() - start_time)/60}')


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
    parser = argparse.ArgumentParser(description='CLI for JB')
    parser.add_argument('text', type=str, help='Input string for model predict')
    text = parser.parse_args().text
    predict_class(text=text, tokenizer=tokenizer, load_model=True)
