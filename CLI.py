from models.DebertaLayerCat import *
from models.Roberta import *

from comet_ml import API
from loss.FocalLoss import *
from utils.trainer import *


def predict_class(text, tokenizer, model, load_model=False, device=device):
    if load_model:
        api = API(api_key='HWfJT3eyByVJWe4nEbi1pGosA')
        api.download_registry_model("danildmitriev1999", "deberta-sota", "1.0.0",
                                    output_path="./", expand=True)

        criterion = FocalLoss().to(device)
        model = ModelTrainer(model=model,
                             criterion=criterion,
                             ).to(device)
        model.load_from_checkpoint(checkpoint_path='/content/distilRoBERTa.ckpt')

    tokens = tokenizer.encode_plus(text,
                                   padding='max_length',
                                   add_special_tokens=True,
                                   max_length=95,
                                   return_tensors="pt").to(device)
    inputs = tokens['input_ids']
    mask = tokens['attention_mask']
    predictions = model(inputs, mask)

    predict = torch.round(torch.sigmoid(predictions)).detach().cpu().numpy()[0]

    return predict


if __name__ == '__main__':
    pass