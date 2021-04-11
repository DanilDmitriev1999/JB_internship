import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CastomDataModule(Dataset):
    def __init__(self,
                 dt: dict,
                 tokenizer,
                 device: str = device) -> None:
        self.tweets = dt['tweets']
        self.labels = dt['labels']
        self.tokenizer = lambda x: tokenizer.encode_plus(x,
                                                         padding='max_length',
                                                         add_special_tokens=True,
                                                         max_length=95,
                                                         return_tensors="pt").to(device)

    @staticmethod
    def prepare_data(text):
        text = text.replace('', '')
        text = text.replace('Â', '')
        text = text.replace('ð', '')
        text = text.replace('¾', '')
        text = text.replace('â', '')
        text = text.replace('µ', '')
        text = text.replace('ï', '')
        return text

    def __len__(self) -> int:
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet_tokenizer = self.tokenizer(self.prepare_data(self.tweets[idx]))
        label = self.labels[idx]

        result = {
            'input_ids': tweet_tokenizer['input_ids'].flatten(),
            'attention_mask': tweet_tokenizer['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
        }

        return result
