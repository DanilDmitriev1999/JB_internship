import re
import numpy as np
import collections
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from pytorch_lightning import seed_everything
seed_everything(294)


class DataExplorer:
    def __init__(self,
                 model_name: str = 'distilroberta-base',
                 undersampling: bool = False) -> None:
        self.dataset = load_dataset("tweets_hate_speech_detection")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.undersampling = undersampling

    def statistic_for_data(self) -> dict:
        count = collections.Counter()
        for i in self.dataset['train']:
            label = i['label']
            count[label] += 1
        return count

    @staticmethod
    def undersampling_data(tweets: List[str],
                           labels: List[int]) -> dict:
        count = collections.Counter()
        for i in labels:
            count[i] += 1
        ration = {0: count[0] // 3, 1: count[1]}

        rus = RandomUnderSampler(random_state=294, ratio=ration)
        X_resampled, y_resampled = rus.fit_resample(tweets.reshape(-1, 1), labels.reshape(-1, 1))
        data = {'tweets': X_resampled.reshape(-1),
                'labels': y_resampled.reshape(-1)}
        return data

    @staticmethod
    def prepare_data(text):
        text = text.lower().replace('user', '')
        text = re.sub('[^\w\s]+|[\d]+', '', text)
        return text

    def class_weights(self):
        count = self.statistic_for_data()
        weights = torch.tensor([1. / i for i in count.values()], dtype=torch.float)
        return weights

    def plot_len_dist(self) -> None:
        lengths_data = [len(self.tokenizer.tokenize(self.prepare_data(i['tweet']))) for i in self.dataset['train']]

        print(f'max: {max(lengths_data)}')
        print(f'min: {min(lengths_data)}')
        print(f'median: {np.quantile(lengths_data, 0.5)}')
        print(f'90q: {np.quantile(lengths_data, 0.9)}')
        print(f'99q: {np.quantile(lengths_data, 0.99)}')

        plt.figure(figsize=(15, 4))
        sns.set(style="darkgrid")
        sns.distplot(lengths_data).set_title('Length distribution')

    def describe_data(self,
                      return_stat: bool = True,
                      plot_data_distribution: bool = True) -> None:
        if return_stat:
            count = self.statistic_for_data()
            print('Distribution by class:')
            print(count)
        if plot_data_distribution:
            self.plot_len_dist()

    def train_val_test_split(self,
                             val_size: float = 0.2,
                             test_size: float = 0.2):
        tweet = np.array([self.prepare_data(i['tweet']) for i in self.dataset['train']])
        labels = np.array([i['label'] for i in self.dataset['train']])

        X_train, X_test, y_train, y_test = train_test_split(tweet, labels,
                                                            stratify=labels,
                                                            test_size=test_size,
                                                            random_state=294)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          stratify=y_train,
                                                          test_size=val_size,
                                                          random_state=294)

        assert len(X_train) == len(y_train), print(
            f'Different data sizes: X_train: {len(X_train)}, y_train: {len(y_train)}')
        assert len(X_val) == len(y_val), print(f'Different data sizes: X_val: {len(X_val)}, y_val: {len(y_val)}')
        assert len(X_test) == len(y_test), print(f'Different data sizes: X_test: {len(X_test)}, y_test: {len(y_test)}')

        if self.undersampling:
            print('Undersampling will be used')
            train_data = self.undersampling_data(X_train, y_train)
        else:
            train_data = {'tweets': X_train,
                          'labels': y_train}

        valid_data = {'tweets': X_val,
                      'labels': y_val}

        test_data = {'tweets': X_test,
                     'labels': y_test}

        print('Train size =', len(train_data['tweets']))
        print('Val size =', len(valid_data['tweets']))
        print('Test size =', len(test_data['tweets']))

        return train_data, valid_data, test_data
