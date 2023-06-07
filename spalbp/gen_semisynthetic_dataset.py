"""
This file is responsible for creating a semi-synthetic dataset for a question paired task.
It will then be saved as a csv
"""
import torch
import string
import random
from datasets import load_dataset

torch.manual_seed(42)

dataset = load_dataset('glue', 'qqp')
dataset = dataset.filter(lambda x: x['label'] == 1)
train, val, test = dataset['train'], dataset['validation'], dataset['test']


def randomize_ds(ds):
    res_ds = []
    for i in range(len(ds)):
        item = train[i]
        q1, q2 = item['question1'], item['question2']
        label = item['label']
        if torch.rand(1).item() < 0.5:  # sample a negative example
            rand_idx = torch.randint(len(train), (1,)).item()
            q2 = train[rand_idx]['question2']
            label = int(i == rand_idx)
        middle_pad_amt = torch.randint(0, 100, (1,)).item()
        middle_pad_content = ''.join(random.choices(string.ascii_letters, k=middle_pad_amt))
        text = f'{q1}{middle_pad_content}{q2}'
        res_ds.append({'text': text, 'label': label})
    return res_ds


randomized_train = randomize_ds(train)
randomized_val = randomize_ds(val)
randomized_test = randomize_ds(test)

import pandas as pd


pd_train = pd.DataFrame(randomized_train)
pd_val = pd.DataFrame(randomized_val)
pd_test = pd.DataFrame(randomized_test)

pd_train.to_csv('semi-synthetic-qqp-train.csv')
pd_val.to_csv('semi-synthetic-qqp-val.csv')
pd_test.to_csv('semi-synthetic-qqp-test.csv')
