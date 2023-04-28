import dataclasses
import os

import tokenizers.implementations
import torch
import transformers
from datasets import load_dataset
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import experiment_utils as utils
import pandas as pd
import evaluate
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer

device = utils.device


def get_berts():
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", padding='max_length')
    return bert, tokenizer


class BertTuned(nn.Module):
    def __init__(self, bert, classes=None):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        self.classification = False
        context = bert.embeddings.position_embeddings.weight.size(0)
        if classes is not None:
            self.classification = True
            # lin = nn.Linear(context * bert.encoder.layer[-1].output.dense.out_features, classes, device=device)
            lin = nn.Linear(768, classes, device=device)
            if classes == 1:
                self.final = nn.Sequential(lin, nn.Sigmoid())
            else:
                self.final = lin
        else:
            self.final = None

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        atm = attention_mask[:, None]
        b = x.size(0)
        _, pooled_output = self.bert(input_ids=x, attention_mask=attention_mask, return_dict=False)
        dropout = self.dropout(pooled_output)
        lin = self.final(dropout)
        final_layer = F.relu(lin)
        return final_layer
"""
encoded = self.bert.embeddings(x)
for layer in self.bert.encoder.layer:
    encoded, = layer(encoded, atm)
if self.final is not None:
    return self.final(encoded.view(b, -1))
return encoded
"""


@dataclasses.dataclass
class GlueOutput:
    loss: Tensor
    logits: Tensor


class GLUETuned(nn.Module):
    """
    Here we wrap our given model in another thing that fine-tunes the model
    end-to-end
    """
    def __init__(self, model, classes=None):
        super().__init__()
        self.model = model
        context = model.pos_embedding.num_embeddings
        self.classification = False
        if classes is not None:
            self.classification = True
            lin = nn.Linear(context*model.t_blocks[-1].ff[-1].out_features, classes, device=device)
            if classes == 1:
                self.final = nn.Sequential(lin, nn.Sigmoid())
            else:
                self.final = lin
        else:
            self.final = None

    def forward(self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor, labels: Tensor) -> GlueOutput:
        b, c = input_ids.shape
        embedded = self.model.embed(input_ids)
        logits = self.model.t_blocks(embedded, attention_mask)
        if self.final is not None:
            if self.classification:
                logits = logits.view(b, -1)  # Collapse embedded sentence to single dimension per item in batch
            logits = self.final(logits)
        loss = F.binary_cross_entropy(logits, labels)
        return GlueOutput(loss, logits)


def load_cola_dataset(path, sep='\t'):
    """df = pd.read_csv(path, sep=sep, names=['id', 'label', 'author_label', 'text'])
    return df.text, df.label"""
    return load_dataset('glue', 'cola')


def load_sst_dataset(dir_path, split=1):
    labels_f = os.path.join(dir_path, 'sentiment_labels.txt')
    labels = pd.read_csv(labels_f, sep='|')
    split_f = os.path.join(dir_path, 'datasetSplit.txt')
    dataset_split = pd.read_csv(split_f)
    sentences_f = os.path.join(dir_path, 'datasetSentences.txt')
    sentences = pd.read_csv(sentences_f, sep='\t')
    data = sentences.merge(labels, on='sentence_index', how='inner').merge(dataset_split, on='sentence_index')
    ds = data[data.splitset_label == split]
    return ds.sentence, ds['sentiment values']


def thing(text, tokenizer, for_bert=False):
    inp_ids = []
    attention_masks = []
    for t in text:
        encoded = tokenizer(t, padding='max_length')
        if hasattr(encoded, 'ids'):
            inp_ids.append(encoded.ids)
        else:
            inp_ids.append(encoded.input_ids)
        attention_masks.append(encoded.attention_mask)
    attn_mask = torch.tensor(attention_masks, device=device)
    c = attn_mask.size(-1)
    if not for_bert:
        attn_mask = attn_mask[:, None, :].expand(-1, c, -1)
    return torch.tensor(inp_ids, device=device), \
        attn_mask.bool()


def chunk_zip(*args, n):
    i = 0
    max_len = len(args[0])
    while i < max_len:
        end = min(i+n, max_len)
        yield (a[i:end] for a in args)
        i = end


def fine_tune_cola(model,
                   optimizer,
                   scheduler,
                   ids,
                   attention_masks,
                   labels,
                   n_epochs: int = 5):
    for i in range(n_epochs):
        for eyeds, attn_mask, lab in chunk_zip(ids, attention_masks, labels, n=5):
            optimizer.zero_grad()
            outs = model(eyeds, attn_mask)
            loss = F.binary_cross_entropy(outs.squeeze(), torch.tensor(lab.to_list(), device=device).float())
            print(f'Loss: {loss.item()}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()


class ColaDataset(Dataset):
    def __init__(self, text, labels):
        self.text = text
        self.labels = labels

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return {"text": self.text[index], "label": self.labels[index]}


def tokenizer_fn(tokenizer):
    def fn(example):
        try:
            return tokenizer(example['sentence'], padding='max_length', truncation=True, return_tensors="pt")
        except:
            derp = thing(example['sentence'], lambda t, padding: tokenizer.encode(t))
            return derp
    return fn


def train_cola(model, tokenizer, train, val):
    tokfn = tokenizer_fn(tokenizer)

    train = train.map(tokfn, batched=True)
    val = val.map(tokfn, batched=True)
    train.set_format('torch')
    val.set_format('torch')
    train, val = train.remove_columns(['sentence', 'idx']), val.remove_columns(['sentence', 'idx'])
    train, val = train.rename_column('label', 'labels'), val.rename_column('label', 'labels')

    optimizer = AdamW(model.parameters(), lr=5e-5)
    train_dl = DataLoader(train.shuffle().select(range(1000)), batch_size=10)
    val_dl = DataLoader(val.shuffle().select(range(1000)), batch_size=10)
    for n in range(10):
        model.train()
        for i, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if i % 10 == 0:
                print(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        metric = evaluate.load("accuracy")
        model.eval()
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch['labels'])
        print(metric.compute())


def main_bert():
    path = 'data/glue/cola_public/raw/in_domain_dev.tsv'
    ds = load_cola_dataset(path)
    train, val = ds['train'], ds['validation']
    model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_cola(model, tokenizer, train, val)


def mainish(args):
    breakpoint()
    ds = load_cola_dataset(None)
    train, val = ds['train'], ds['validation']
    tokenizer = utils.get_tokenizer(args)
    model = utils.get_model(args, tokenizer.get_vocab_size())
    train_cola(model, tokenizer, train, val)


def main(args):
    path = 'data/glue/cola_public/raw/in_domain_dev.tsv'
    text, labels = load_cola_dataset(path)
    model, tokenizer = get_berts()
    # tokenizer = utils.get_tokenizer(args)
    ids, attn_masks = thing(text, tokenizer)
    model = BertTuned(model, classes=1)
    # model = utils.get_model(args, tokenizer.get_vocab_size())
    # model = GLUETuned(model, classes=1)
    optimizer, scheduler = utils.learners(model, args, load=False)
    fine_tune_cola(model, optimizer, scheduler, ids, attn_masks, labels, n_epochs=10)


if __name__ == '__main__':
    # main_bert()
    # main(utils.parse_args())
    mainish(utils.parse_args())
