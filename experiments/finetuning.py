import os

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import experiment_utils as utils
import pandas as pd

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
        self.classification = False
        context = bert.embeddings.position_embeddings.weight.size(0)

        if classes is not None:
            self.classification = True
            lin = nn.Linear(context * bert.encoder.layer[-1].output.dense.out_features, classes, device=device)
            if classes == 1:
                self.final = nn.Sequential(lin, nn.Sigmoid())
            else:
                self.final = lin
        else:
            self.final = None

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        atm = attention_mask[:, None]
        b = x.size(0)
        encoded = self.bert.embeddings(x)
        for layer in self.bert.encoder.layer:
            encoded, = layer(encoded, atm)
        if self.final is not None:
            return self.final(encoded.view(b, -1))
        return encoded


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

    def forward(self, x: Tensor, attn_masks: Tensor):
        b, c = x.shape
        embedded = self.model.embed(x)
        logits = self.model.t_blocks(embedded, attn_masks)
        if self.final is not None:
            if self.classification:
                logits = logits.view(b, -1)  # Collapse embedded sentence to single dimension per item in batch
            logits = self.final(logits)
        return logits


def load_cola_dataset(path, sep='\t'):
    df = pd.read_csv(path, sep=sep, names=['id', 'label', 'author_label', 'text'])
    return df.text, df.label


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
    main(utils.parse_args())
