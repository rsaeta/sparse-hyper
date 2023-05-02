import dataclasses

from tqdm import tqdm
import torch
from datasets import load_dataset
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import experiment_utils as utils
import evaluate
from transformers import BertForSequenceClassification, BertTokenizer

device = utils.device


def get_berts():
    bert = BertForSequenceClassification.from_pretrained("bert-base-uncased").to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", padding='max_length')
    return bert, tokenizer


class BertTuned(nn.Module):
    def __init__(self, bert, classes=None):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        self.classification = False
        if classes is not None:
            self.classification = True
            lin = nn.Linear(768, classes, device=device)
            if classes == 1:
                self.final = nn.Sequential(lin, nn.Sigmoid())
            else:
                self.final = lin
        else:
            self.final = None

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        _, pooled_output = self.bert(input_ids=x, attention_mask=attention_mask, return_dict=False)
        dropout = self.dropout(pooled_output)
        lin = self.final(dropout)
        final_layer = F.relu(lin)
        return final_layer


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
        self.classification = False
        if classes is not None:
            self.classification = True
            lin = nn.Linear(model.t_blocks[-1].ff[-1].out_features, classes, device=device)
            if classes == 1:
                self.final = nn.Sequential(lin, nn.Sigmoid())
            else:
                self.final = lin
        else:
            self.final = None

    def forward(self, input_ids: Tensor, *, attention_mask: Tensor, labels: Tensor = None) -> GlueOutput:
        b, c = input_ids.shape
        embedded = self.model.embed(input_ids)
        logits = self.model.t_blocks(embedded, attention_mask)
        if self.final is not None:
            if self.classification:
                logits = logits[:, 0, :]  # take the first token of the sequence as the pooled output ([CLS])
            logits = self.final(logits)
        if labels is not None:
            loss = F.binary_cross_entropy(logits.squeeze(1), labels.float())
        else:
            loss = None
        return GlueOutput(loss, logits)


def load_and_process(tokenizer, *args, batch_size=32):
    """
    Loads the dataset and does the label replacement for the GLUE datasets.
    Returns a tuple of train and validation datasets.
    """
    dataset = load_dataset(*args)
    train, val = dataset['train'], dataset['validation']
    tokfn = tokenizer_fn(tokenizer)

    train = train.map(tokfn, batched=True)
    val = val.map(tokfn, batched=True)
    train.set_format('torch')
    val.set_format('torch')
    train, val = train.remove_columns(['sentence', 'idx']), val.remove_columns(['sentence', 'idx'])
    train, val = train.rename_column('label', 'labels'), val.rename_column('label', 'labels')

    train_dl = DataLoader(train.shuffle(), batch_size=batch_size)
    val_dl = DataLoader(val.shuffle(), batch_size=batch_size)

    return train_dl, val_dl


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


def thing2(text1, text2, tokenizer=None, for_bert=False):
    inp_ids = []
    attention_masks = []
    for t1, t2 in zip(text1, text2):
        encoded = tokenizer(t1, t2, padding='max_length')
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


def tokenizer_fn(tokenizer):
    def fn(example):
        if 'question1' in example:
            derp = thing2(example['question1'], example['question2'], tokenizer=lambda t1, t2, padding: tokenizer.encode(t1, t2))
            seq_ids, attn_masks = derp
            return {'input_ids': seq_ids, 'attention_mask': attn_masks}
        try:
            return tokenizer(example['sentence'], padding='max_length', truncation=True, return_tensors="pt")
        except:
            pass
        derp = thing(example['sentence'], tokenizer=lambda t, padding: tokenizer.encode(t))
        seq_ids, attn_masks = derp
        return {'input_ids': seq_ids, 'attention_mask': attn_masks}
    return fn


def _train(model, optimizer, scheduler, train_dl, val_dl):
    for n in range(5):
        model.train()
        for i, batch in tqdm(enumerate(train_dl)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if i % 10 == 0:
                print(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        metric = evaluate.load("accuracy")
        model.eval()
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            if logits.size(-1) > 1:
                predictions = torch.argmax(logits, dim=-1)
            else:
                predictions = torch.round(logits)
            metric.add_batch(predictions=predictions, references=batch['labels'])
        print(metric.compute())


def bert_main(args):
    model, tokenizer = get_berts()
    train_dl, val_dl = load_and_process(tokenizer, 'glue', 'sst2', batch_size=args.batch_size)
    optimizer, scheduler = utils.learners(model, args, load=False)
    _train(model, optimizer, scheduler, train_dl, val_dl)


def main(args):
    finetune_ds = args.finetune_ds.split(',')
    assert len(finetune_ds) == 2, 'Must specify a GLUE dataset with two parts'
    tokenizer = utils.get_tokenizer(args)
    model = utils.get_model(args, tokenizer.get_vocab_size())
    model = GLUETuned(model, classes=1)
    optimizer, scheduler = utils.learners(model, args)
    train_dl, val_dl = load_and_process(tokenizer, finetune_ds[0], finetune_ds[1])
    _train(model, optimizer, scheduler, train_dl, val_dl)


if __name__ == '__main__':
    """tokenizer = tokenizers.BertWordPieceTokenizer.from_file('tokenizers/wordpiece_enwik8.txt')
    tokenizer.enable_padding(length=256)
    ds = load_and_process(tokenizer, 'glue', 'sst2')
    breakpoint()"""
    bert_main(utils.parse_args())
