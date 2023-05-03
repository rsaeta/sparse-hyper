import dataclasses

from tqdm import tqdm
import wandb
import torch
from datasets import load_dataset
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import experiment_utils as utils
import evaluate
from transformers import BertForSequenceClassification, BertTokenizer

device = utils.device


def init_wandb(args):
    wandb.init(
        project='model-finetuning',
        config={
            'model': args.load_model,
            'dataset': args.finetune_ds,
        }
    )


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
    def __init__(self, model, classes=None, activate=True, loss_fn=None):
        super().__init__()
        self.model = model
        self.classification = False
        self.loss_fn = loss_fn
        if classes is not None:
            self.classification = True
            lin = nn.Linear(model.t_blocks[-1].ff[-1].out_features, classes, device=device)
            if classes == 1:
                if activate:
                    self.final = nn.Sequential(lin, nn.Sigmoid())
                else:
                    self.final = lin
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
            loss = self.loss_fn(logits.squeeze(1), labels.float())
        else:
            loss = None
        return GlueOutput(loss, logits)


def load_and_process(tokenizer, *args, batch_size=32, subset=1000):
    """
    Loads the dataset and does the label replacement for the GLUE datasets.
    Returns a tuple of train and validation datasets.
    """
    dataset = load_dataset(*args)
    valkey = 'validation_matched' if 'validation_matched' in dataset else 'validation'
    train, val = dataset['train'], dataset[valkey]
    if subset is not None:
        subset = min(subset, len(train), len(val))
        train = train.select(range(subset))
        val = val.select(range(subset))
    tokfn = tokenizer_fn(tokenizer)

    train = train.map(tokfn, batched=True)
    val = val.map(tokfn, batched=True)
    train.set_format('torch')
    val.set_format('torch')

    training_cols = ['input_ids', 'attention_mask', 'label']
    cols_to_remove = [c for c in train.features if c not in training_cols]
    train = train.remove_columns(cols_to_remove)
    val = val.remove_columns(cols_to_remove)

    train, val = train.rename_column('label', 'labels'), val.rename_column('label', 'labels')

    train_dl = DataLoader(train.shuffle(), batch_size=batch_size)
    val_dl = DataLoader(val.shuffle(), batch_size=batch_size)

    return train_dl, val_dl


def sample_single_sentence(text, tokenizer, for_bert=False):
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


def sample_sentence_pair(text1, text2, tokenizer=None, for_bert=False):
    inp_ids = []
    attention_masks = []
    for t1, t2 in zip(text1, text2):
        encoded = tokenizer(t1, t2, padding='max_length')
        if hasattr(encoded, 'ids'):
            inp_ids.append(encoded.ids[0:256])
        else:
            inp_ids.append(encoded.input_ids[0:256])
        attention_masks.append(encoded.attention_mask[0:256])
    attn_mask = torch.tensor(attention_masks, device=device)
    c = attn_mask.size(-1)
    if not for_bert:
        attn_mask = attn_mask[:, None, :].expand(-1, c, -1)
    return torch.tensor(inp_ids, device=device), \
        attn_mask.bool()


def tokenizer_fn(tokenizer):
    def fn(example):
        columns = [c for c in example.keys() if c not in ['idx', 'label']]
        if len(columns) > 1:
            sample = sample_sentence_pair(example[columns[0]], example[columns[1]], tokenizer=lambda t1, t2, padding: tokenizer.encode(t1, t2))
            seq_ids, attn_masks = sample
            return {'input_ids': seq_ids, 'attention_mask': attn_masks}
        col = columns[0]
        try:
            return tokenizer(example[col], padding='max_length', truncation=True, return_tensors="pt")
        except:
            pass
        sample = sample_single_sentence(example[col], tokenizer=lambda t, padding: tokenizer.encode(t))
        seq_ids, attn_masks = sample
        return {'input_ids': seq_ids, 'attention_mask': attn_masks}
    return fn


def _train(model, optimizer, scheduler, train_dl, val_dl, criterion):
    for n in range(5):
        model.train()
        for i, batch in tqdm(enumerate(train_dl)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if i % 10 == 0:
                wandb.log({'loss': loss.item()})
                print(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        metric = evaluate.load(criterion)
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
        res = metric.compute()
        wandb.log({'criterion': res})
        print(res)


def bert_main(args):
    model, tokenizer = get_berts()
    train_dl, val_dl = load_and_process(tokenizer, 'glue', 'sst2', batch_size=args.batch_size)
    optimizer, scheduler = utils.learners(model, args, load=False)
    _train(model, optimizer, scheduler, train_dl, val_dl)


def main(args):
    finetune_ds = args.finetune_ds.split(',')
    assert len(finetune_ds) == 2, 'Must specify a GLUE dataset with two parts'
    tokenizer = utils.get_tokenizer(args)
    train_dl, val_dl = load_and_process(tokenizer, finetune_ds[0], finetune_ds[1])
    model = utils.get_model(args, tokenizer.get_vocab_size())

    # Regression
    if train_dl.dataset.features['labels'].dtype == 'float32':
        classes = 1
        activate = False
        loss_fn = F.mse_loss
        criterion = 'mse'
    # Classification
    else:
        classes = train_dl.dataset.features['labels'].num_classes
        activate = True
        loss_fn = F.binary_cross_entropy
        criterion = 'accuracy'
    model = GLUETuned(model, classes=classes, activate=activate, loss_fn=loss_fn)
    optimizer, scheduler = utils.learners(model, args, load=False)
    _train(model, optimizer, scheduler, train_dl, val_dl, criterion)


if __name__ == '__main__':
    main(utils.parse_args())
