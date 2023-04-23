import torch
from torch import nn, Tensor
from torch.nn import functional as F
import experiment_utils as utils
import pandas as pd

device = utils.device


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


def thing(text, tokenizer):
    inp_ids = []
    attention_masks = []
    for t in text:
        encoded = tokenizer.encode(t)
        inp_ids.append(encoded.ids)
        attention_masks.append(encoded.attention_mask)
    attn_mask = torch.tensor(attention_masks, device=device)
    c = attn_mask.size(-1)
    attn_mask = attn_mask[:, None, :].expand(-1, c, -1)
    return torch.tensor(inp_ids, device=device), \
        attn_mask.bool()


def main(args):
    path = 'data/glue/cola_public/raw/in_domain_dev.tsv'
    text, labels = load_cola_dataset(path)
    tokenizer = utils.get_tokenizer(args)
    ids, attn_masks = thing(text, tokenizer)
    model = utils.get_model(args, tokenizer.get_vocab_size())
    model = GLUETuned(model, classes=1)
    optimizer, scheduler = utils.learners(model, args, load=False)
    for n in range(10):
        print(f'Training for epoch {n+1}')

        for eyeds, attn_msk, lab in zip(ids, attn_masks, labels):
            optimizer.zero_grad()
            outs = model(eyeds[None], attn_msk[None])
            loss = F.binary_cross_entropy(outs.squeeze(0), torch.tensor([lab], device=device).float())
            print(f'Loss: {loss.item()}')
            loss.backward()
            optimizer.step()
            scheduler.step()


if __name__ == '__main__':
    main(utils.parse_args())
