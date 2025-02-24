from datasets import load_dataset
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import wandb

from config import RunConfig
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from utils import (
    post_process_cfg,
    setup,
    device,
    learners,
    save_model,
    init_wandb,
    get_tokenizer,
    get_model,
    get_criterion,
)


class GlueDataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 tokenizer,
                 seq_len,
                 negative_rate=0.5,
                 subset=200):
        dataset = load_dataset('glue', dataset_name).filter(lambda i: i['label'] == 1)  # Only load positive examples
        self.train, self.val, self.test = dataset['train'], dataset['validation'], dataset['test']
        if subset:
            self.train = self.train.select(range(subset))
            self.val = self.val.select(range(subset))
            self.test = self.test.select(range(subset))
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.negative_rate = negative_rate

    def __getitem__(self, idx):
        item = self.train[idx]
        pad_token = self.tokenizer.token_to_id('[PAD]')
        q1, q2 = item['question1'], item['question2']
        if torch.rand(1).item() < self.negative_rate:  # sample a negative example
            rand_idx = torch.randint(len(self.train), (1,)).item()
            q2 = self.train[rand_idx]['question2']
            label = int(idx == rand_idx)
        else:
            label = item['label']  # Should always be 1
        q1, q2 = self.tokenizer.encode(q1), self.tokenizer.encode(q2)
        if pad_token not in q2.ids:
            q2_len = self.seq_len - 1
        else:
            q2_len = q2.ids.index(pad_token) - 1  # ignore starting [CLS] token
        q2_start = len(q1.ids) - q2_len + 1
        ids = torch.tensor(q1.ids)[:self.seq_len]
        ids[q2_start:] = torch.tensor(q2.ids[1:q2_len])
        attn_mask = torch.tensor(q1.attention_mask)[:self.seq_len]
        attn_mask[q2_start:] = torch.tensor(q2.attention_mask[1:q2_len])

        # Expand the attention mask to a symmetric matrix
        attn_mask = attn_mask[:, None].expand(-1, self.seq_len)
        attn_mask = torch.ones_like(attn_mask)
        label = F.one_hot(torch.tensor(label), num_classes=2).float()
        return {'input_ids': ids,
                'attention_mask': attn_mask,
                'labels': label}

    def __len__(self):
        return len(self.train)


def _train(cfg: RunConfig):
    setup(cfg)
    tokenizer = get_tokenizer(cfg)
    model = get_model(cfg)
    optimizer, scheduler = learners(model, cfg)
    criterion = get_criterion(cfg)
    train_cfg = cfg.experiment.training
    dataset = GlueDataset(cfg.experiment.data.name, tokenizer, cfg.experiment.context_size)
    dataloader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)
    tokens_seen = 0
    i = 0
    while i < train_cfg.num_batches:
        for batch in dataloader:
            i += 1
            if i >= train_cfg.num_batches:
                break
            model.train()
            optimizer.zero_grad()

            batch = {k: v.to(device) for k, v in batch.items()}

            tokens_seen += batch['input_ids'].numel()
            outputs = model(batch['input_ids'], batch['attention_mask'])

            loss = criterion(outputs, batch['labels'])
            accuracy = (outputs.argmax(dim=-1) == batch['labels'].argmax(dim=-1)).float().mean()
            if i % train_cfg.log_every == 0:
                to_log = {'loss': loss.item(),
                          'tokens_seen': tokens_seen,
                          'lr': scheduler.get_last_lr()[0],
                          'accuracy': accuracy.item()}
                if 'WANDB_MODE' in os.environ:
                    print(f'Batch {i} : {to_log}')
                wandb.log(to_log, step=i)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clipping_value)
            optimizer.step()
            scheduler.step()
            if i % train_cfg.save_every == 0 and cfg.experiment.save_dir is not None:
                save_model(cfg, model, optimizer, scheduler, i // train_cfg.save_every)


cs = ConfigStore.instance()
cs.store(name='run', node=RunConfig)


@hydra.main(version_base=None, config_path='config', config_name='synthetic_qa')
def main(cfg: OmegaConf):
    cfg = post_process_cfg(cfg)
    init_wandb(cfg)
    _train(cfg)


if __name__ == '__main__':
    main()
