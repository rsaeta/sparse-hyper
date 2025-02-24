import os
import torch
import wandb

import transformers
from config import RunConfig
from datasets import load_dataset
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from utils import (
    post_process_cfg,
    init_wandb,
    get_tokenizer,
    get_model,
    get_criterion,
    save_model,
    setup,
    learners,
    device,
    MaskedLMDataset,
)


cs = ConfigStore.instance()
cs.store(name='run', node=RunConfig)


def _train(cfg: RunConfig):
    setup(cfg)
    model = get_model(cfg)
    optimizer, scheduler = learners(model, cfg)
    tokenizer, teacher_model = map(lambda c: c.from_pretrained('bert-base-uncased'),
                                   (transformers.BertTokenizer, transformers.BertForMaskedLM))
    teacher_model.to(device)
    train_cfg = cfg.experiment.training

    dataset = load_dataset('enwik8').filter(lambda x: 200 < len(tokenizer.encode(x['text'])) > 300)['train']
    dataset = MaskedLMDataset(dataset, tokenizer, cfg.experiment.context_size, cfg.experiment.mask_prob)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)
    alpha = cfg.experiment.true_loss_weight
    for epoch in range(train_cfg.num_epochs):
        for i, batch in enumerate(dataloader):
            if i >= train_cfg.num_batches:
                break
            model.train()
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            teacher_logits = teacher_outputs.logits.detach()
            logits, aux_loss = model(batch['input_ids'], batch['attention_mask'].expand(-1, cfg.experiment.context_size, -1))

            assert teacher_logits.shape == logits.shape, f'{teacher_logits.shape} != {logits.shape}'

            vocab_size = teacher_logits.shape[-1]
            flattened_logits = logits.view(-1, vocab_size)
            flattened_teacher_logits = teacher_logits.view(-1, vocab_size)
            flattened_mask = batch['mask'].view(-1)
            flattened_targets = batch['targets'].view(-1)
            flattened_idx = flattened_mask.nonzero().squeeze()

            fflogits = flattened_logits[flattened_idx]
            ffteacher_logits = flattened_teacher_logits[flattened_idx]
            fftargets = flattened_targets[flattened_idx]

            kl_loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(fflogits, dim=-1),
                                                 torch.nn.functional.softmax(ffteacher_logits, dim=-1),
                                                 reduction='batchmean')

            true_loss = torch.nn.functional.cross_entropy(fflogits, fftargets, reduction='mean')
            loss = ((1-alpha)*kl_loss) + (alpha*true_loss) + aux_loss
            accuracy = (flattened_logits[flattened_idx].argmax(dim=-1) == flattened_targets[flattened_idx]).float().mean()
            to_log = {'loss': loss.item(),
                      'lr': scheduler.get_last_lr()[0],
                      'kl_loss': kl_loss.item(),
                      'aux_loss': aux_loss.item(),
                      'true_loss': true_loss.item(),
                      'accuracy': accuracy.item()}
            if 'WANDB_MODE' in os.environ:
                if i % train_cfg.log_every == 0:
                    print(f'Batch {i} : {to_log}')
            wandb.log(to_log, step=epoch*len(dataloader) + i)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clipping_value)
            optimizer.step()
            scheduler.step()
            if i % train_cfg.save_every == 0 and cfg.experiment.save_dir is not None:
                save_model(cfg, model, optimizer, scheduler, i // train_cfg.save_every)


@hydra.main(version_base=None, config_path='config', config_name='knowledge_distillation')
def main(cfg: OmegaConf):
    cfg = post_process_cfg(cfg)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    init_wandb(cfg)
    _train(cfg)


if __name__ == '__main__':
    main()
