"""
This file will hold experiments around making synthetic masks for the MLM task by replacing
tokens with the [MASK] token but then placing the target outside the receptive field for a
normal convolutional model. This will allow us to see if the model can learn to use the
adaptive receptive field to learn to predict the target token.
"""
from config import RunConfig
import os
import hydra
from hydra.core.config_store import ConfigStore
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from utils import (
    setup,
    learners,
    save_model,
    init_wandb,
    post_process_cfg,
    get_model,
)
from synth_data_gen import (
    random_sample_data2, random_sample_simple
)


def _train(cfg: RunConfig):
    model = get_model(cfg)
    setup(cfg)
    optimizer, scheduler = learners(model, cfg)
    tokens_seen = 0
    train_cfg = cfg.experiment.training
    data_sample_function = (lambda: random_sample_simple(cfg.experiment.training.batch_size, cfg.experiment.context_size)) \
        if cfg.experiment.data_gen == 'simple' \
        else (lambda: random_sample_data2(cfg.experiment.training.batch_size, cfg.experiment.context_size, cfg.experiment.offset))
    if cfg.experiment.watch_model:
        wandb.watch(model)
    for i in range(train_cfg.num_batches):
        model.train()
        optimizer.zero_grad()
        breakpoint()
        data_sample = data_sample_function()
        seqs_inputs, attention_masks, targets, mask = data_sample

        logits, aux_loss = model(seqs_inputs, attention_masks)
        num_classes = logits.size(-1)
        flattened_logits = logits.view(-1, num_classes)
        flattened_targets = targets.view(-1)
        flat_mask_idx = (~mask).view(-1).nonzero().view(-1)
        loss = F.cross_entropy(flattened_logits[flat_mask_idx], flattened_targets[flat_mask_idx], reduction='mean')
        if i % train_cfg.log_every == 0:
            to_log = {'loss': loss.item(), 'tokens_seen': tokens_seen, 'lr': scheduler.get_last_lr()[0]}
            if 'WANDB_MODE' in os.environ:
                print(to_log)
            wandb.log(to_log, step=i)
        loss = loss + aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clipping_value)
        optimizer.step()
        scheduler.step()
        tokens_seen += seqs_inputs.size(0) * seqs_inputs.size(1)

        if i % train_cfg.validation_every == 0:
            model.eval()
            with torch.no_grad():
                eval_sample = random_sample_data2(
                    train_cfg.batch_size, cfg.experiment.context_size, cfg.experiment.offset
                )
                seqs_inputs, attention_masks, targets, mask = eval_sample
                logits, _ = model(seqs_inputs, attention_masks)
                num_classes = logits.size(-1)
                flattened_logits = logits.view(-1, num_classes)
                flattened_targets = targets.view(-1)
                flat_mask_idx = (~mask).view(-1).nonzero().view(-1)
                loss = F.cross_entropy(flattened_logits[flat_mask_idx], flattened_targets[flat_mask_idx], reduction='mean')
            to_log = {'val_loss': loss.item()}
            if 'WANDB_MODE' in os.environ:
                print(to_log)
            wandb.log(to_log, step=i)

        if cfg.experiment.save_dir is not None and i % train_cfg.save_every == 0:
            save_model(cfg, model, optimizer, scheduler, i // train_cfg.save_every)


cs = ConfigStore.instance()
cs.store(name='run', node=RunConfig)


@hydra.main(version_base=None, config_path='config', config_name='synthetic_mask')
def main(cfg: OmegaConf):
    cfg = post_process_cfg(cfg)
    print(OmegaConf.to_yaml(cfg, resolve=True))
    init_wandb(cfg)
    _train(cfg)


if __name__ == '__main__':
    main()
