import torch
from pathlib import Path
from omegaconf import OmegaConf
from lib.models import GeneratingTransformer
from synthetic_mask import random_sample_data2


def interact(dir: Path):
    conf_file = dir / 'config.yaml'
    model_file = dir / 'model.pt'
    cfg = OmegaConf.load(conf_file)
    sd = torch.load(model_file, map_location=torch.device('cuda'))

    model = GeneratingTransformer(cfg.model)
    model.load_state_dict(sd)
    model.to('cuda')

    sample = random_sample_data2(10, cfg.experiment.context_size, cfg.experiment.offset)
    seqs_inputs, attention_masks, targets, mask = map(lambda x: x.to('cuda'), sample)

    model.eval()
    eval_logits, _ = model(seqs_inputs, attention_masks)

    model.train()
    train_logits, _ = model(seqs_inputs, attention_masks)

    num_classes = eval_logits.size(-1)
    eval_logits = eval_logits.view(-1, num_classes)
    train_logits = train_logits.view(-1, num_classes)
    targets = targets.view(-1)
    mask = mask.view(-1)
    mask_idx = (~mask).nonzero().squeeze(-1)

    eval_logits = eval_logits[mask_idx]
    train_logits = train_logits[mask_idx]
    targets = targets[mask_idx]

    eval_accuracy = (eval_logits.argmax(-1) == targets).float().mean().item()
    train_accuracy = (train_logits.argmax(-1) == targets).float().mean().item()

    breakpoint()





def main():
    interact(Path('models/foobar_lets_go_k_1_big_g'))


if __name__ == '__main__':
    main()