import os.path

import torch
from pathlib import Path
from omegaconf import OmegaConf
from lib.models import GeneratingTransformer
from synthetic_mask import random_sample_data2
from plot_utils import quickplot, make_gif
from utils import find_latest_model, get_model


def load_config(dip: Path):
    conf_file = dip / "config.yaml"
    cfg = OmegaConf.load(conf_file)
    return cfg


def load_model(cfg, dip: Path, model_name: str = None):
    if model_name is None:
        model_file = find_latest_model(dip)
    else:
        model_file = dip / model_name
    sd = torch.load(model_file, map_location=torch.device("cuda"))
    model = get_model(cfg)
    model.load_state_dict(sd)
    return model


def load_dir(dip: Path, model_name: str = None):
    cfg = load_config(dip)
    model = load_model(cfg, dip, model_name)
    return cfg, model


def get_native_attentions(model, seqs_inputs, attention_masks):
    attended = model.embed(seqs_inputs)
    for layer in model.transformer.layers[:-1]:
        attended = layer(attended, attention_masks)
    layer = model.transformer.layers[-1]
    attended = layer.norm1(attended)
    _, attentions = layer.self_attn(
        attended,
        attended,
        attended,
        (~(attention_masks[:, 0]).bool()),
        need_weights=True,
        average_attn_weights=False,
    )
    return attentions


def get_attentions(model, seqs_inputs, attention_masks):
    if not isinstance(model, GeneratingTransformer):
        return get_native_attentions(model, seqs_inputs, attention_masks)
    attended = model.embed(seqs_inputs)

    num_layers = len(model.t_blocks)
    b, c = seqs_inputs.shape
    num_heads = (
        model.t_blocks[0].attend.num_heads
        if hasattr(model.t_blocks[0].attend, "num_heads")
        else model.t_blocks[0].attend.n_heads
    )

    layers_attentions = torch.empty(num_layers, b, num_heads, c, c)

    for i, t_block in enumerate(model.t_blocks):
        attended, attentions = t_block.attend(
            attended, attention_masks, output_attentions=True
        )
        layers_attentions[i] = attentions

    return layers_attentions


def run_thing(cfg: OmegaConf, model):
    sample = random_sample_data2(
        10,
        cfg.experiment.context_size,
        5,
        cfg.experiment.num_classes,
        cfg.experiment.offset,
    )
    seqs_inputs, attention_masks, targets, mask = map(lambda x: x.to("cuda"), sample)
    model.eval()
    eval_logits, _ = model(seqs_inputs, attention_masks)
    eval_attentions = get_attentions(model, seqs_inputs, attention_masks)

    model.train()
    train_logits, _ = model(seqs_inputs, attention_masks)
    train_attentions = get_attentions(model, seqs_inputs, attention_masks)

    num_classes = eval_logits.size(-1)
    flat_eval_logits = eval_logits.view(-1, num_classes)
    flat_train_logits = train_logits.view(-1, num_classes)
    flat_targets = targets.view(-1)

    #  flat_mask = mask.view(-1)

    flat_mask = (seqs_inputs != 4).view(-1)
    mask_idx = (~flat_mask).nonzero().squeeze(-1)

    filtered_flat_eval_logits = flat_eval_logits[mask_idx]
    filtered_flat_train_logits = flat_train_logits[mask_idx]
    filtered_flat_targets = flat_targets[mask_idx]

    train_loss = torch.nn.functional.cross_entropy(
        filtered_flat_train_logits, filtered_flat_targets, reduction="none"
    )
    eval_loss = torch.nn.functional.cross_entropy(
        filtered_flat_eval_logits, filtered_flat_targets, reduction="none"
    )

    eval_accuracy = (
        (filtered_flat_eval_logits.argmax(-1) == filtered_flat_targets)
        .float()
        .mean()
        .item()
    )
    train_accuracy = (
        (filtered_flat_train_logits.argmax(-1) == filtered_flat_targets)
        .float()
        .mean()
        .item()
    )
    breakpoint()
    model.train()
    # attended2 = model.t_blocks[1].attend(attended1, attention_masks)
    # attended3, attentions = model.t_blocks[2].attend(attended2, attention_masks, output_attentions=True)
    return train_attentions, eval_attentions


def main():
    cfg, model = load_dir(Path("models/fixing_mask_simple_sparse"))
    run_thing(cfg, model)


def plot_attentions_over_time(dip: Path):
    cfg = load_config(dip)
    i = 0
    model_name = f"checkpoint_{i}_model.pt"
    while os.path.exists(dip / model_name):
        model = load_model(cfg, dip, model_name)
        train_attentions, eval_attentions = run_thing(cfg, model)
        quickplot(
            train_attentions[-1].mean(dim=1)[0],
            filename=dip / f"train_attentions_{i}",
            title=f"Training attention at step {i}",
        )
        quickplot(
            eval_attentions[-1].mean(dim=1)[0],
            filename=dip / f"eval_attentions_{i}",
            title=f"Validation attention at step {i}",
        )
        i += 1
        model_name = f"checkpoint_{i}_model.pt"
    make_gif(dip)


if __name__ == "__main__":
    cfg, model = load_dir(Path('models/synth_hydra_knowing_vocab_256_learned_pos_0_ngadditional_1_k'))
    run_thing(cfg, model)
    plot_attentions_over_time(
        Path("models/synth_hydra_knowing_vocab_256_learned_pos_0_ngadditional_1_k")
    )
