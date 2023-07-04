import os.path

import torch
from pathlib import Path
from omegaconf import OmegaConf
from lib.models import GeneratingTransformer
from synthetic_mask import random_sample_data2
from synthetic_mask_all import random_sample_data
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
    b, c = seqs_inputs.shape
    num_layers = len(model.transformer.layers)
    num_heads = model.transformer.layers[0].self_attn.num_heads
    attentions = torch.empty(num_layers, b, num_heads, c, c)
    attended = model.embed(seqs_inputs)
    for i, layer in enumerate(model.transformer.layers):
        attended, attention = layer.self_attn(
            attended,
            attended,
            attended,
            (~(attention_masks[:, 0]).bool()),
            need_weights=True,
            average_attn_weights=False,
        )
        attended = layer.norm1(attended)
        attentions[i] = attention
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


def run_thing_dict(cfg: OmegaConf, model, sample_method=random_sample_data2):
    sample = sample_method(
        10,
        cfg.experiment.context_size,
        5,
        cfg.experiment.num_classes,
        cfg.experiment.offset,
    )
    if len(sample) == 4:
        seqs_inputs, attention_masks, targets, mask = map(lambda x: x.to("cuda"), sample)
    else:
        seqs_inputs, attention_masks, targets = map(lambda x: x.to("cuda"), sample)
        mask = torch.zeros_like(seqs_inputs)
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

    model.train()

    return {
        "seqs_inputs": seqs_inputs,
        "attention_masks": attention_masks,
        "targets": targets,
        "mask": mask,
        "eval_logits": eval_logits,
        "eval_attentions": eval_attentions,
        "train_logits": train_logits,
        "train_attentions": train_attentions,
        "filtered_flat_targets": filtered_flat_targets,
        "filtered_flat_eval_logits": filtered_flat_eval_logits,
        "filtered_flat_train_logits": filtered_flat_train_logits,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "eval_accuracy": eval_accuracy,
        "train_accuracy": train_accuracy,
    }

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

    model.train()
    return train_attentions, eval_attentions


def main():
    cfg, model = load_dir(Path("models/synth_hydra_knowing_lr_1e-3_vocab_256_learned_pos_k_2"))
    run_thing(cfg, model)


def get_sigmas(dip: Path):
    cfg = load_config(dip)
    i = 0
    model_name = f"checkpoint_{i}_model.pt"
    sigmas_per_models = []
    iternum = 1
    while os.path.exists(dip / model_name):
        model = load_model(cfg, dip, model_name)
        sigs = model.t_blocks[0].attend.psigmas
        sigmas_per_models.append(sigs[None])
        i += iternum
        model_name = f"checkpoint_{i}_model.pt"
    return torch.cat(sigmas_per_models)


def plot_attentions_over_time(dip: Path):
    cfg = load_config(dip)
    i = 0
    model_name = f"checkpoint_{i}_model.pt"
    sample_method = random_sample_data if "_all_" in cfg.experiment.save_dir else random_sample_data2
    iternum = 1
    while os.path.exists(dip / model_name):
        if os.path.exists(dip / f"train_attentions_{i}.png"):
            i += iternum
            continue
        model = load_model(cfg, dip, model_name)
        d = run_thing_dict(cfg, model, sample_method=sample_method)
        train_attentions = d["train_attentions"].cpu()
        eval_attentions = d["eval_attentions"].cpu()
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
        i += iternum
        model_name = f"checkpoint_{i}_model.pt"
    make_gif(dip)


def find_anomaly(cfg, model):
    i = 0
    while True:
        i += 1
        if i % 100 == 0:
            print(f"Attempt {i}")
        d = run_thing_dict(cfg, model)
        if d["train_loss"].max() > 0.3:
            breakpoint()


if __name__ == "__main__":
    sigmas = get_sigmas(Path("models/synth_mask_all_simple_sparse_2_k_learned_pos_fix_densities_sum"))
    breakpoint()
    main()
