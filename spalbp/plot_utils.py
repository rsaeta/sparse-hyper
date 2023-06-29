import torch
import matplotlib.pyplot as plt
import glob
from PIL import Image
from pathlib import Path


def quickplot(attentions: torch.Tensor, filename=None, title=None):
    attentions = attentions.detach().cpu()
    plt.matshow(attentions)
    if title is not None:
        plt.title(title)
    plt.colorbar()
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def generate_gif(globs, name):
    frames = []
    for img in globs:
        frame = Image.open(img)
        frames.append(frame)
    frames[0].save(
        f"{name}_attention.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=300,
        loop=0,
    )


def sort_png(a):
    return int(a.split("_")[-1].split(".")[0])


def make_gif(dip: Path):
    training_imgs = sorted(glob.glob(str(dip / "train_attentions_*.png")), key=sort_png)
    eval_imgs = sorted(glob.glob(str(dip / "eval_attentions_*.png")), key=sort_png)
    generate_gif(training_imgs, str(dip / "train"))
    generate_gif(eval_imgs, str(dip / "eval"))
