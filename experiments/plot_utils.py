from typing import Tuple
from torch import Tensor
import matplotlib.pyplot as plt


def attention_viz(coords: Tensor, sigmas: Tensor, size: Tuple, save_file: str = None):
    coords, sigmas = coords.cpu(), sigmas.cpu()
    h, w = size
    plt.xlim(0, w)
    plt.ylim(0, h)
    for c, s in zip(coords, sigmas):
        circ = plt.Circle(tuple(c), s)
        plt.gca().add_patch(circ)
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()
