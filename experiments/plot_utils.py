from typing import Tuple
from torch import Tensor
import matplotlib.pyplot as plt


def attention_viz(coords: Tensor, sigmas: Tensor, size: Tuple, save_file: str = None):
    coords, sigmas = coords.detach().cpu(), sigmas.detach().cpu()
    h, w = size
    plt.xlim(-1, w)
    plt.ylim(-1, h)
    plt.grid(which='both', markevery=1)
    plt.scatter(coords[:, 1], coords[:, 0])
    plt.gca().invert_yaxis()
    if save_file is not None:
        plt.savefig(save_file)
    else:
        plt.show()
    plt.clf()
