import torch
import matplotlib.pyplot as plt


def quickplot(attentions: torch.Tensor, filename=None):
    attentions = attentions.detach().cpu()
    plt.matshow(attentions)
    if filename is not None:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
