import matplotlib.pyplot as plt


def attention_viz(coords, sigmas, size, save_file=None):
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
