import numpy as np


def get_hist(plt, data,
             bins=100, ran=None, xlabel='', title='', alpha=1):
    """ Make a histogram
    """
    if ran is None:
        a, b = np.histogram(data, bins=bins, density=True)
    else:
        a, b = np.histogram(data, bins, ran, density=True)

    x = (b[1:] + b[:-1])/2
    b_width = np.diff(b)[0]
    plt.bar(x, a, b_width, alpha=alpha)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title(title, fontsize=20)
