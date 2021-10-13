import numpy as np
from functools import reduce
from collections.abc import Iterable
import matplotlib.pyplot as plt
from icecream import ic


def get(dic, ks):
    """
    :param dic: Potentially multi-level dictionary
    :param ks: Potentially `.`-separated keys
    """
    ks = ks.split('.')
    # print(dic, ks)
    return reduce(lambda acc, elm: acc[elm], ks, dic)


def keys(dic, prefix=''):
    """
    :return: All potentially-nested keys
    """
    def _full(k_):
        return k_ if prefix == '' else f'{prefix}.{k_}'
    for k, v in dic.items():
        if isinstance(v, dict):
            for k__ in keys(v, prefix=_full(k)):
                yield k__
        else:
            yield _full(k)


def pp(scan, n=5):
    """
    Pretty print a dictionary
    """
    for k in keys(scan):
        def _get():
            v = get(scan, k)
            return v[:n] if isinstance(v, Iterable) else v
        print(f'{k: <20} {_get()}')


def assert_arr_same(arr):
    """ Assert array contains the same elements """
    np.testing.assert_equal(arr, np.full(arr.shape, arr[0]))


def plot_1d(arr, label=None):
    plt.figure(figsize=(16, 9))
    plt.plot(np.arange(arr.size), arr, label=label, marker='o', markersize=0.3, linewidth=0.25)
    if label:
        plt.legend()
    plt.show()
