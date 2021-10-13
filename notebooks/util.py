import numpy as np
import json
import subprocess
from functools import reduce
from collections.abc import Iterable
import matplotlib.pyplot as plt
import seaborn as sns
from icecream import ic

sns.set_style('dark')


def get_scans(fnm):
    f = open(fnm, 'r')
    scans = json.load(f)
    f.close()
    return scans


def sys_out(cmd):
    return subprocess.getoutput(cmd)


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
    plt.plot(np.arange(arr.size), arr, label=label, marker='o', ms=0.3, lw=0.25)
    if label:
        plt.legend()
    plt.show()


def plot_laser(scan, title=None, save=False):
    a_max, a_min = scan['angle_max'], scan['angle_min']
    n_beams = round(((a_max - a_min) / scan['angle_increment']) + 1)
    theta = np.linspace(a_min, a_max, num=n_beams)

    plt.figure(figsize=(16, 9))
    ax = plt.subplot(1, 1, 1, polar=True)
    ax.plot(theta, np.array(scan['ranges']), marker='o', ms=0.3, lw=0.25)
    ax.plot(0, 0, marker='o', ms=4)
    ax.set_theta_offset(-np.pi / 2.0)

    t = 'Laser scan range plot'
    if title:
        t = f'{t}, {title}'
    plt.title(t)
    if save:
        plt.savefig(f'{title}.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    import os
    os.chdir('../data')

    fnms = sys_out('ls').split('\n')
    s = get_scans(fnms[0])[0]
    plot_laser(s, title='HSR laser scan', save=True)
